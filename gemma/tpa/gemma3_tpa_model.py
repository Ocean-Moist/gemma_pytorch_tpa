# File: gemma3_isp_kv_model.py
"""
Inference-only Gemma model implementation with Interaction Subspace Projection KV (ISP-KV) Attention.
"""
import gc
import json
import os
from typing import List, Tuple, Union, Mapping, Optional, Sequence, Any

import torch
import torch.nn.functional as F
from torch import nn

from .. import config as gemma_config
from .. import model as gemma_model  # For helpers like RMSNorm, Embedding, MLP, RoPE, Sampler etc.
from .. import tokenizer
from ..model import RMSNorm, GemmaMLP, Embedding, Sampler, precompute_freqs_cis, apply_rotary_emb


class ISP_KVAttention(nn.Module):
    """
    Interaction Subspace Projection Key-Value (ISP-KV) Attention.

    Computes full Q/K/V, projects K and V onto pre-computed low-rank bases
    derived from QK-interaction and V-output subspace respectively. Caches
    these projections (pk, pv). Reconstructs K/V from cache during attention,
    applies RoPE post-reconstruction, and performs standard attention.
    """

    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.config = config
        self.attn_type = attn_type # For potential future use (e.g., sliding window mask)

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim # Assume Dq=Dk=Dv=head_dim

        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        self.heads_per_group = self.num_heads // self.num_kv_heads

        # --- Retrieve ISP-KV Ranks (Set during conversion/loading) ---
        # r_k: Rank for the key interaction subspace (per query head)
        # r_v: Rank for the value output subspace (per value group)
        self.r_k = getattr(config, 'r_k', 16) # Example default, should come from conversion
        self.r_v = getattr(config, 'r_v', 16) # Example default

        # --- Original Projection Layers (Weights loaded from GQA model) ---
        # Option 1: Keep combined QKV projection
        # self.qkv_proj = gemma_model.Linear(
        #     self.hidden_size,
        #     (self.num_heads + 2 * self.num_kv_heads) * self.head_dim,
        #     quant=config.quant)
        # Option 2: Use separate Q, K, V layers (cleaner for ISP-KV logic)
        self.W_q = gemma_model.Linear(self.hidden_size, self.num_heads * self.head_dim, config.quant)
        self.W_k = gemma_model.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, config.quant)
        self.W_v = gemma_model.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, config.quant)

        # --- Output Projection (Weight loaded from GQA model) ---
        self.o_proj = gemma_model.Linear(
            self.num_heads * self.head_dim, self.hidden_size, quant=config.quant
        )

        # --- ISP-KV Basis Buffers (Loaded from conversion results) ---
        # V_r: Basis for projecting K, derived from SVD(Wq^T Wk). Per Query Head.
        self.register_buffer('V_r_basis', torch.zeros(self.num_heads, self.head_dim, self.r_k))
        # Z_v: Basis for projecting V, derived from Left Singular Vectors of Wv. Per KV Group.
        self.register_buffer('Z_v_basis', torch.zeros(self.num_kv_heads, self.head_dim, self.r_v))
        self.k_head_dim = config.head_dim
        self.v_head_dim = config.head_dim
        # --- Optional QK Norm ---
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if getattr(config, 'use_qk_norm', False)
            else None
        )
        self.key_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if getattr(config, 'use_qk_norm', False)
            else None
        )

        # --- Scaling ---
        if getattr(config, 'query_pre_attn_scalar', None) is not None:
            self.scaling = (config.query_pre_attn_scalar) ** -0.5
        else:
            self.scaling = self.head_dim ** -0.5

        # --- ISP-KV Cache (pk, pv projections ONLY) ---
        self.cache_pk: Optional[torch.Tensor] = None # Shape: [B, T, N_h, r_k]
        self.cache_pv: Optional[torch.Tensor] = None # Shape: [B, T, N_kv, r_v]

    def _init_kv_cache(self, batch_size: int, max_seq_len: int, dtype: torch.dtype, device: torch.device):
        """Initialize factorized KV cache storing only pk and pv projections."""
        self.cache_pk = torch.zeros((batch_size, max_seq_len, self.num_heads, self.r_k),
                                    device=device, dtype=dtype)
        self.cache_pv = torch.zeros((batch_size, max_seq_len, self.num_kv_heads, self.r_v),
                                    device=device, dtype=dtype)
        print(f"Initialized ISP-KV Cache (Projections only): pk={self.cache_pk.shape}, pv={self.cache_pv.shape}")


    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: torch.Tensor, # RoPE freqs for Q/K head dim (absolute positions)
            kv_write_indices: torch.Tensor, # Indices for THIS step's cache write
            kv_cache: Optional[Tuple] = None, # Ignored by ISP_KV layers
            mask: Optional[torch.Tensor] = None, # Causal mask
            local_mask: Optional[torch.Tensor] = None, # Optional sliding window mask
    ) -> torch.Tensor:
        bsz, q_seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # --- 1. Compute Full Q, K, V ---
        q = self.W_q(hidden_states).view(bsz, q_seq_len, self.num_heads, self.head_dim)
        k = self.W_k(hidden_states).view(bsz, q_seq_len, self.num_kv_heads, self.head_dim)
        v = self.W_v(hidden_states).view(bsz, q_seq_len, self.num_kv_heads, self.head_dim)

        # --- 2. Project K -> pk ---
        # V_r_basis has shape [N_h, Dk, r_k]. K has shape [b, q_s, N_kv, Dk].
        # We need to project K for EACH query head using ITS corresponding V_r basis.
        pk_list = []
        for h in range(self.num_heads):
            g = h // self.heads_per_group # Corresponding KV group index
            k_group = k[:, :, g, :] # Select K for the group: [b, q_s, Dk]
            V_r_h = self.V_r_basis[h, :, :] # Select basis for head h: [Dk, r_k]
            # Project: einsum('bqd,dr->bqr', k_group, V_r_h)
            pk_h = torch.matmul(k_group, V_r_h) # Shape: [b, q_s, r_k]
            pk_list.append(pk_h)
        pk = torch.stack(pk_list, dim=2) # Shape: [b, q_s, N_h, r_k]

        # --- 3. Project V -> pv ---
        # Z_v_basis has shape [N_kv, Dv, r_v]. V has shape [b, q_s, N_kv, Dv].
        # Project using einsum for clarity: 'bqgd,gdr->bqgr'
        pv = torch.einsum('bqgd,gdr->bqgr', v, self.Z_v_basis) # Shape: [b, q_s, N_kv, r_v]

        # --- 4. KV Cache Update (pk, pv) ---
        if kv_write_indices is None: kv_write_indices = torch.arange(q_seq_len, device=device)
        max_cache_len = self.cache_pk.shape[1] if self.cache_pk is not None else 0
        needed_cache_size = kv_write_indices.max().item() + 1 if kv_write_indices.numel() > 0 else q_seq_len

        if self.cache_pk is None or bsz > self.cache_pk.shape[0] or needed_cache_size > max_cache_len:
            max_seq_len_config = getattr(self.config, 'max_position_embeddings', 2048)
            new_max_len = max(max_seq_len_config, needed_cache_size, max_cache_len)
            self._init_kv_cache(bsz, new_max_len, dtype, device)
            max_cache_len = new_max_len

        # Clamp indices and filter invalid ones (if sequence > max_cache_len)
        current_kv_write_indices = kv_write_indices.clamp(max=max_cache_len - 1)
        valid_idx_mask = kv_write_indices < max_cache_len
        if current_kv_write_indices.numel() > 0:
            pk_to_cache = pk[:, valid_idx_mask, :, :]
            pv_to_cache = pv[:, valid_idx_mask, :, :]
            indices_to_use = current_kv_write_indices[valid_idx_mask]
            if indices_to_use.numel() > 0:
                self.cache_pk[:bsz].index_copy_(1, indices_to_use, pk_to_cache)
                self.cache_pv[:bsz].index_copy_(1, indices_to_use, pv_to_cache)

        # --- 5. Read Projections from Cache ---
        kv_seq_len = needed_cache_size
        kv_seq_len = min(kv_seq_len, max_cache_len)
        pk_cached = self.cache_pk[:bsz, :kv_seq_len] # Shape: [b, kv_s, N_h, r_k]
        pv_cached = self.cache_pv[:bsz, :kv_seq_len] # Shape: [b, kv_s, N_kv, r_v]

        # --- 6. Reconstruct K_hat ---
        # Use V_r_basis.transpose(-1, -2) which has shape [N_h, r_k, Dk]
        # einsum: 'bshr,hrd->bshd'
        k_hat = torch.einsum('bshr,hrd->bshd', pk_cached, self.V_r_basis.transpose(-1, -2))
        # k_hat shape: [b, kv_s, N_h, Dk]

        # --- 7. Reconstruct V_hat ---
        # Use Z_v_basis.transpose(-1, -2) which has shape [N_kv, r_v, Dv]
        # einsum: 'bsgr,grd->bsgd'
        v_hat = torch.einsum('bsgr,grd->bsgd', pv_cached, self.Z_v_basis.transpose(-1, -2))
        # v_hat shape: [b, kv_s, N_kv, Dv]

        # --- 8. Apply RoPE (Corrected) ---
        # Apply RoPE separately to fresh query 'q' and reconstructed key 'k_hat'
        # using their corresponding frequency slices.

        # Select frequency slices based on absolute positions
        # Ensure kv_write_indices is not empty before calling min/max
        if kv_write_indices.numel() > 0:
            q_start_pos = kv_write_indices.min().item() # Get Python int
            q_end_pos = kv_write_indices.max().item() + 1 # Get Python int
            q_pos = torch.arange(q_start_pos, q_end_pos, device=device)
        else:
            # Handle empty case - perhaps create an empty position tensor
            # Or this case might be guaranteed not to happen if q_seq_len > 0
            q_pos = torch.empty(0, dtype=torch.long, device=device)


        k_pos = torch.arange(kv_seq_len, device=device) # kv_seq_len is already an int

        # Clamp positions to max length of precomputed freqs
        max_freq_len = freqs_cis.shape[0]
        q_pos = q_pos.clamp(max=max_freq_len - 1)
        k_pos = k_pos.clamp(max=max_freq_len - 1)

        if q_pos.numel() > 0: # Only select if positions exist
            freqs_cis_q_slice = freqs_cis.index_select(0, q_pos) # Shape [q_s, Dq//2] complex
        else:
            # Need appropriate empty tensor shape if q_pos is empty
            freqs_cis_q_slice = torch.empty((0, freqs_cis.shape[1]), dtype=freqs_cis.dtype, device=device)

        freqs_cis_k_slice = freqs_cis.index_select(0, k_pos) # Shape [kv_s, Dk//2] complex

        # Adapt freqs_cis_k slice if K head dimension differs from Q head dimension
        if self.head_dim != self.k_head_dim:
            # Assuming RoPE was calculated based on self.head_dim (Dq)
            print(f"Warning: Adapting RoPE frequencies for K head_dim ({self.k_head_dim}) from Q head_dim ({self.head_dim})")
            freqs_cis_k_slice = freqs_cis_k_slice[:, :self.k_head_dim//2]

        # Call apply_rotary_emb twice using the original function signature:
        q_rot = apply_rotary_emb(q, freqs_cis=freqs_cis_q_slice) if q_pos.numel() > 0 else q # Handle empty case
        k_hat_rot = apply_rotary_emb(k_hat, freqs_cis=freqs_cis_k_slice)
        # v_hat remains unrotated

        # --- 9. Optional QK Norm ---
        if self.query_norm is not None and self.key_norm is not None:
            q_rot = self.query_norm(q_rot)
            k_hat_rot = self.key_norm(k_hat_rot)

        # --- 10. GQA Grouping for V ---
        # v_hat is [b, kv_s, N_kv, Dv]. Repeat to match query heads.
        v_hat_grouped = v_hat.repeat_interleave(self.heads_per_group, dim=2)
        # v_hat_grouped shape: [b, kv_s, N_h, Dv]

        # --- 11. Standard Attention Computation ---
        # Prepare for matmul: [b, h, seq, dim]
        q_att = q_rot.transpose(1, 2)      # [b, h, q_s, Dq]
        k_att = k_hat_rot.transpose(1, 2)  # [b, h, kv_s, Dk]
        v_att = v_hat_grouped.transpose(1, 2) # [b, h, kv_s, Dv]

        # Calculate scores
        # Matmul: [b, h, q_s, Dk] @ [b, h, Dk, kv_s] -> [b, h, q_s, kv_s]
        attn_scores = torch.matmul(q_att, k_att.transpose(2, 3)) * self.scaling

        # Apply Masking (Causal and Optional Local)
        # Select the correct mask slice based on current q_seq_len and kv_seq_len
        current_effective_mask = None
        if self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING and self.config.sliding_window_size is not None and local_mask is not None:
            current_mask_to_use = local_mask
        else:
            current_mask_to_use = mask

        if current_mask_to_use is not None:
            # Slice the mask: target shape [b, 1, q_s, kv_s] or broadcastable
            mask_slice = current_mask_to_use[:, :, :q_seq_len, :kv_seq_len]
            if mask_slice.dim() == 3: mask_slice = mask_slice.unsqueeze(1) # Add head dim if needed
            # Ensure dimensions match or broadcast
            if attn_scores.shape[-2:] == mask_slice.shape[-2:]:
                attn_scores = attn_scores + mask_slice
            else:
                print(f"Warning: Mask shape mismatch. Scores: {attn_scores.shape}, Mask slice: {mask_slice.shape}. Skipping mask.")


        # Softcapping (Optional)
        if getattr(self.config, 'attn_logit_softcapping', None) is not None:
            softcap = self.config.attn_logit_softcapping
            attn_scores = torch.tanh(attn_scores / softcap) * softcap

        # Softmax
        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(dtype=dtype) # [b, h, q_s, kv_s]

        # Weighted Value Summation
        # Matmul: [b, h, q_s, kv_s] @ [b, h, kv_s, Dv] -> [b, h, q_s, Dv]
        attn_output = torch.matmul(attn_probs, v_att)

        # --- 12. Final Projection ---
        # Reshape back: [b, q_s, h, Dv] -> [b, q_s, h*Dv]
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output_reshaped = attn_output.view(bsz, q_seq_len, self.num_heads * self.head_dim) # Assumes Dv=Dq=head_dim

        final_output = self.o_proj(attn_output_reshaped)

        # NaN check
        if torch.isnan(final_output).any():
            print("WARNING: NaN detected in final ISP_KVAttention output. Replacing with zeros.")
            final_output = torch.nan_to_num(final_output, nan=0.0)

        return final_output


# --- Wrapper Layers and Model (Largely unchanged, just use ISP_KVAttention) ---

class ISP_KVDecoderLayer(nn.Module):
    """Gemma decoder layer using ISP_KVAttention."""
    def __init__(
            self,
            config: gemma_config.GemmaConfig,
            attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.attn_type = attn_type
        self.self_attn = ISP_KVAttention( # Use ISP_KVAttention
            config=config,
            attn_type=self.attn_type,
        )
        # Use standard MLP and norms from gemma_model (or copied definitions)
        self.mlp = GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Optional norms for Gemma 2/3 style
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, 'use_pre_ffw_norm', False) else None)
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, 'use_post_ffw_norm', False) else None)

    def forward(
            self,
            hidden_states: torch.Tensor,
            freqs_cis: torch.Tensor, # Pass the specific freqs for this layer's attn_type
            kv_write_indices: torch.Tensor,
            kv_cache: Optional[Tuple] = None, # Ignored by ISP_KVAttention
            mask: torch.Tensor = None,
            local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # --- Self Attention Block ---
        residual = hidden_states
        attn_input = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=attn_input,
            freqs_cis=freqs_cis, # Pass down the selected frequencies
            kv_write_indices=kv_write_indices,
            kv_cache=None, # Pass None, ISP_KVAttention manages internal cache
            mask=mask,
            local_mask=local_mask,
        )
        normed_attn_output = self.post_attention_layernorm(attn_output)
        hidden_states = residual + normed_attn_output

        # --- MLP Block ---
        residual = hidden_states
        mlp_input = hidden_states
        if self.pre_feedforward_layernorm is not None: mlp_input = self.pre_feedforward_layernorm(mlp_input)
        mlp_output = self.mlp(mlp_input)
        if self.post_feedforward_layernorm is not None: mlp_output = self.post_feedforward_layernorm(mlp_output)
        hidden_states = residual + mlp_output

        return hidden_states

class ISP_KVModel(nn.Module):
    """Gemma model backbone using ISP_KVDecoderLayer."""
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            # Determine attn_type for this layer
            attn_type = (
                config.attn_types[i % len(config.attn_types)]
                if getattr(config, 'attn_types', None) is not None
                else gemma_config.AttentionType.GLOBAL
            )
            self.layers.append(ISP_KVDecoderLayer(config, attn_type)) # Use ISP_KVDecoderLayer

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            # Accept the map of full RoPE buffers
            full_freqs_cis_map: Mapping[gemma_config.AttentionType, torch.Tensor],
            kv_write_indices: torch.Tensor,
            kv_caches: Optional[List[Tuple]] = None, # Ignored
            mask: Optional[torch.Tensor] = None,
            local_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # Select the correct FULL RoPE buffer for this layer's attention type
            layer_freqs_cis = full_freqs_cis_map.get(layer.attn_type)
            if layer_freqs_cis is None:
                print(f"Warning: Could not find freqs_cis for attn_type {layer.attn_type}. Using GLOBAL.")
                layer_freqs_cis = full_freqs_cis_map.get(gemma_config.AttentionType.GLOBAL)

            # Select the relevant slice based on positions AFTER selecting the buffer
            # This needs the actual positions being processed (derived from kv_write_indices?)
            # Let's assume the top-level model passes the SLICED freqs_cis down.
            # **Correction:** The forward signature expects the *sliced* freqs_cis.
            # The top-level model needs to handle slicing based on input_positions.

            hidden_states = layer(
                hidden_states=hidden_states,
                freqs_cis=layer_freqs_cis, # Pass the SLICED freqs for this step
                kv_write_indices=kv_write_indices,
                kv_cache=None, # Pass None
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states


class GemmaForCausalLMwithISP_KV(nn.Module):
    """Top-level Gemma model for causal LM using ISP_KVAttention."""
    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.dtype = config.get_dtype()
        max_seq_len = config.max_position_embeddings
        head_dim = config.head_dim # Used for RoPE calculation
        vocab_size = config.vocab_size

        # --- Tokenizer ---
        tokenizer_path = getattr(config, 'tokenizer', None)
        if tokenizer_path and isinstance(tokenizer_path, str) and os.path.exists(tokenizer_path):
            self.tokenizer = tokenizer.Tokenizer(tokenizer_path)
        else:
            print(f"Warning: Tokenizer not found or path not specified in config ('{tokenizer_path}'). Tokenizer not loaded.")
            self.tokenizer = None

        # --- Embeddings ---
        self.text_token_embedder = Embedding(vocab_size, config.hidden_size, config.quant)

        # --- Backbone Model ---
        self.model = ISP_KVModel(config) # Use the ISP_KV backbone

        # --- LM Head / Sampler ---
        self.sampler = Sampler(vocab_size, config) # Shares weights with embedder

        # --- Precompute RoPE Frequencies ---
        # Register potentially different buffers for Gemma 2/3 local/global types
        # Or a single buffer for Gemma 1 style
        if config.architecture in (gemma_config.Architecture.GEMMA_2, gemma_config.Architecture.GEMMA_3):
            rope_lengths = getattr(config, 'rope_wave_length', {})
            defaults = { gemma_config.AttentionType.LOCAL_SLIDING: 10_000, gemma_config.AttentionType.GLOBAL: 10_000 }
            self._register_freqs_cis('local_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get(gemma_config.AttentionType.LOCAL_SLIDING, defaults[gemma_config.AttentionType.LOCAL_SLIDING]))
            self._register_freqs_cis('global_freqs_cis', head_dim, max_seq_len, theta=rope_lengths.get(gemma_config.AttentionType.GLOBAL, defaults[gemma_config.AttentionType.GLOBAL]), rope_scaling_factor=getattr(config, 'rope_scaling_factor', 1))
        else: # Gemma 1
            self._register_freqs_cis('freqs_cis', head_dim, max_seq_len, theta=getattr(config, 'rope_theta', 10000))


    def _register_freqs_cis(self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1):
        try:
            freqs_cis_data = precompute_freqs_cis(head_dim, max_seq_len * 2, theta=theta, rope_scaling_factor=rope_scaling_factor)
            self.register_buffer(name, freqs_cis_data)
            print(f"Registered RoPE buffer '{name}' with shape {freqs_cis_data.shape}")
        except Exception as e:
            print(f"ERROR registering RoPE buffer '{name}': {e}. Registering empty tensor.")
            self.register_buffer(name, torch.empty(0))


    @torch.no_grad()
    def forward(self,
                input_token_ids: torch.Tensor,
                input_positions: torch.Tensor, # Absolute positions for RoPE slicing
                kv_write_indices: torch.Tensor, # Cache indices for THIS step
                kv_caches: Optional[List[Tuple]] = None, # Ignored
                mask: Optional[torch.Tensor] = None, # Causal mask
                output_positions: Optional[torch.Tensor] = None, # For sampler
                temperatures: Optional[torch.Tensor] = None,
                top_ps: Optional[torch.Tensor] = None,
                top_ks: Optional[torch.Tensor] = None,
                local_mask: Optional[torch.Tensor] = None, # Sliding window mask
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        device = input_token_ids.device

        # --- Prepare SLICED RoPE Frequencies MAP ---
        # Select the correct slice of the precomputed buffer based on input_positions
        freqs_cis_map_sliced = {}
        if hasattr(self, 'freqs_cis'): # Gemma 1
            if self.freqs_cis.device != device: self.freqs_cis = self.freqs_cis.to(device)
            current_freqs_cis = self.freqs_cis.index_select(0, input_positions)
            freqs_cis_map_sliced[gemma_config.AttentionType.GLOBAL] = current_freqs_cis
            freqs_cis_map_sliced[gemma_config.AttentionType.LOCAL_SLIDING] = current_freqs_cis
        else: # Gemma 2/3
            if hasattr(self, 'local_freqs_cis'):
                if self.local_freqs_cis.device != device: self.local_freqs_cis = self.local_freqs_cis.to(device)
                freqs_cis_map_sliced[gemma_config.AttentionType.LOCAL_SLIDING] = self.local_freqs_cis.index_select(0, input_positions)
            if hasattr(self, 'global_freqs_cis'):
                if self.global_freqs_cis.device != device: self.global_freqs_cis = self.global_freqs_cis.to(device)
                freqs_cis_map_sliced[gemma_config.AttentionType.GLOBAL] = self.global_freqs_cis.index_select(0, input_positions)
        # --- RoPE slicing done ---

        # --- Embeddings ---
        if self.text_token_embedder.weight.device != device: self.text_token_embedder = self.text_token_embedder.to(device)
        hidden_states = self.text_token_embedder(input_token_ids)
        normalizer = torch.tensor(self.config.hidden_size ** 0.5, dtype=self.dtype, device=device)
        hidden_states = hidden_states * normalizer

        # --- Adjust Masks ---
        q_len = input_token_ids.shape[1]
        # kv_len depends on the maximum index written to cache so far
        kv_len = kv_write_indices.max().item() + 1 if kv_write_indices.numel() > 0 else q_len
        current_mask = mask[:, :, :q_len, :kv_len] if mask is not None else None
        current_local_mask = local_mask[:, :, :q_len, :kv_len] if local_mask is not None else None

        # --- Pass through ISP_KV Backbone ---
        if next(self.model.parameters()).device != device: self.model = self.model.to(device)

        hidden_states = self.model(
            hidden_states=hidden_states,
            # Pass the map containing SLICED frequency buffers
            full_freqs_cis_map=freqs_cis_map_sliced, # Pass the sliced map
            kv_write_indices=kv_write_indices,
            kv_caches=None, # ISP_KV layers manage internal cache
            mask=current_mask,
            local_mask=current_local_mask,
        )

        # --- Sampling ---
        embedder_weight = self.text_token_embedder.weight
        # Handle potential quantization scaler
        if self.config.quant and hasattr(self.text_token_embedder, 'weight_scaler'):
            embedder_weight = embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1)

        if next(self.sampler.parameters(), torch.tensor(0)).device != device: self.sampler = self.sampler.to(device)
        if output_positions is None: output_positions = torch.tensor([q_len - 1], device=device, dtype=torch.long)

        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    # --- load_weights Method ---
    def load_weights(self, model_path: str):
        """
        Loads weights into the ISP-KV model.

        Expects a state dictionary containing:
        - Standard non-attention weights (embeddings, norms, MLP).
        - Original projection weights (e.g., 'qkv_proj.weight', 'o_proj.weight').
        - ISP-KV basis buffers ('V_r_basis', 'Z_v_basis').
        Keys must match the layer names in GemmaForCausalLMwithISP_KV.
        """
        print(f"Loading weights for ISP-KV model from: {model_path}")
        if os.path.isfile(model_path):
            # Load the entire checkpoint (contains state_dict and possibly config)
            checkpoint = torch.load(model_path, map_location="cpu", weights_only=False) # Use weights_only=False if config is also saved
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict): # Assume it's just the state_dict
                state_dict = checkpoint
            else:
                raise ValueError("Invalid checkpoint format. Expected dict with 'model_state_dict' or just the state_dict.")

            load_result = self.load_state_dict(state_dict, strict=False)
            print(f"Loaded from single file. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

        elif os.path.isdir(model_path):
            # Handle sharded checkpoints
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            if not os.path.exists(index_path):
                # Attempt to load safetensors if index not found
                index_path_st = os.path.join(model_path, 'model.safetensors.index.json')
                if os.path.exists(index_path_st):
                    print("Found safetensors index, attempting to load...")
                    # Add safetensors loading logic here if needed
                    from safetensors.torch import load_file
                    index_path = index_path_st # Use safetensors index
                    # Basic sharded safetensors loading
                    with open(index_path, "r") as f:
                        index = json.load(f)
                    shard_files = sorted(list(set(index["weight_map"].values()))) # Ensure order
                    loaded_state_dict = {}
                    for shard_file in shard_files:
                        shard_path = os.path.join(model_path, shard_file)
                        loaded_state_dict.update(load_file(shard_path, device="cpu"))
                    load_result = self.load_state_dict(loaded_state_dict, strict=False)
                    del loaded_state_dict # Free memory
                    gc.collect()
                    print(f"Loaded from sharded safetensors. Missing: {load_result.missing_keys}, Unexpected: {load_result.unexpected_keys}")

                else:
                    raise FileNotFoundError(f"Cannot find weights index file (pytorch_model.bin.index.json or model.safetensors.index.json) in directory: {model_path}")

            else: # Load sharded .bin files
                print(f"Loading from sharded .bin checkpoint: {model_path}")
                with open(index_path, "r") as f:
                    index = json.load(f)
                shard_files = sorted(list(set(index["weight_map"].values())))
                loaded_keys = set()
                for shard_file in shard_files:
                    shard_path = os.path.join(model_path, shard_file)
                    state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                    current_load_result = self.load_state_dict(state_dict, strict=False)
                    # Manually track missing/unexpected across shards if needed, strict=False handles most cases
                    loaded_keys.update(state_dict.keys())
                    del state_dict
                    gc.collect()

                # Final check (optional, strict=False usually suffices)
                all_model_keys = set(self.state_dict().keys())
                final_missing = list(all_model_keys - loaded_keys)
                print(f"Loaded from sharded .bin. Final missing keys (approx): {final_missing}")
        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")

        print("ISP-KV weights loading attempt finished.")

    # --- generate Method (Largely unchanged from standard model, but ensure cache handling is updated if needed) ---
    @torch.no_grad()
    def generate(
            self,
            prompts: Union[str, Sequence[str]],
            device: Any = None,
            max_tokens: int = 100, # Renamed from output_len
            temperature: Optional[float] = 1.0,
            top_p: float = 0.95,
            top_k: int = 64,
    ) -> Union[str, Sequence[str]]:
        """Generates responses using the ISP-KV model."""
        if self.tokenizer is None: raise RuntimeError("Tokenizer not loaded.")
        if device is None: device = next(self.parameters()).device
        self.to(device) # Ensure model is on target device

        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt: prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens_list = [self.tokenizer.encode(p, bos=True) for p in prompts]
        min_prompt_len = min(len(p) for p in prompt_tokens_list)
        max_prompt_len = max(len(p) for p in prompt_tokens_list)
        max_seq_len = self.config.max_position_embeddings
        total_len = min(max_seq_len, max_prompt_len + max_tokens)

        # --- Input/Output Tensors ---
        all_token_ids = torch.full((batch_size, total_len), self.tokenizer.pad_id, dtype=torch.long, device=device)
        for i, tokens in enumerate(prompt_tokens_list):
            all_token_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)

        # --- KV Cache Init (ISP-KV specific) ---
        # Reset internal caches within each ISP_KVAttention layer
        for layer in self.model.layers:
            if isinstance(layer.self_attn, ISP_KVAttention):
                layer.self_attn.cache_pk = None
                layer.self_attn.cache_pv = None
        print("ISP-KV internal caches reset for generation.")

        # --- Attention Mask ---
        causal_mask = torch.full((1, 1, total_len, total_len), -torch.inf, device=device, dtype=self.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        local_mask = None # Define local mask logic if needed based on config

        # --- Generation Loop ---
        tokens_generated = 0
        current_pos = 0 # Tracks the next position *to be generated*

        # Prefill phase
        if min_prompt_len > 0:
            prefill_ids = all_token_ids[:, :max_prompt_len]
            prefill_pos = torch.arange(0, max_prompt_len, device=device)
            write_indices = prefill_pos # Write indices match absolute positions
            output_pos = torch.tensor([max_prompt_len - 1] * batch_size, device=device) # Sample only the last token for each item

            _ , _ = self( # Ignore output token during prefill
                input_token_ids=prefill_ids,
                input_positions=prefill_pos, # Absolute positions
                kv_write_indices=write_indices,
                mask=causal_mask[:, :, :max_prompt_len, :max_prompt_len],
                output_positions=output_pos,
                temperatures=None, top_ps=None, top_ks=None,
                local_mask=local_mask[:, :, :max_prompt_len, :max_prompt_len] if local_mask else None,
            )
            current_pos = max_prompt_len # Set starting position for decoding

        # Decode phase
        stop_generation = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for _ in range(max_tokens):
            if current_pos >= total_len: break # Stop if we reach max length

            # Prepare inputs for the current step (decode 1 token)
            decode_input_ids = all_token_ids[:, current_pos-1:current_pos] # Previous token is input
            decode_pos = torch.tensor([current_pos - 1] * batch_size, device=device) # Absolute position of the input token
            write_indices = decode_pos # Cache write index is the input token's position
            output_pos = torch.tensor([0] * batch_size, device=device) # Sample from the single output position

            # Select the correct mask slice: query=decode_pos, keys up to current_pos
            # Slice dim 2 (query pos) and dim 3 (key pos)
            mask_slice = causal_mask[:, :, decode_pos : decode_pos+1, :current_pos]
            local_mask_slice = local_mask[:, :, decode_pos:decode_pos+1, :current_pos] if local_mask else None

            temp_tensor = torch.tensor([temperature] * batch_size, device=device) if temperature is not None else None
            topp_tensor = torch.tensor([top_p] * batch_size, device=device)
            topk_tensor = torch.tensor([top_k] * batch_size, device=device, dtype=torch.long)

            next_token, _ = self(
                input_token_ids=decode_input_ids,
                input_positions=decode_pos, # Pass absolute position
                kv_write_indices=write_indices,
                mask=mask_slice,
                output_positions=output_pos,
                temperatures=temp_tensor,
                top_ps=topp_tensor,
                top_ks=topk_tensor,
                local_mask=local_mask_slice,
            ) # Shape [batch_size, 1]

            # Update generated tokens only for sequences that haven't stopped
            # Squeeze next_token if it has an extra dim
            next_token = next_token.squeeze(-1) if next_token.ndim > 1 else next_token
            effective_next_token = torch.where(stop_generation, torch.tensor(self.tokenizer.pad_id, device=device, dtype=torch.long), next_token)
            all_token_ids[:, current_pos] = effective_next_token

            tokens_generated += 1
            current_pos += 1

            # Check for EOS and update stop mask
            stop_generation = stop_generation | (effective_next_token == self.tokenizer.eos_id)
            if stop_generation.all():
                break

        # --- Detokenization ---
        results = []
        for i, tokens in enumerate(all_token_ids.tolist()):
            prompt_len = len(prompt_tokens_list[i])
            start = prompt_len
            # Find actual end based on first pad or EOS after prompt
            try:
                end_pad = tokens.index(self.tokenizer.pad_id, start)
            except ValueError:
                end_pad = total_len
            try:
                end_eos = tokens.index(self.tokenizer.eos_id, start)
            except ValueError:
                end_eos = total_len
            end = min(end_pad, end_eos, start + tokens_generated)

            generated_tokens = tokens[start:end]
            results.append(self.tokenizer.decode(generated_tokens))

        return results[0] if is_str_prompt else results