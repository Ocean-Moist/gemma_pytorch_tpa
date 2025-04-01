# File: svdtpa_model.py
"""
Inference-only Gemma model implementation with SVD-based Tensor Product Attention (SVD-TPA / Constant B-Factor TPA).
"""
import gc
import json
import os

import torch
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Tuple, Union, Mapping, Optional, Sequence, Any

# Assuming these imports point to the correct modules relative to this file
from .. import config as gemma_config
from .. import model as gemma_model # For helpers like RMSNorm, Embedding, MLP, RoPE
from .. import tokenizer

# Note: RMSNorm is defined below again for self-containment, but could use gemma_model.RMSNorm
class RMSNorm(nn.Module):
    """RMS Normalization module."""
    def __init__(
            self,
            dim: int,
            eps: float = 1e-6,
            add_unit_offset: bool = True,
    ):
        super().__init__()
        self.eps = eps
        self.add_unit_offset = add_unit_offset
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        # Compute in float32 for stability
        return x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x)
        # Gemma 1 applies weight first, Gemma 2 applies after norm
        if self.add_unit_offset:
            # Gemma 1 style: scale = 1 + weight
            output = output * (1 + self.weight.float())
        else:
            # Gemma 2 style: scale = weight
            output = output * self.weight.float()
        return output.type_as(x)


class SVDTPAAttention(nn.Module):
    """
    SVD-based Tensor Product Attention (Constant B-Factor).
    Caches only A factors, reconstructs Q/K/V using constant B factors, applies RoPE post-reconstruction.
    """

    def __init__(self, config: gemma_config.GemmaConfig, attn_type: gemma_config.AttentionType):
        super().__init__()
        self.config = config
        self.attn_type = attn_type # For potential future use (e.g., sliding window mask)

        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.hidden_size = config.hidden_size

        # --- Retrieve Actual Head Dimensions (Set by conversion) ---
        self.head_dim = getattr(config, 'q_head_dim', config.head_dim)
        self.k_head_dim = getattr(config, 'k_head_dim', self.head_dim)
        self.v_head_dim = getattr(config, 'v_head_dim', self.head_dim)

        # --- Retrieve TPA Ranks and Layout Info (Set by conversion) ---
        self.q_per_head_ranks = getattr(config, 'q_per_head_ranks', [])
        self.q_max_head_rank = getattr(config, 'q_max_head_rank', 0)
        self.q_head_offsets = getattr(config, 'q_head_offsets', [])
        self.total_q_rank = getattr(config, 'total_q_rank', 0) # Sum of actual per-head ranks used

        self.k_rank = getattr(config, 'k_rank', 0) # Max rank used across K groups
        self.v_rank = getattr(config, 'v_rank', 0) # Max rank used across V groups

        if not self.q_per_head_ranks or self.q_max_head_rank == 0 or not self.q_head_offsets or self.total_q_rank == 0 or self.k_rank == 0 or self.v_rank == 0:
            print("WARNING: SVDTPAAttention initialized with missing rank/offset info in config. "
                  "Ensure config is updated after factorization.")
            # Provide minimal defaults to avoid crashing, but model won't work correctly
            self.q_per_head_ranks = [1] * self.num_heads
            self.q_max_head_rank = 1
            self.q_head_offsets = list(range(self.num_heads + 1))
            self.total_q_rank = self.num_heads
            self.k_rank = 1
            self.v_rank = 1


        # --- A-Factor Linear Layers ONLY ---
        # W_A_q projects hidden_state to the concatenated ranks of all Q heads
        self.W_A_q = nn.Linear(self.hidden_size, self.total_q_rank, bias=False)
        # W_A_k projects hidden_state to concatenated ranks of all K groups
        self.W_A_k = nn.Linear(self.hidden_size, self.num_kv_heads * self.k_rank, bias=False)
        # W_A_v projects hidden_state to concatenated ranks of all V groups
        self.W_A_v = nn.Linear(self.hidden_size, self.num_kv_heads * self.v_rank, bias=False)

        # --- Constant B-Factor Buffers ONLY ---
        # These are populated during weight loading from the conversion results
        self.register_buffer('B_const_q', torch.zeros(self.num_heads, self.q_max_head_rank, self.head_dim))
        self.register_buffer('B_const_k', torch.zeros(self.num_kv_heads, self.k_rank, self.k_head_dim))
        self.register_buffer('B_const_v', torch.zeros(self.num_kv_heads, self.v_rank, self.v_head_dim))

        # --- Output Projection ---
        # Input dim must match concatenated head outputs (num_heads * Q head_dim)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        # --- Optional QK Norm ---
        self.query_norm = (
            RMSNorm(self.head_dim, eps=config.rms_norm_eps)
            if getattr(config, 'use_qk_norm', False)
            else None
        )
        self.key_norm = (
            RMSNorm(self.k_head_dim, eps=config.rms_norm_eps)
            if getattr(config, 'use_qk_norm', False)
            else None
        )

        # --- Scaling ---
        if getattr(config, 'query_pre_attn_scalar', None) is not None:
            # Gemma 2/3 uses a pre-attention scalar
            self.scaling = (config.query_pre_attn_scalar) ** -0.5
        else:
            # Gemma 1 style scaling
            self.scaling = self.head_dim ** -0.5

        # --- A-Factor Only KV Cache ---
        self.cache_kA: Optional[torch.Tensor] = None
        self.cache_vA: Optional[torch.Tensor] = None

        # --- GQA Helpers ---
        self.heads_per_kv_group = self.num_heads // self.num_kv_heads
        # Create buffer for group index mapping for efficient repeat_interleave
        self.register_buffer('group_indices', torch.arange(self.num_kv_heads).repeat_interleave(self.heads_per_kv_group), persistent=False)


    def _init_kv_cache(self, batch_size: int, max_seq_len: int, dtype: torch.dtype, device: torch.device):
        """Initialize factorized KV cache storing only A factors."""
        # Cache shapes: [batch_size, max_seq_len, num_kv_heads, rank]
        self.cache_kA = torch.zeros((batch_size, max_seq_len, self.num_kv_heads, self.k_rank),
                                    device=device, dtype=dtype)
        self.cache_vA = torch.zeros((batch_size, max_seq_len, self.num_kv_heads, self.v_rank),
                                    device=device, dtype=dtype)
        # print(f"Initialized SVDTPA KV Cache (A-Factors only): kA={self.cache_kA.shape}, vA={self.cache_vA.shape}")


    def forward(
            self,
            hidden_states: torch.Tensor,
            # Argument rename to reflect it's the FULL buffer
            full_freqs_cis: torch.Tensor,
            kv_write_indices: torch.Tensor,
            kv_cache: Optional[Tuple] = None, # Ignored
            mask: Optional[torch.Tensor] = None,
            local_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, q_seq_len, _ = hidden_states.shape
        device = hidden_states.device
        dtype = hidden_states.dtype

        # --- 1. Project A Factors ---
        # (A factor projection code remains the same)
        A_q_factors = self.W_A_q(hidden_states)
        A_k = self.W_A_k(hidden_states)
        A_v = self.W_A_v(hidden_states)
        A_k = A_k.view(bsz, q_seq_len, self.num_kv_heads, self.k_rank)
        A_v = A_v.view(bsz, q_seq_len, self.num_kv_heads, self.v_rank)

        # --- 2. KV Cache Update ---
        # (Cache update logic remains the same)
        if kv_write_indices is None: kv_write_indices = torch.arange(q_seq_len, device=device)
        max_cache_len = self.cache_kA.shape[1] if self.cache_kA is not None else 0
        needed_cache_size = kv_write_indices.max().item() + 1 if kv_write_indices.numel() > 0 else q_seq_len
        if self.cache_kA is None or bsz > self.cache_kA.shape[0] or needed_cache_size > max_cache_len:
            max_seq_len_config = getattr(self.config, 'max_position_embeddings', 2048)
            new_max_len = max(max_seq_len_config, needed_cache_size, max_cache_len)
            self._init_kv_cache(bsz, new_max_len, dtype, device)
            max_cache_len = new_max_len
        current_kv_write_indices = kv_write_indices.clamp(max=max_cache_len - 1)
        valid_idx_mask = kv_write_indices < max_cache_len
        if current_kv_write_indices.numel() > 0:
            A_k_to_cache = A_k[:, valid_idx_mask, :, :]
            A_v_to_cache = A_v[:, valid_idx_mask, :, :]
            indices_to_use = current_kv_write_indices[valid_idx_mask]
            if indices_to_use.numel() > 0:
                self.cache_kA[:bsz].index_copy_(1, indices_to_use, A_k_to_cache)
                self.cache_vA[:bsz].index_copy_(1, indices_to_use, A_v_to_cache)

        # --- 3. Read K/V A-Factors ---
        # (Read logic remains the same)
        kv_seq_len = needed_cache_size
        kv_seq_len = min(kv_seq_len, max_cache_len)
        A_k_cached = self.cache_kA[:bsz, :kv_seq_len]
        A_v_cached = self.cache_vA[:bsz, :kv_seq_len]

        # --- 4. GQA Repeat A ---
        # (Repeat logic remains the same)
        if self.num_kv_heads < self.num_heads:
            A_k_repeated = A_k_cached[:, :, self.group_indices, :]
            A_v_repeated = A_v_cached[:, :, self.group_indices, :]
        else:
            A_k_repeated = A_k_cached
            A_v_repeated = A_v_cached

        # --- 5. Reconstruct Q, K, V ---
        # (Reconstruction logic remains the same)
        q_unrotated_list = []
        # ... (loop and einsum for q) ...
        for h in range(self.num_heads):
            head_rank = self.q_per_head_ranks[h]
            if head_rank == 0: continue
            start_A_idx = self.q_head_offsets[h]
            end_A_idx = self.q_head_offsets[h+1]
            head_A_q = A_q_factors[:, :, start_A_idx:end_A_idx]
            head_B_const_q = self.B_const_q[h, :head_rank, :]
            q_head = torch.einsum('bqr,rd->bqd', head_A_q, head_B_const_q) / head_rank
            q_unrotated_list.append(q_head)
        if not q_unrotated_list: raise ValueError("No query heads reconstructed.")
        q_unrotated = torch.stack(q_unrotated_list, dim=2)
        k_unrotated = torch.einsum('bkhr,hrd->bkhd', A_k_repeated, self.B_const_k[self.group_indices]) / self.k_rank
        v = torch.einsum('bkhr,hrd->bkhd', A_v_repeated, self.B_const_v[self.group_indices]) / self.v_rank


        # --- 6. Apply RoPE (Post-Reconstruction) ---
        # NOW index the FULL frequency buffer using ABSOLUTE positions

        # Indices for the CURRENT input query tokens (absolute positions)
        current_q_pos_indices = torch.arange(
            kv_write_indices.min(), kv_write_indices.max() + 1, device=device
        )
        # Indices for ALL keys/values in the cache (absolute positions)
        k_pos_indices = torch.arange(kv_seq_len, device=device)

        # Select frequencies from the FULL buffer using these absolute position indices
        # Ensure indices are within the bounds of the full buffer
        max_freq_len = full_freqs_cis.shape[0]
        current_q_pos_indices = current_q_pos_indices.clamp(max=max_freq_len - 1)
        k_pos_indices = k_pos_indices.clamp(max=max_freq_len - 1)

        # Index the full buffer passed as argument
        freqs_cis_q_step = full_freqs_cis.index_select(0, current_q_pos_indices)

        # Reshape/slice full_freqs_cis for K dim if needed
        if self.head_dim == self.k_head_dim:
            full_freqs_cis_k = full_freqs_cis
        else:
            full_freqs_cis_k = full_freqs_cis[:, :self.k_head_dim // 2]
        freqs_cis_k_step = full_freqs_cis_k.index_select(0, k_pos_indices)


        # Reshape freqs for broadcast
        q_rot = q_unrotated # Shape [b, q_s, h, d_q]
        k_rot = k_unrotated # Shape [b, kv_s, h, d_k]
        freqs_cis_q_b = gemma_model.reshape_for_broadcast(freqs_cis_q_step, q_rot)
        freqs_cis_k_b = gemma_model.reshape_for_broadcast(freqs_cis_k_step, k_rot)

        # Apply RoPE separately
        q = gemma_model.apply_rotary_emb(q_rot, freqs_cis_q_b)
        k = gemma_model.apply_rotary_emb(k_rot, freqs_cis_k_b)


        # --- 7. Optional QK Norm ---
        # (QK Norm logic remains the same)
        if self.query_norm is not None and self.key_norm is not None:
            q = self.query_norm(q)
            k = self.key_norm(k)

        # --- 8. Standard Attention Computation ---
        # (Attention computation remains the same)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if q.shape[-1] != k.shape[-1]:
            raise ValueError(f"Q dim {q.shape[-1]} != K dim {k.shape[-1]}")
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling


        # --- Masking, Softcapping ---
        # (Masking/Softcapping logic remains the same, using masks passed down)
        if self.attn_type == gemma_config.AttentionType.LOCAL_SLIDING and self.config.sliding_window_size is not None and local_mask is not None:
            current_effective_mask = local_mask # Already sliced in top-level forward
        else:
            current_effective_mask = mask # Already sliced in top-level forward

        if current_effective_mask is not None:
            # Add head dim for broadcasting if necessary
            if current_effective_mask.dim() == 3: # Shape [b, q_s, kv_s]
                current_effective_mask = current_effective_mask.unsqueeze(1) # -> [b, 1, q_s, kv_s]

            # Check shape before adding
            if attn_scores.shape[-2:] == current_effective_mask.shape[-2:]: # Compare q_s, kv_s dims
                attn_scores = attn_scores + current_effective_mask
            else:
                print(f"Warning: Mask shape mismatch during attention. Scores: {attn_scores.shape}, Mask: {current_effective_mask.shape}. Skipping mask.")


        if getattr(self.config, 'attn_logit_softcapping', None) is not None:
            softcap = self.config.attn_logit_softcapping
            attn_scores = torch.tanh(attn_scores / softcap) * softcap
        attn_probs = F.softmax(attn_scores.float(), dim=-1).to(dtype=dtype)


        # --- Weighted Value Summation ---
        # (Value summation remains the same)
        attn_output = torch.matmul(attn_probs, v)


        # --- 9. Final Projection ---
        # (Final projection logic remains the same, including dim checks/warnings)
        attn_output = attn_output.transpose(1, 2).contiguous()
        output_expected_dim = self.num_heads * self.head_dim
        output_actual_dim = self.num_heads * self.v_head_dim
        if output_actual_dim != output_expected_dim:
            print(f"ERROR/Warning: Final attention output dim {output_actual_dim} != expected o_proj input {output_expected_dim}.")
            # Handle mismatch if necessary (e.g., pad/truncate or project)
            attn_output_reshaped = attn_output.view(bsz, q_seq_len, output_actual_dim) # Might fail o_proj
        else:
            attn_output_reshaped = attn_output.view(bsz, q_seq_len, output_expected_dim)
        final_output = self.o_proj(attn_output_reshaped)


        # (NaN check remains the same)
        if torch.isnan(final_output).any():
            print("WARNING: NaN detected in final SVDTPA output. Replacing with zeros.")
            final_output = torch.nan_to_num(final_output, nan=0.0)

        return final_output

class SVDTPADecoderLayer(nn.Module):
    """Gemma decoder layer using SVDTPA attention."""

    def __init__(
            self,
            config: gemma_config.GemmaConfig,
            attn_type: gemma_config.AttentionType,
    ):
        super().__init__()
        self.attn_type = attn_type
        # Instantiate the corrected SVDTPAAttention
        self.self_attn = SVDTPAAttention(
            config=config,
            attn_type=self.attn_type,
        )
        # Use standard MLP and norms from gemma_model
        self.mlp = gemma_model.GemmaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            quant=config.quant,
        )
        self.input_layernorm = RMSNorm(config.hidden_size,
                                       eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size,
                                                eps=config.rms_norm_eps)
        # Keep optional norms based on config (Gemma 2/3 style)
        self.pre_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, 'use_pre_ffw_norm', False)
            else None
        )
        self.post_feedforward_layernorm = (
            RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            if getattr(config, 'use_post_ffw_norm', False)
            else None
        )

    def forward(
            self,
            hidden_states: torch.Tensor,
            # Accept the full RoPE buffer for this layer
            full_freqs_cis: torch.Tensor,
            kv_write_indices: torch.Tensor,
            kv_cache: Optional[Tuple] = None, # Ignored
            mask: torch.Tensor = None,
            local_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        # --- Self Attention Block ---
        residual = hidden_states
        attn_input = self.input_layernorm(hidden_states)
        attn_output = self.self_attn(
            hidden_states=attn_input,
            # Pass the FULL buffer to the attention module
            full_freqs_cis=full_freqs_cis,
            kv_write_indices=kv_write_indices,
            kv_cache=None, # Pass None
            mask=mask,
            local_mask=local_mask,
        )
        normed_attn_output = self.post_attention_layernorm(attn_output)
        hidden_states = residual + normed_attn_output

        # --- MLP Block ---
        # (MLP logic remains the same)
        residual = hidden_states
        mlp_input = hidden_states
        if self.pre_feedforward_layernorm is not None:
            mlp_input = self.pre_feedforward_layernorm(mlp_input)
        mlp_output = self.mlp(mlp_input)
        if self.post_feedforward_layernorm is not None:
            mlp_output = self.post_feedforward_layernorm(mlp_output)
        hidden_states = residual + mlp_output

        return hidden_states

class SVDTPAModel(nn.Module):
    """Gemma model backbone using SVDTPADecoderLayer."""

    def __init__(self, config: gemma_config.GemmaConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size

        self.layers = nn.ModuleList()
        for i in range(config.num_hidden_layers):
            # Determine attn_type for this layer (Gemma 2/3 logic)
            attn_type = (
                config.attn_types[i % len(config.attn_types)]
                if getattr(config, 'attn_types', None) is not None
                else gemma_config.AttentionType.GLOBAL
            )
            self.layers.append(SVDTPADecoderLayer(config, attn_type))

        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: torch.Tensor,
            # Accept the map of full buffers
            full_freqs_cis_map: Mapping[gemma_config.AttentionType, torch.Tensor],
            kv_write_indices: torch.Tensor,
            kv_caches: Optional[List[Tuple]] = None, # Ignored
            mask: Optional[torch.Tensor] = None,
            local_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for i in range(len(self.layers)):
            layer = self.layers[i]
            # Select the correct FULL buffer for this layer's attention type
            layer_full_freqs_cis = full_freqs_cis_map.get(layer.attn_type)
            if layer_full_freqs_cis is None:
                # Fallback or error if specific type not found (shouldn't happen with current logic)
                print(f"Warning: Could not find freqs_cis for attn_type {layer.attn_type}. Using GLOBAL.")
                layer_full_freqs_cis = full_freqs_cis_map.get(gemma_config.AttentionType.GLOBAL)

            hidden_states = layer(
                hidden_states=hidden_states,
                # Pass the selected FULL buffer down
                full_freqs_cis=layer_full_freqs_cis,
                kv_write_indices=kv_write_indices,
                kv_cache=None, # Pass None
                mask=mask,
                local_mask=local_mask,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states

class GemmaForCausalLMwithSVDTPA(nn.Module):
    """Top-level Gemma model for causal LM using SVDTPA attention."""
    def __init__(
            self,
            config: gemma_config.GemmaConfig,
    ):
        super().__init__()
        self.config = config
        self.dtype = config.get_dtype() # Get dtype from config helper
        max_seq_len = config.max_position_embeddings
        # Use potentially different head dims from config
        q_head_dim = getattr(config, 'q_head_dim', config.head_dim)
        # k_head_dim = getattr(config, 'k_head_dim', config.head_dim) # Not needed directly here
        vocab_size = config.vocab_size

        # --- Tokenizer ---
        # Load tokenizer based on path in config
        tokenizer_path = getattr(config, 'tokenizer', None)
        if tokenizer_path and isinstance(tokenizer_path, str):
            try:
                self.tokenizer = tokenizer.Tokenizer(tokenizer_path)
            except FileNotFoundError:
                print(f"Warning: Tokenizer file not found at {tokenizer_path}. Using placeholder.")
                # Create a dummy tokenizer or raise error if essential
                self.tokenizer = None # Or handle appropriately
        else:
            print("Warning: Tokenizer path not found in config. Tokenizer not loaded.")
            self.tokenizer = None

        # --- Embeddings ---
        self.text_token_embedder = gemma_model.Embedding(vocab_size, config.hidden_size, config.quant)

        # --- Backbone Model ---
        self.model = SVDTPAModel(config) # Use the SVDTPA backbone

        # --- LM Head / Sampler ---
        # Note: Gemma shares weights between embedding and LM head
        self.sampler = gemma_model.Sampler(vocab_size, config) # Uses embedding weights

        # --- Precompute RoPE Frequencies ---
        # Use Q head dim for RoPE computation
        rope_head_dim = q_head_dim
        if config.architecture in (
                gemma_config.Architecture.GEMMA_2, gemma_config.Architecture.GEMMA_3
        ):
            if getattr(config, 'rope_wave_length', None) is None:
                # Try default theta if wavelength missing
                print("Warning: rope_wave_length missing in config for Gemma 2/3. Using theta=10000.")
                config.rope_wave_length = {} # Avoid error below

            rope_lengths = config.rope_wave_length
            defaults = {
                gemma_config.AttentionType.LOCAL_SLIDING: 10_000,
                gemma_config.AttentionType.GLOBAL: 10_000,
            }
            # Register potentially different freqs for local/global attention types
            self._register_freqs_cis('local_freqs_cis', rope_head_dim, max_seq_len,
                                     theta=rope_lengths.get(gemma_config.AttentionType.LOCAL_SLIDING, defaults[gemma_config.AttentionType.LOCAL_SLIDING]))
            self._register_freqs_cis('global_freqs_cis', rope_head_dim, max_seq_len,
                                     theta=rope_lengths.get(gemma_config.AttentionType.GLOBAL, defaults[gemma_config.AttentionType.GLOBAL]),
                                     rope_scaling_factor=getattr(config, 'rope_scaling_factor', 1))
        else: # Gemma 1 style
            self._register_freqs_cis('freqs_cis', rope_head_dim, max_seq_len,
                                     theta=getattr(config, 'rope_theta', 10000)) # Use rope_theta from config if present

    def _register_freqs_cis(
            self, name: str, head_dim: int, max_seq_len: int, theta: int = 10_000, rope_scaling_factor: int = 1
    ):
        """Helper to register RoPE frequencies buffer."""
        try:
            freqs_cis_data = gemma_model.precompute_freqs_cis(
                head_dim,
                max_seq_len * 2, # Precompute longer for potential extrapolation
                theta=theta,
                rope_scaling_factor=rope_scaling_factor
            )
            self.register_buffer(name, freqs_cis_data)
            print(f"Registered RoPE buffer '{name}' with shape {freqs_cis_data.shape}")
        except Exception as e:
            print(f"ERROR registering RoPE buffer '{name}': {e}")
            # Register an empty buffer to avoid crashing downstream if init fails
            self.register_buffer(name, torch.empty(0))


    @torch.no_grad()
    def forward(self,
                input_token_ids: torch.Tensor,
                input_positions: torch.Tensor,
                kv_write_indices: torch.Tensor, # Indices for THIS step
                kv_caches: Optional[List[Tuple]] = None, # Ignored by SVDTPA layers
                mask: Optional[torch.Tensor] = None,
                output_positions: Optional[torch.Tensor] = None,
                temperatures: Optional[torch.Tensor] = None,
                top_ps: Optional[torch.Tensor] = None,
                top_ks: Optional[torch.Tensor] = None,
                local_mask: Optional[torch.Tensor] = None,
                **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:

        device = input_token_ids.device

        # --- Prepare RoPE Frequencies MAP ---
        # Pass the FULL buffers down, indexing will happen inside attention
        freqs_cis_map = {}
        if hasattr(self, 'freqs_cis'): # Gemma 1 style
            if self.freqs_cis.device != device: self.freqs_cis = self.freqs_cis.to(device)
            # Pass the full buffer
            freqs_cis_map[gemma_config.AttentionType.GLOBAL] = self.freqs_cis
            freqs_cis_map[gemma_config.AttentionType.LOCAL_SLIDING] = self.freqs_cis
        else: # Gemma 2/3 style
            if hasattr(self, 'local_freqs_cis'):
                if self.local_freqs_cis.device != device: self.local_freqs_cis = self.local_freqs_cis.to(device)
                freqs_cis_map[gemma_config.AttentionType.LOCAL_SLIDING] = self.local_freqs_cis
            if hasattr(self, 'global_freqs_cis'):
                if self.global_freqs_cis.device != device: self.global_freqs_cis = self.global_freqs_cis.to(device)
                freqs_cis_map[gemma_config.AttentionType.GLOBAL] = self.global_freqs_cis
        # --- RoPE preparation done ---

        # --- Embeddings ---
        # (Embedding code remains the same)
        if self.text_token_embedder.weight.device != device:
            self.text_token_embedder = self.text_token_embedder.to(device)
        hidden_states = self.text_token_embedder(input_token_ids)
        normalizer = torch.tensor(self.config.hidden_size ** 0.5,
                                  dtype=self.dtype, device=device)
        hidden_states = hidden_states * normalizer

        # --- Adjust Masks ---
        # (Mask adjustment code remains the same)
        q_len = input_token_ids.shape[1]
        kv_len = kv_write_indices.max().item() + 1 if kv_write_indices.numel() > 0 else q_len
        current_mask = None
        if mask is not None:
            current_mask = mask[:, :, :q_len, :kv_len]
        current_local_mask = None
        if local_mask is not None:
            current_local_mask = local_mask[:, :, :q_len, :kv_len]

        # --- Pass through SVDTPA Backbone ---
        if next(self.model.parameters()).device != device:
            self.model = self.model.to(device)

        hidden_states = self.model(
            hidden_states=hidden_states,
            # Pass the map containing FULL frequency buffers
            full_freqs_cis_map=freqs_cis_map,
            kv_write_indices=kv_write_indices,
            kv_caches=None, # SVDTPA layers manage internal cache
            mask=current_mask,
            local_mask=current_local_mask,
        )

        # --- Sampling ---
        # (Sampling code remains the same)
        embedder_weight = self.text_token_embedder.weight
        if self.config.quant:
            if hasattr(self.text_token_embedder, 'weight_scaler'):
                embedder_weight = embedder_weight * self.text_token_embedder.weight_scaler.unsqueeze(-1)
            else:
                print("Warning: Quantized model missing weight_scaler for embedder.")
        if next(self.sampler.parameters(), torch.tensor(0)).device != device:
            self.sampler = self.sampler.to(device)
        if output_positions is None:
            output_positions = torch.tensor([q_len - 1], device=device, dtype=torch.long)
        next_tokens, logits = self.sampler(
            embedding=embedder_weight,
            hidden_states=hidden_states,
            output_positions=output_positions,
            temperatures=temperatures,
            top_ps=top_ps,
            top_ks=top_ks,
        )
        return next_tokens, logits

    # --- Keep generate method (needs adaptation if tokenizer is missing) ---
    @torch.no_grad()
    def generate(
            self,
            prompts: Union[str, Sequence[str]],
            device: Any = None,
            max_tokens: int = 100, # Renamed from output_len for clarity
            temperature: Optional[float] = 1.0,
            top_p: float = 0.95,
            top_k: int = 64,
    ) -> Union[str, Sequence[str]]:
        """Generates responses using the SVDTPA model."""

        if self.tokenizer is None:
            raise RuntimeError("Tokenizer not loaded. Cannot use generate method.")

        if device is None:
            device = next(self.parameters()).device
        self.to(device) # Ensure model is on the target device

        is_str_prompt = isinstance(prompts, str)
        if is_str_prompt:
            prompts = [prompts]

        batch_size = len(prompts)
        prompt_tokens_list = [self.tokenizer.encode(p, bos=True) for p in prompts] # Use 'bos' argument
        min_prompt_len = min(len(p) for p in prompt_tokens_list)
        max_prompt_len = max(len(p) for p in prompt_tokens_list)
        max_seq_len = self.config.max_position_embeddings

        # --- Input/Output Tensors ---
        # total_len is the maximum length needed in the cache and output tensor
        total_len = min(max_seq_len, max_prompt_len + max_tokens)

        # Stores all generated token IDs (prompt + generation)
        all_token_ids = torch.full((batch_size, total_len), self.tokenizer.pad_id, dtype=torch.long, device=device)
        for i, tokens in enumerate(prompt_tokens_list):
            all_token_ids[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=device)

        # --- KV Cache Init ---
        # SVDTPA Attention layers manage their own cache internally, we don't need to pass explicit caches.
        # However, we need to clear the internal caches before starting generation.
        for layer in self.model.layers:
            if isinstance(layer.self_attn, SVDTPAAttention):
                layer.self_attn.cache_kA = None
                layer.self_attn.cache_vA = None

        # --- Attention Mask ---
        # Create a full causal mask up to total_len
        causal_mask = torch.full((1, 1, total_len, total_len), -torch.inf, device=device, dtype=self.dtype)
        causal_mask = torch.triu(causal_mask, diagonal=1)
        # local_mask TBD if needed

        # --- Generation Loop ---
        tokens_generated = 0
        input_pos = 0 # Tracks the starting position for the next forward pass

        # Prefill phase
        if min_prompt_len > 0:
            prefill_ids = all_token_ids[:, :max_prompt_len]
            prefill_pos = torch.arange(0, max_prompt_len, device=device)
            write_indices = prefill_pos
            output_pos = torch.tensor([max_prompt_len - 1], device=device) # Sample only the last token

            _, _ = self( # Ignore output token during prefill
                input_token_ids=prefill_ids,
                input_positions=prefill_pos,
                kv_write_indices=write_indices,
                mask=causal_mask[:, :, :max_prompt_len, :max_prompt_len],
                output_positions=output_pos,
                temperatures=None, # No sampling needed during prefill
                top_ps=None,
                top_ks=None,
            )
            input_pos = max_prompt_len # Set starting position for decoding

        # Decode phase
        stop_generation = torch.zeros(batch_size, dtype=torch.bool, device=device)
        for current_pos_offset in range(max_tokens):
            current_pos = input_pos + current_pos_offset
            if current_pos >= total_len: break # Stop if we reach max length

            # Prepare inputs for the current step (decode 1 token)
            decode_input_ids = all_token_ids[:, current_pos-1:current_pos] # Previous token is input
            decode_pos = torch.tensor([current_pos - 1], device=device)
            write_indices = decode_pos
            output_pos = torch.tensor([0], device=device) # We sample from the output of this single step

            # Select the correct mask slice for this step: query=current_pos-1, keys up to current_pos
            mask_slice = causal_mask[:, :, current_pos-1:current_pos, :current_pos]

            # Prepare sampling parameters
            temp_tensor = torch.tensor([temperature] * batch_size, device=device) if temperature is not None else None
            topp_tensor = torch.tensor([top_p] * batch_size, device=device)
            topk_tensor = torch.tensor([top_k] * batch_size, device=device, dtype=torch.long)

            next_token, _ = self(
                input_token_ids=decode_input_ids,
                input_positions=decode_pos,
                kv_write_indices=write_indices,
                mask=mask_slice,
                output_positions=output_pos,
                temperatures=temp_tensor,
                top_ps=topp_tensor,
                top_ks=topk_tensor,
            ) # Shape [batch_size, 1]

            next_token = next_token.squeeze(1) # Shape [batch_size]

            # Update generated tokens, only for sequences that haven't stopped
            next_token = torch.where(stop_generation, torch.tensor(self.tokenizer.pad_id, device=device), next_token)
            all_token_ids[:, current_pos] = next_token
            tokens_generated += 1

            # Check for EOS and update stop mask
            stop_generation = stop_generation | (next_token == self.tokenizer.eos_id)
            if stop_generation.all():
                break

        # --- Detokenization ---
        results = []
        for i, tokens in enumerate(all_token_ids.tolist()):
            prompt_len = len(prompt_tokens_list[i])
            # Extract generated part (after prompt, up to where generation stopped or max_len)
            start = prompt_len
            end = min(prompt_len + tokens_generated, total_len)
            generated_tokens = tokens[start:end]

            # Trim at EOS if found
            if self.tokenizer.eos_id in generated_tokens:
                eos_index = generated_tokens.index(self.tokenizer.eos_id)
                generated_tokens = generated_tokens[:eos_index]

            results.append(self.tokenizer.decode(generated_tokens))

        return results[0] if is_str_prompt else results

    def load_weights(self, model_path: str):
        # Standard weight loading logic (copied from gemma_model)
        # Assumes SVDTPA model state_dict matches saved factorized weights
        if os.path.isfile(model_path):
            print(f"Loading SVDTPA weights from single file: {model_path}")
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            if 'model_state_dict' in state_dict: state_dict = state_dict['model_state_dict']
            load_result = self.load_state_dict(state_dict, strict=False)
        elif os.path.isdir(model_path):
            index_path = os.path.join(model_path, 'pytorch_model.bin.index.json')
            if not os.path.exists(index_path):
                raise FileNotFoundError(f"Cannot find weights index file in directory: {index_path}")
            print(f"Loading SVDTPA weights from sharded checkpoint: {model_path}")
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            shard_files = list(set(index["weight_map"].values()))
            loaded_keys = set()
            for shard_file in shard_files:
                shard_path = os.path.join(model_path, shard_file)
                state_dict = torch.load(shard_path, map_location="cpu", weights_only=True)
                load_result = self.load_state_dict(state_dict, strict=False)
                loaded_keys.update(state_dict.keys())
                del state_dict
                gc.collect()
            # Final check for missing keys after loading all shards
            all_model_keys = set(self.state_dict().keys())
            missing_keys = list(all_model_keys - loaded_keys)
            if missing_keys: load_result.missing_keys.extend(missing_keys) # Add any truly missing keys

        else:
            raise FileNotFoundError(f"Model path not found: {model_path}")

        # Report loading status
        if load_result.missing_keys:
            print(f"Warning: Missing keys during SVDTPA weight loading: {load_result.missing_keys[:10]}...")
        if load_result.unexpected_keys:
            print(f"Warning: Unexpected keys during SVDTPA weight loading: {load_result.unexpected_keys[:10]}...")
        print("SVDTPA weights loaded.")


# Expose helpers needed by external modules like generate
precompute_freqs_cis = gemma_model.precompute_freqs_cis
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """Applies the rotary embedding to tensors with shape [bsz, seq_len, num_heads, head_dim]."""
    # x shape: [bsz, seq_len, num_heads, head_dim]
    # freqs_cis shape: broadcastable, e.g., [1, seq_len, 1, head_dim//2] complex

    # Reshape x to [bsz, seq_len, num_heads, head_dim//2, 2] and convert to complex
    x_reshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    x_complex = torch.view_as_complex(x_reshaped)
    # x_complex shape: [bsz, seq_len, num_heads, head_dim//2]

    # Ensure freqs_cis is broadcastable. Expected shape e.g., [1, seq_len, 1, head_dim//2]
    # The reshape_for_broadcast helper should handle this.
    if freqs_cis.shape != (1, x.shape[1], 1, x.shape[-1] // 2) and freqs_cis.shape != (x.shape[1], x.shape[-1] // 2) :
        # Add specific check for expected broadcast shape or original shape
        # If it's just [seq_len, dim//2], reshape it here
        if freqs_cis.shape == (x.shape[1], x.shape[-1] // 2):
            print("Reshaping freqs_cis inside apply_rotary_emb")
            freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(2) # Add batch and head dims
        # else: # Add more robust shape checking if needed
        #    print(f"Warning: Unexpected freqs_cis shape {freqs_cis.shape} in apply_rotary_emb")


    # Apply rotary embedding through complex multiplication
    x_rotated_complex = x_complex * freqs_cis
    # x_rotated_complex shape: [bsz, seq_len, num_heads, head_dim//2]

    # Convert back to real representation
    x_rotated_real = torch.view_as_real(x_rotated_complex)
    # x_rotated_real shape: [bsz, seq_len, num_heads, head_dim//2, 2]

    # Reshape back to original head_dim
    x_out = x_rotated_real.reshape(*x.shape) # Reshape back to [bsz, seq_len, num_heads, head_dim]

    return x_out.type_as(x) # Cast back to original dtype
reshape_for_broadcast = gemma_model.reshape_for_broadcast