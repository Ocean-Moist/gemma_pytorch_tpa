# -*- coding: utf-8 -*-
"""Full Gemma-3 1 B wrapper that swaps vanilla heads for GCB heads."""
import math, torch
from typing import List, Tuple

from gemma import config as gcfg, tokenizer
from gemma.model import GemmaForCausalLM, RMSNorm, precompute_freqs_cis

from .gcb_meta import GCBMeta
from .layers    import GCBHead, GCCache
from .debug_utils import dbg, check

# --------------------------------------------------------------------
class GemmaForCausalLM_GCB(torch.nn.Module):
    def __init__(
        self,
        cfg: gcfg.GemmaConfig,
        meta_path: str,
        ckpt_path: str,
        r_k=8, r_a=4, r_b=4, r_v=8,
    ):
        super().__init__()
        self.cfg = cfg
        self.tok = tokenizer.Tokenizer(cfg.tokenizer)

        # ----- Load vanilla backbone weights ----------------------
        cfg.quant = False                                 # Turn off quantization to match FP16 checkpoint
        self.base = GemmaForCausalLM(cfg)                 # full model (has embedder)
        # Unwrap the model_state_dict from the checkpoint dictionary
        sd = torch.load(ckpt_path, mmap=True, weights_only=True)["model_state_dict"]
        self.base.load_state_dict(sd, strict=False)
        self.embedder = self.base.embedder                # keep a handle
        self.backbone = self.base.model                   # decoder stack only

        # ---------- Insert GCB heads + caches ---------------------
        meta = GCBMeta.load(meta_path)
        r_k = next(iter(meta.heads.values())).P_r.shape[1]
        r_v = meta.layers[0].Z_r.shape[1]
        d_k, d_v = cfg.head_dim, cfg.head_dim      # Gemma uses same dim

        n_heads, max_seq = cfg.num_attention_heads, cfg.max_position_embeddings
        self.layer_caches: List[GCCache] = torch.nn.ModuleList()

        for l_idx, layer in enumerate(self.backbone.layers):
            cache = GCCache(max_seq, n_heads, r_k, r_v,
                            d_k, device='cpu')   # will move with .to()
            self.layer_caches.append(cache)

            # keep the original GemmaAttention
            layer.attn_vanilla = layer.self_attn

            # build per-head GCB replacements
            gcb_heads = torch.nn.ModuleList()
            for h_idx in range(n_heads):
                aux = meta.heads[(l_idx, h_idx)]
                head = GCBHead(aux, d_k, d_v, r_k, r_v)
                head.Z_r = meta.layers[l_idx].Z_r.float()
                gcb_heads.append(head)
            layer.self_attn = gcb_heads          # overwrite

        self.final_norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.register_buffer(
            'freqs_cis',
            precompute_freqs_cis(d_k, max_seq * 2, theta=10_000)
        )

    # ----------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        device: str = "cuda",
        out_len: int = 128,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: int = 64,
    ) -> str:
        self.to(device).eval()

        ids = torch.tensor(self.tok.encode(prompt), device=device).unsqueeze(0)
        B, init_len = ids.shape
        max_len = init_len + out_len
        assert max_len <= self.cfg.max_position_embeddings

        hidden = self.embedder(ids) * math.sqrt(self.cfg.hidden_size)

        # ---------- autoregressive loop ---------------------------
        for step in range(max_len):
            if step >= hidden.size(1):           # extend hidden with last token
                token_embed = self.embedder(ids[:, -1:])
                hidden = torch.cat(
                    [hidden, token_embed * math.sqrt(self.cfg.hidden_size)], 1
                )

            self._transform_layerwise(hidden, step)

            if step < init_len - 1:
                continue    # still inside prompt

            logits_raw = hidden[:, step] @ self.embedder.weight.T
            
            # Add a sanity check for embedder weight
            with torch.no_grad():
                print(f"Embedder weight - dtype: {self.embedder.weight.dtype}, mean abs: {self.embedder.weight.abs().mean().item():.4f}")
                
            dbg("Final Logits Raw", logits_raw, step=step)
            
            logits = logits_raw.clone()
            if temperature:
                logits /= temperature
            dbg("Final Logits Scaled", logits, step=step)
            
            probs = torch.softmax(logits, -1)
            dbg("Final Probs", probs, step=step)
            
            # Check if probabilities are reasonable
            max_prob = probs.max().item()
            top5_probs, top5_indices = torch.topk(probs, 5, dim=-1)
            dbg("Top-5 Probs", top5_probs, step=step)
            dbg("Top-5 Indices", top5_indices, step=step)
            
            # Log token IDs for top probabilities
            token_info = []
            for idx in top5_indices[0]:
                token_text = self.tok.decode([idx.item()])
                token_info.append(f"{idx.item()}:{token_text}")
            print(f"Step {step} - Top tokens: {', '.join(token_info)}")

            # nucleus + top-k
            sv, si = torch.sort(probs, -1, True)
            keep = torch.cumsum(sv, -1) <= top_p
            sv = sv * keep
            if top_k:
                sv[..., top_k:] = 0
            sv = sv / sv.sum(-1, keepdim=True)
            dbg("Nucleus+TopK Probs", sv, step=step)
            
            next_id = torch.multinomial(sv, 1)   # (B , 1)
            dbg("Next Token ID", next_id, step=step)
            next_token_text = self.tok.decode([next_id.item()])
            print(f"Step {step} - Selected token: {next_id.item()}:{next_token_text}")

            ids = torch.cat([ids, next_id], 1)
            if next_id.item() == self.tok.eos_id:
                break

        return self.tok.decode(ids[0].tolist())[len(prompt):]

    # ----------------------------------------------------------------
    def _transform_layerwise(self, hidden: torch.Tensor, step: int):
        """Run one transformer layer stack for a single time-step."""
        B, _ = hidden.shape[:2]
        pos = torch.tensor([step], device=hidden.device)
        freqs_row = self.freqs_cis.index_select(0, pos)   # (1 , d/2 , 2)

        h_step = hidden[:, step:step + 1]                 # (B , 1 , d_model)
        dbg("h_step IN", h_step, step=step)

        for l_idx, layer in enumerate(self.backbone.layers):
            d_k = self.cfg.head_dim
            attn = layer.attn_vanilla
            
            dbg("layer_in", h_step, l_idx=l_idx, step=step)

            # ---- *Pre-LN* then QKV projection (restores scale discipline) ----
            h_norm = layer.input_layernorm(h_step) # <<< FIX: NORMALIZE h_step FIRST
            dbg("h_norm", h_norm, l_idx=l_idx, step=step)
            
            qkv = attn.qkv_proj(h_norm)            # <<< Use normalized state for QKV
            qkv = qkv.squeeze(1)                   # (B , qkv_dim)
            dbg("qkv", qkv, l_idx=l_idx, step=step)

            nh, nkv = attn.num_heads, attn.num_kv_heads
            d_q = nh * d_k
            d_kv = nkv * d_k
            q_flat, k_flat, v_flat = torch.split(qkv, [d_q, d_kv, d_kv], dim=-1)

            q_heads = q_flat.view(B, nh, d_k)             # (B , H , d_k)
            k_kv    = k_flat.view(B, nkv, d_k)
            v_kv    = v_flat.view(B, nkv, d_k)
            
            dbg("q_heads", q_heads, l_idx=l_idx, step=step)
            dbg("k_kv", k_kv, l_idx=l_idx, step=step)
            dbg("v_kv", v_kv, l_idx=l_idx, step=step)

            n_q_per_kv = nh // nkv

            logits_all: List[torch.Tensor] = []
            values_all: List[torch.Tensor] = []

            for h_idx, gcb_head in enumerate(layer.self_attn):
                q_h = q_heads[:, h_idx]                   # (B , d_k)
                kv_idx = h_idx // n_q_per_kv
                k_h = k_kv[:, kv_idx]
                v_h = v_kv[:, kv_idx]
                
                # Apply the missing RMS normalization from the vanilla attention
                if attn.query_norm is not None and attn.key_norm is not None:
                    q_h = attn.query_norm(q_h)
                    k_h = attn.key_norm(k_h)

                # --- NEW: vanilla Gemma scale (query *and* key) ---------------
                scale = 1.0 / math.sqrt(self.cfg.head_dim)      # 1 / √d_k
                q_h *= scale
                k_h *= scale
                
                dbg("q_h_norm", q_h, l_idx, h_idx, step)
                dbg("k_h_norm", k_h, l_idx, h_idx, step)

                lg, vh = gcb_head(
                    q_h, k_h, v_h, freqs_row,
                    self.layer_caches[l_idx],
                    step, h_idx
                )
                if lg is not None:                # skip during pure pre-fill
                    logits_all.append(lg)         # list of (B , step)
                    values_all.append(vh)         # list of (step , d_v)
                    dbg(f"logits_L{l_idx}H{h_idx}", lg, l_idx, h_idx, step)
                    dbg(f"values_L{l_idx}H{h_idx}", vh, l_idx, h_idx, step)

            if logits_all:     # not empty once we pass the first token
                logits = torch.stack(logits_all)          # (H , B , step)
                values = torch.stack(values_all)          # (H , step , d_v)
                dbg("stacked_logits", logits, l_idx=l_idx, step=step)
                
                attn_weights = torch.softmax(logits, -1)  # (H , B , step)
                dbg("attn_weights", attn_weights, l_idx=l_idx, step=step)
                
                # Check attention weight distribution
                if step > 0:
                    uniform_weights = 1.0 / step
                    max_weights = attn_weights.max(dim=-1)[0]
                    min_weights = attn_weights.min(dim=-1)[0]
                    entropy = -(attn_weights * torch.log(attn_weights + 1e-8)).sum(dim=-1)
                    max_entropy = -uniform_weights * step * torch.log(torch.tensor([uniform_weights], device=attn_weights.device))
                    
                    # Calculate entropy ratio (1.0 = perfectly uniform, close to 0 = peaky distribution)
                    entropy_ratio = entropy / max_entropy
                    dbg("attn_entropy_ratio", entropy_ratio, l_idx=l_idx, step=step)
                    
                    # Warn if attention weights are too concentrated
                    check(entropy_ratio.min() > 0.3,
                          f"Attention weights too concentrated, min entropy ratio: {entropy_ratio.min().item():.3f}",
                          l_idx, -1, step, level="WARN")
                
                # --- build per-head context ------------------------------------------------
                ctx = (attn_weights.unsqueeze(-1) * values).sum(-2)   # (H , B , d_v)
                dbg("context_vectors", ctx, l_idx=l_idx, step=step)

                # Ensure ctx shape is correct before o_proj
                ctx = ctx.permute(1, 0, 2).reshape(B, 1, -1)
                in_dim = attn.o_proj.weight.shape[1]
                ctx = ctx[..., :in_dim]
                ctx = layer.attn_vanilla.o_proj(ctx)
                dbg("ctx_after_proj", ctx, l_idx=l_idx, step=step)
                
                # • Vanilla order = LN(ctx)  →  add residual
                ctx_norm = layer.post_attention_layernorm(ctx)
                dbg("ctx_norm", ctx_norm, l_idx=l_idx, step=step)
                h_step   = h_step + ctx_norm
                dbg("h_step_post_attn", h_step, l_idx=l_idx, step=step)

            # ------------- Feed-forward + norms (Revised for Gemma 2/3 compatibility) ---
            res = h_step # Residual connection starts from state *after* attention output is added

            # Check and apply pre-FFW norm if it exists
            if hasattr(layer, 'pre_feedforward_layernorm') and layer.pre_feedforward_layernorm is not None:
                 h_ff_in = layer.pre_feedforward_layernorm(res)
                 dbg("h_ff_in_pre", h_ff_in, l_idx=l_idx, step=step)
            # Fallback: Apply post-attention norm if pre-FFW norm doesn't exist (older style or specific configs)
            elif hasattr(layer, 'post_attention_layernorm'):
                 h_ff_in = layer.post_attention_layernorm(res)
                 dbg("h_ff_in_post", h_ff_in, l_idx=l_idx, step=step)
            else:
                 # Should not happen in standard Gemma models, but handle defensively
                 h_ff_in = res # Pass residual directly if no relevant norm found
                 dbg("h_ff_in_direct", h_ff_in, l_idx=l_idx, step=step)

            h_ff_out = layer.mlp(h_ff_in)
            dbg("h_ff_out", h_ff_out, l_idx=l_idx, step=step)

            # Apply post-FFW LN if present, *then* residual add
            if hasattr(layer, 'post_feedforward_layernorm') and layer.post_feedforward_layernorm is not None:
                 h_ff_out = layer.post_feedforward_layernorm(h_ff_out)
                 dbg("h_ff_out_norm", h_ff_out, l_idx=l_idx, step=step)

            h_step = res + h_ff_out
            dbg("layer_out", h_step, l_idx=l_idx, step=step)

        final_h_step = self.final_norm(h_step)
        dbg("h_step OUT", final_h_step, step=step)
        hidden[:, step:step + 1] = final_h_step