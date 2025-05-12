#!/usr/bin/env python3
"""
sanity_check.py  – compare logits of the *original* Gemma checkpoint and an
Energy-DK-SVD conversion at **r = head_dim** (no-op rank).

Includes detailed intermediate logit comparisons for a specified attention layer/head.
"""
from __future__ import annotations
from safetensors import safe_open

import argparse
import copy
import json
import os
import sys
import torch
import gc
from typing import Any, Tuple, List, Mapping, Optional

# Assuming gemma codebase is in PYTHONPATH or accessible
# Ensure your PYTHONPATH includes the root of the gemma_pytorch_tpa repository
from gemma import config as gcfg
from gemma import model as gm_ref # Contains RoPE, RMSNorm, etc.
from gemma import model_dksvd as gm_edk
from gemma import tokenizer as gtokenizer

# Helper to print differences
def print_diff(name: str, tensor_ref: torch.Tensor, tensor_edk: torch.Tensor, tol: float = 1e-5, verbose: bool = True):
    if tensor_ref.shape != tensor_edk.shape:
        if verbose: print(f"❌ SHAPE MISMATCH for {name}: REF={tensor_ref.shape}, EDK={tensor_edk.shape}")
        return False
    diff = (tensor_ref.float() - tensor_edk.float()).abs() # Compute diff in float32 for stability
    max_d = diff.max().item()
    rms_d = diff.pow(2).mean().sqrt().item()
    passed = max_d < tol
    status = "✅" if passed else "❌"
    if verbose or not passed:
        print(f"{status} {name+':':<30} Max|Δ| = {max_d:<12.5e} RMS|Δ| = {rms_d:<12.5e}")
    return passed

# --- Weight Loading Functions (adapted from user's snippet & run_edksvd) ---
def _load_weights_from_state_dict(model: torch.nn.Module, state_dict: dict, model_name: str):
    # Filter out keys not expected by the model (e.g. optimizer state, 'freqs_cis' if buffer)
    model_keys = model.state_dict().keys()
    filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_keys}

    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)

    # It's common for freqs_cis to be missing if it's a buffer not saved in older checkpoints
    # or handled differently. We register it as a buffer in model init.
    freqs_cis_keys = [k for k in missing if 'freqs_cis' in k]
    for k_fc in freqs_cis_keys:
        missing.remove(k_fc)

    if missing:
        print(f"⚠️  {model_name} - Missing keys during load: {missing}")
    if unexpected:
        print(f"⚠️  {model_name} - Unexpected keys during load (ignored): {unexpected}")

def _load_checkpoint(model: torch.nn.Module, ckpt_path: str, model_name: str) -> None:
    if not os.path.exists(ckpt_path):
        print(f"❌ ERROR: Checkpoint path not found: {ckpt_path}")
        sys.exit(1)

    if os.path.isfile(ckpt_path):
        print(f"Loading single-file checkpoint for {model_name} from {ckpt_path} …")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if isinstance(state, dict) and "model_state_dict" in state:
            state = state["model_state_dict"]
        _load_weights_from_state_dict(model, state, model_name)
    else: # Sharded directory
        index_file = os.path.join(ckpt_path, "pytorch_model.bin.index.json")
        if not os.path.isfile(index_file):
            # Try model.safetensors.index.json for safetensors format
            index_file_st = os.path.join(ckpt_path, "model.safetensors.index.json")
            if os.path.isfile(index_file_st):
                index_file = index_file_st
                # Basic support for safetensors by listing all .safetensors files
                # Note: This doesn't use the index for selective loading, just loads all shards.
                # A proper safetensors loader would be more robust.
                shard_files_map = {}
                with open(index_file, "r", encoding="utf-8") as f:
                    index_data = json.load(f)
                    shard_files_map = index_data["weight_map"] # This gives param -> shard file

                all_shard_filenames = sorted(list(set(shard_files_map.values())))
                print(f"Loading sharded SafeTensors checkpoint for {model_name} from {ckpt_path} (found {len(all_shard_filenames)} shards)...")

                for i, shard_filename in enumerate(all_shard_filenames):
                    shard_path = os.path.join(ckpt_path, shard_filename)
                    # print(f"Loading shard {i+1}/{len(all_shard_filenames)}: {shard_filename}")
                    shard_state = {}
                    with safe_open(shard_path, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            shard_state[key] = f.get_tensor(key)
                    _load_weights_from_state_dict(model, shard_state, f"{model_name} (shard {shard_filename})")
                    del shard_state
                    gc.collect()
                return # Done with safetensors loading

            else: # No index file found
                print(f"❌ ERROR: Could not find a checkpoint index file (pytorch_model.bin.index.json or model.safetensors.index.json) in '{ckpt_path}'.")
                sys.exit(1)

        # Handling for .bin sharded checkpoints
        print(f"Loading sharded .bin checkpoint for {model_name} from directory {ckpt_path} …")
        with open(index_file, "r", encoding="utf-8") as f:
            index = json.load(f)

        shard_filenames = sorted(list(set(index["weight_map"].values())))
        for i, shard_filename in enumerate(shard_filenames):
            shard_path = os.path.join(ckpt_path, shard_filename)
            # print(f"Loading shard {i+1}/{len(shard_filenames)}: {shard_filename}")
            shard_state = torch.load(shard_path, map_location="cpu", weights_only=True)
            _load_weights_from_state_dict(model, shard_state, f"{model_name} (shard {shard_filename})")
            del shard_state
            gc.collect()

# --- Main Sanity Check Logic ---
def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description="Sanity check E-DK-SVD conversion with intermediate logit comparison.")
    ap.add_argument("--orig_ckpt", required=True, help="Original Gemma checkpoint path (.pt, .bin, or HF dir)")
    ap.add_argument("--edk_ckpt",  required=True, help="Converted E-DK-SVD checkpoint path (.pt or HF dir)")
    ap.add_argument("--prompt",    default="1+1=", help="Prompt to test")
    ap.add_argument("--dtype",     default="float32",
                    choices=["float32","bfloat16","float16"])
    ap.add_argument("--tol", type=float, default=1e-4, help="Failure threshold on max-abs final logit diff")
    ap.add_argument("--intermediate_tol", type=float, default=1e-4, help="Display threshold for intermediate diffs")
    ap.add_argument("--layer_idx", type=int, default=0, help="Layer index to inspect for intermediate values (0-indexed)")
    ap.add_argument("--head_idx_in_group", type=int, default=0, help="Head index within GQA group to inspect (0-indexed)")
    args = ap.parse_args(argv)

    torch_dtype_str = args.dtype
    if args.dtype == "float16": torch_dtype = torch.float16
    elif args.dtype == "bfloat16": torch_dtype = torch.bfloat16
    else: torch_dtype = torch.float32 # default float32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}, computation dtype: {torch_dtype_str}")
    if device.type == "cuda" and torch_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("⚠️ BF16 is not supported on this GPU. Falling back to float32 for computation.")
        torch_dtype = torch.float32
        torch_dtype_str = "float32"


    # ---------------------------------------------------------------- Config
    # Use float32 for config's internal dtype string to avoid issues with get_config_for_1b,
    # then cast model to the desired computation dtype.
    cfg = gcfg.get_config_for_1b(dtype="float32") # Internal config dtype
    cfg.quant = False # Ensure no quantization for this test

    # Attempt to load tokenizer
    try:
        tok = gtokenizer.Tokenizer(cfg.tokenizer)
    except FileNotFoundError:
        print(f"❌ ERROR: Tokenizer file not found at {cfg.tokenizer}.")
        print("Please ensure the tokenizer path in gemma/config.py (get_config_for_1b) is correct,")
        print("or provide it via an environment variable if the config reads from one.")
        sys.exit(1)

    prompt_ids = torch.tensor([tok.encode(args.prompt)], dtype=torch.int64, device=device)
    B, T = prompt_ids.shape
    if T == 0:
        print("❌ ERROR: Prompt tokenized to an empty sequence.")
        sys.exit(1)

    # ---------------------------------------------------------------- Reference Model
    print("\nLoading Reference Model...")
    model_ref = gm_ref.GemmaForCausalLM(copy.deepcopy(cfg)).to(device).eval()
    _load_checkpoint(model_ref, args.orig_ckpt, "Reference Model")
    model_ref.to(torch_dtype)


    # ---------------------------------------------------------------- EDKSVD Model (r=head_dim)
    print("\nLoading E-DK-SVD Model (r=head_dim)...")
    cfg_edk = copy.deepcopy(cfg)
    cfg_edk.qk_rank = cfg.head_dim  # For no-op check, rank is full head dimension
    model_edk = gm_edk.GemmaForCausalLMDKSVD(cfg_edk).to(device).eval()
    _load_checkpoint(model_edk, args.edk_ckpt, "E-DK-SVD Model")
    model_edk.to(torch_dtype)

    # --- Intermediate Checks ---
    print(f"\n--- Intermediate Value Comparison for Layer {args.layer_idx}, Head-in-group {args.head_idx_in_group} ---")

    layer_idx = args.layer_idx
    head_idx_in_group = args.head_idx_in_group
    group_idx = 0 # Gemma 1B has 1 KV group

    if not (0 <= layer_idx < cfg.num_hidden_layers):
        print(f"❌ Invalid layer_idx {layer_idx}. Must be between 0 and {cfg.num_hidden_layers-1}.")
        sys.exit(1)

    num_q_per_kv = cfg.num_attention_heads // cfg.num_key_value_heads
    if not (0 <= head_idx_in_group < num_q_per_kv):
        print(f"❌ Invalid head_idx_in_group {head_idx_in_group}. Must be < {num_q_per_kv}.")
        sys.exit(1)
    overall_head_idx = group_idx * num_q_per_kv + head_idx_in_group

    all_intermediate_passed = True

    with torch.no_grad():
        # 1. Get common input 'x_common_norm' to the self_attn block of the target layer
        # We need to pass the prompt through embeddings and all preceding layers.
        # This ensures the input to the layer being inspected is as realistic as possible.

        # KV caches for pre-computation pass
        temp_kv_caches_ref: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(cfg.num_hidden_layers):
            k_c = torch.zeros((B, cfg.max_position_embeddings, cfg.num_key_value_heads, cfg.head_dim), dtype=torch_dtype, device=device)
            v_c = torch.zeros((B, cfg.max_position_embeddings, cfg.num_key_value_heads, cfg.head_dim), dtype=torch_dtype, device=device)
            temp_kv_caches_ref.append((k_c, v_c))

        temp_kv_caches_edk: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for _ in range(cfg_edk.num_hidden_layers):
            k_c = torch.zeros((B, cfg_edk.max_position_embeddings, cfg_edk.num_key_value_heads, cfg_edk.qk_rank), dtype=torch_dtype, device=device)
            v_c = torch.zeros((B, cfg_edk.max_position_embeddings, cfg_edk.num_key_value_heads, cfg_edk.head_dim), dtype=torch_dtype, device=device)
            temp_kv_caches_edk.append((k_c, v_c))

        current_positions_for_input = torch.arange(T, device=device)

        # Get freqs_cis for the prompt length
        # For Gemma3 architecture (used by 1b config)
        freqs_cis_input_ref = {}
        freqs_cis_input_ref[gcfg.AttentionType.LOCAL_SLIDING] = model_ref.local_freqs_cis.index_select(0, current_positions_for_input)
        freqs_cis_input_ref[gcfg.AttentionType.GLOBAL] = model_ref.global_freqs_cis.index_select(0, current_positions_for_input)

        freqs_cis_input_edk = {}
        freqs_cis_input_edk[gcfg.AttentionType.LOCAL_SLIDING] = model_edk.local_freqs_cis.index_select(0, current_positions_for_input)
        freqs_cis_input_edk[gcfg.AttentionType.GLOBAL] = model_edk.global_freqs_cis.index_select(0, current_positions_for_input)

        # Minimal causal mask for the prompt
        input_mask = torch.full((1, 1, T, T), torch.finfo(torch_dtype).min, device=device)
        input_mask = torch.triu(input_mask, diagonal=1)

        # Get hidden_states input to the target layer
        hs_ref = model_ref.embedder(prompt_ids) * (cfg.hidden_size**0.5)
        hs_edk = model_edk.embedder(prompt_ids) * (cfg_edk.hidden_size**0.5)
        if not print_diff("Initial Embedding Output", hs_ref, hs_edk, tol=args.intermediate_tol): all_intermediate_passed = False

        for i in range(layer_idx):
            hs_ref = model_ref.model.layers[i](
                hidden_states=hs_ref,
                freqs_cis=freqs_cis_input_ref.get(model_ref.model.layers[i].attn_type),
                kv_write_indices=current_positions_for_input,
                kv_cache=temp_kv_caches_ref[i],
                mask=input_mask,
                local_mask=None # Simplified for this check, assuming global or full causal
            )
            hs_edk = model_edk.model.layers[i](
                hidden_states=hs_edk,
                freqs_cis=freqs_cis_input_edk.get(model_edk.model.layers[i].attn_type),
                kv_write_indices=current_positions_for_input,
                kv_cache=temp_kv_caches_edk[i],
                mask=input_mask,
                local_mask=None
            )
            if not print_diff(f"Output of Layer {i}", hs_ref, hs_edk, tol=args.intermediate_tol): all_intermediate_passed = False

        # Input to the target layer's input_layernorm
        x_pre_norm_ref = hs_ref
        x_pre_norm_edk = hs_edk

        # Apply the target layer's input_layernorm
        x_common_norm_ref = model_ref.model.layers[layer_idx].input_layernorm(x_pre_norm_ref)
        x_common_norm_edk = model_edk.model.layers[layer_idx].input_layernorm(x_pre_norm_edk)

        if not print_diff("Input to Self-Attn (x_common_norm)", x_common_norm_ref, x_common_norm_edk, tol=args.intermediate_tol):
            all_intermediate_passed = False
        # Use the reference one as the truly common input if they differ slightly to isolate attention block
        x_common_for_attn = x_common_norm_ref

        # --- Reference Model Path ---
        ref_attn_module = model_ref.model.layers[layer_idx].self_attn

        qkv_ref = ref_attn_module.qkv_proj(x_common_for_attn)
        xq_ref_all, xk_ref_all, _ = qkv_ref.split(
            [ref_attn_module.q_size, ref_attn_module.kv_size, ref_attn_module.kv_size], dim=-1
        )
        xq_ref_all_v = xq_ref_all.view(B, T, ref_attn_module.num_heads, ref_attn_module.head_dim)
        xk_ref_all_v = xk_ref_all.view(B, T, ref_attn_module.num_kv_heads, ref_attn_module.head_dim)

        q_proj_ref = xq_ref_all_v[:, :, overall_head_idx, :]
        k_proj_ref = xk_ref_all_v[:, :, group_idx, :]

        q_norm_ref = ref_attn_module.query_norm(q_proj_ref) if ref_attn_module.query_norm else q_proj_ref
        k_norm_ref = ref_attn_module.key_norm(k_proj_ref) if ref_attn_module.key_norm else k_proj_ref

        ref_layer_attn_type = ref_attn_module.attn_type
        selected_freqs_cis_ref = freqs_cis_input_ref.get(ref_layer_attn_type)

        q_rope_ref = gm_ref.apply_rotary_emb(q_norm_ref.view(B, T, 1, cfg.head_dim), selected_freqs_cis_ref).squeeze(2)
        k_rope_ref = gm_ref.apply_rotary_emb(k_norm_ref.view(B, T, 1, cfg.head_dim), selected_freqs_cis_ref).squeeze(2)

        # --- EDKSVD Model Path ---
        edk_attn_module = model_edk.model.layers[layer_idx].self_attn

        q_proj_edk = edk_attn_module.q_linears[group_idx][head_idx_in_group](x_common_for_attn)
        k_proj_edk = edk_attn_module.k_linears[group_idx](x_common_for_attn)

        q_norm_edk = edk_attn_module.query_norm(q_proj_edk) if edk_attn_module.query_norm else q_proj_edk
        k_norm_edk = edk_attn_module.key_norm(k_proj_edk) if edk_attn_module.key_norm else k_proj_edk

        edk_layer_attn_type = edk_attn_module.attn_type
        selected_freqs_cis_edk = freqs_cis_input_edk.get(edk_layer_attn_type)

        q_rope_edk = gm_edk.apply_rotary_emb(q_norm_edk.view(B, T, 1, cfg_edk.qk_rank), selected_freqs_cis_edk).squeeze(2)
        k_rope_edk = gm_edk.apply_rotary_emb(k_norm_edk.view(B, T, 1, cfg_edk.qk_rank), selected_freqs_cis_edk).squeeze(2)

        print("\n--- Comparing Q path ---")
        if not print_diff("Q_proj (after linear)", q_proj_ref, q_proj_edk, tol=args.intermediate_tol): all_intermediate_passed = False
        if not print_diff("Q_norm (after RMSNorm)", q_norm_ref, q_norm_edk, tol=args.intermediate_tol): all_intermediate_passed = False
        if not print_diff("Q_rope (after RoPE)", q_rope_ref, q_rope_edk, tol=args.intermediate_tol): all_intermediate_passed = False

        print("\n--- Comparing K path ---")
        if not print_diff("K_proj (after linear)", k_proj_ref, k_proj_edk, tol=args.intermediate_tol): all_intermediate_passed = False
        if not print_diff("K_norm (after RMSNorm)", k_norm_ref, k_norm_edk, tol=args.intermediate_tol): all_intermediate_passed = False
        if not print_diff("K_rope (after RoPE)", k_rope_ref, k_rope_edk, tol=args.intermediate_tol): all_intermediate_passed = False

        # Attention Scores (pre-softmax, for the specific head)
        q_s_ref = q_rope_ref.view(B, 1, T, cfg.head_dim) * ref_attn_module.scaling
        k_s_ref = k_rope_ref.view(B, 1, T, cfg.head_dim)
        scores_ref = torch.matmul(q_s_ref, k_s_ref.transpose(-2, -1))

        q_s_edk = q_rope_edk.view(B, 1, T, cfg_edk.qk_rank) * edk_attn_module.scaling
        k_s_edk = k_rope_edk.view(B, 1, T, cfg_edk.qk_rank)
        scores_edk = torch.matmul(q_s_edk, k_s_edk.transpose(-2, -1))

        print("\n--- Comparing Attention Scores (pre-softmax) ---")
        if not print_diff("Scores", scores_ref, scores_edk, tol=args.intermediate_tol): all_intermediate_passed = False

    # --- Final Logit Comparison ---
    print("\n--- Final Logit Comparison ---")
    # Use the KV caches populated during the intermediate checks pass for the target layer
    # For other layers, they are zero, which is fine for a single prompt pass.
    # Reset kv_write_indices for the final full pass.
    kv_write_indices_final = torch.arange(T, device=device)
    output_positions_final = torch.tensor([T - 1], device=device)

    # Corrected mask for final logit pass:
    max_seq_len = cfg.max_position_embeddings # Or cfg_edk, should be same
    # Create a mask of shape (1, 1, T_query, max_seq_len_kv_cache)
    # This mask should be causal for the actual prompt length T_query
    # and allow attention to all T_query key/value entries.
    final_input_mask = torch.full((1, 1, T, max_seq_len), 0.0, dtype=torch_dtype, device=device)
    # Apply causal masking only for the T_query x T_query part
    causal_mask_part = torch.triu(torch.full((T, T), torch.finfo(torch_dtype).min, device=device), diagonal=1)
    final_input_mask[:, :, :T, :T] = causal_mask_part
    # The rest of the mask (:, :, :T, T_query:max_seq_len) remains 0, allowing attention to
    # the K/V cache entries beyond the current query, which are padding for prefill but part of the tensor dimension.

    # Use fresh KV caches for final logit check to ensure clean comparison
    final_kv_caches_ref: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(cfg.num_hidden_layers):
        k_c = torch.zeros((B, cfg.max_position_embeddings, cfg.num_key_value_heads, cfg.head_dim), dtype=torch_dtype, device=device)
        v_c = torch.zeros((B, cfg.max_position_embeddings, cfg.num_key_value_heads, cfg.head_dim), dtype=torch_dtype, device=device)
        final_kv_caches_ref.append((k_c, v_c))

    final_kv_caches_edk: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(cfg_edk.num_hidden_layers):
        k_c = torch.zeros((B, cfg_edk.max_position_embeddings, cfg_edk.num_key_value_heads, cfg_edk.qk_rank), dtype=torch_dtype, device=device)
        v_c = torch.zeros((B, cfg_edk.max_position_embeddings, cfg_edk.num_key_value_heads, cfg_edk.head_dim), dtype=torch_dtype, device=device)
        final_kv_caches_edk.append((k_c, v_c))


    with torch.no_grad():
        _, logits_ref_final = model_ref(
            input_token_ids=prompt_ids,
            input_positions=current_positions_for_input,
            kv_write_indices=kv_write_indices_final,
            kv_caches=final_kv_caches_ref,
            mask=final_input_mask,
            output_positions=output_positions_final,
            temperatures=None,
            top_ps=torch.ones(B, device=device),
            top_ks=torch.ones(B, dtype=torch.long, device=device) * cfg.vocab_size
        )
        logits_ref_final = logits_ref_final.squeeze(0).squeeze(0) # if B=1, T_out=1

        _, logits_edk_final = model_edk(
            input_token_ids=prompt_ids,
            input_positions=current_positions_for_input,
            kv_write_indices=kv_write_indices_final,
            kv_caches=final_kv_caches_edk,
            mask=final_input_mask,
            output_positions=output_positions_final,
            temperatures=None,
            top_ps=torch.ones(B, device=device),
            top_ks=torch.ones(B, dtype=torch.long, device=device) * cfg_edk.vocab_size
        )
        logits_edk_final = logits_edk_final.squeeze(0).squeeze(0)

    final_passed = print_diff("Final Logits", logits_ref_final, logits_edk_final, tol=args.tol, verbose=True)

    if not all_intermediate_passed:
        print("\n⚠️ Some intermediate checks failed or showed differences above threshold.")

    if not final_passed:
        print(f"❌ Sanity check FAILED – final logit max|Δ| >= tolerance {args.tol:.1e}")
        sys.exit(1)

    print(f"\n✅ Sanity check PASSED – final logit max|Δ| < tolerance {args.tol:.1e}")
    sys.exit(0)

if __name__ == "__main__":
    main(sys.argv[1:])