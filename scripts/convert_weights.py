import torch
import gc
from absl import app, flags
import os
import sys

# Ensure the gemma module can be found if script is run from a different directory
# Assuming standard project structure where 'scripts' is a sibling of 'gemma'
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from gemma import config as gemma_config
# We don't need to import gemma_model here as we are directly manipulating state_dicts

FLAGS = flags.FLAGS

flags.DEFINE_string('ckpt_path', None, 'Path to the original Gemma checkpoint file.', required=True)
flags.DEFINE_string('variant', '1b', 'Model variant (e.g., 1b for Gemma 1b models like gemma-1.1-1b-it).')
flags.DEFINE_string('output_path', None, 'Path to save the DK-SVD compressed checkpoint.', required=True)
flags.DEFINE_integer('rank_r', 64, 'Target rank r_g for QK compression.')
flags.DEFINE_string('dtype', 'bfloat16', 'Target dtype for the new weights (e.g., float32, bfloat16, float16).')
flags.DEFINE_string('device', 'cuda', 'Device to run the conversion on (cuda or cpu).')


def get_torch_dtype(dtype_str: str):
    if dtype_str == "float32":
        return torch.float32
    elif dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")

def dequantize_weight(q_weight_int8: torch.Tensor, q_scaler: torch.Tensor):
    # q_weight_int8: (out_features, in_features), dtype=torch.int8
    # q_scaler: (out_features), dtype can vary (e.g. float32, bfloat16)
    # Ensure scaler is on the same device and float for multiplication
    return q_weight_int8.float() * q_scaler.to(device=q_weight_int8.device, dtype=torch.float32).unsqueeze(-1)


def main(_):
    if not torch.cuda.is_available() and FLAGS.device == 'cuda':
        print("CUDA not available, falling back to CPU.")
        device = torch.device('cpu')
    elif FLAGS.device == 'cuda':
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    target_dtype = get_torch_dtype(FLAGS.dtype)
    svd_dtype = torch.float32 # SVD computation should be in high precision

    print(f"Loading original model config for variant: {FLAGS.variant}")
    model_config = gemma_config.get_model_config(FLAGS.variant)

    print(f"Loading original checkpoint from: {FLAGS.ckpt_path}")
    original_checkpoint = torch.load(FLAGS.ckpt_path, map_location='cpu')
    original_state_dict = original_checkpoint['model_state_dict']

    first_qkv_scaler_key = 'model.layers.0.self_attn.qkv_proj.weight_scaler'
    input_is_quantized = first_qkv_scaler_key in original_state_dict
    print(f"Input model appears to be {'quantized' if input_is_quantized else 'not quantized'}.")

    new_dk_svd_state_dict_parts = {} # Holds only the new DK-SVD computed weights

    hidden_size = model_config.hidden_size
    num_attention_heads = model_config.num_attention_heads
    num_key_value_heads = model_config.num_key_value_heads
    head_dim = model_config.head_dim

    W_Q_total_dim = num_attention_heads * head_dim
    W_KV_total_dim = num_key_value_heads * head_dim

    rank_r = FLAGS.rank_r
    if rank_r <= 0:
        raise ValueError("rank_r must be positive.")
    if rank_r >= head_dim: # For GQA, W_K is d x d_k, so A_g is d x d. Max rank of A_g related to d_k.
        # Max rank of W_Q W_K^T is min(rank(W_Q), rank(W_K)).
        # Rank(W_K) is at most min(d, d_k). For skinny W_K, it's d_k.
        # So rank_r should be <= d_k. The doc says r_g < d_k.
        print(f"Warning: rank_r ({rank_r}) is not less than head_dim ({head_dim}). Check if this is intended.")

    # Sanity check for Gemma 1b (single GQA group where N_kv=1)
    if num_key_value_heads != 1:
        print(f"Warning: This script is primarily designed for GQA models with num_key_value_heads=1 (like Gemma 1b). "
              f"Current model has num_key_value_heads={num_key_value_heads}. Ensure logic adapts if needed.")

    for i in range(model_config.num_hidden_layers):
        print(f"Processing layer {i} on device {device}...")
        layer_prefix = f"model.layers.{i}.self_attn."

        # --- QKV Projection ---
        qkv_weight_key = layer_prefix + "qkv_proj.weight"
        W_qkv_ckpt_raw = original_state_dict[qkv_weight_key].to(device) # Move to device

        if input_is_quantized:
            qkv_scaler_key = layer_prefix + "qkv_proj.weight_scaler"
            qkv_scaler = original_state_dict[qkv_scaler_key].to(device)
            W_qkv_dequant = dequantize_weight(W_qkv_ckpt_raw, qkv_scaler).to(svd_dtype)
        else:
            W_qkv_dequant = W_qkv_ckpt_raw.to(svd_dtype)

        # W_qkv_dequant shape: ((num_heads + 2 * num_kv_heads) * head_dim, hidden_size)

        W_Q_ckpt_part = W_qkv_dequant[:W_Q_total_dim, :]
        W_K_ckpt_part = W_qkv_dequant[W_Q_total_dim : W_Q_total_dim + W_KV_total_dim, :]
        W_V_ckpt_part = W_qkv_dequant[W_Q_total_dim + W_KV_total_dim :, :]

        W_Q_math = W_Q_ckpt_part.T # (hidden_size, W_Q_total_dim)
        W_K_math = W_K_ckpt_part.T # (hidden_size, W_KV_total_dim) -> for GQA N_kv=1, this is (d, d_k)
        W_V_math = W_V_ckpt_part.T # (hidden_size, W_KV_total_dim)

        print(f"  Shapes: W_Q_math: {W_Q_math.shape}, W_K_math: {W_K_math.shape}, W_V_math: {W_V_math.shape}")
        print(f"  Forming A_g = W_Q W_K^T ({W_Q_math.shape} @ {W_K_math.T.shape})...")
        A_g = W_Q_math @ W_K_math.T # (hidden_size, hidden_size)

        print(f"  Performing SVD on A_g (shape {A_g.shape})...")
        U_g, s_g_vec, Vh_g = torch.linalg.svd(A_g)

        U_grg = U_g[:, :rank_r]
        s_grg_select_vec = s_g_vec[:rank_r]
        s_grg_vec_sqrt = torch.sqrt(torch.clamp(s_grg_select_vec, min=1e-12)) # Clamp for stability
        Sigma_grg_sqrt_diag = torch.diag(s_grg_vec_sqrt)

        V_grg = Vh_g[:rank_r, :].T # Vh_g is V.T; V_grg is (hidden_size, rank_r)

        WQ_new_g_math = U_grg @ Sigma_grg_sqrt_diag # (hidden_size, rank_r)
        WK_new_g_math = V_grg @ Sigma_grg_sqrt_diag # (hidden_size, rank_r)

        new_dk_svd_state_dict_parts[layer_prefix + "q_proj_dksvd.weight"] = WQ_new_g_math.T.to(target_dtype)
        new_dk_svd_state_dict_parts[layer_prefix + "k_proj_dksvd.weight"] = WK_new_g_math.T.to(target_dtype)
        new_dk_svd_state_dict_parts[layer_prefix + "v_proj.weight"] = W_V_math.T.to(target_dtype)

        # --- O Projection ---
        o_proj_weight_key = layer_prefix + "o_proj.weight"
        W_o_ckpt_raw = original_state_dict[o_proj_weight_key].to(device)

        if input_is_quantized:
            o_proj_scaler_key = layer_prefix + "o_proj.weight_scaler"
            o_proj_scaler = original_state_dict[o_proj_scaler_key].to(device)
            W_o_dequant = dequantize_weight(W_o_ckpt_raw, o_proj_scaler).to(svd_dtype)
        else:
            W_o_dequant = W_o_ckpt_raw.to(svd_dtype)

        # W_o_dequant shape: (hidden_size, num_attention_heads * head_dim)
        # New o_proj input dim for single GQA group is head_dim (or d_v)
        W_o_new_ckpt = W_o_dequant[:, :head_dim]
        new_dk_svd_state_dict_parts[layer_prefix + "o_proj_dksvd.weight"] = W_o_new_ckpt.to(target_dtype)

        del A_g, U_g, s_g_vec, Vh_g, U_grg, Sigma_grg_sqrt_diag, V_grg, WQ_new_g_math, WK_new_g_math
        del W_Q_math, W_K_math, W_V_math, W_qkv_dequant, W_o_dequant
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()

    # Construct the final state dict
    final_new_state_dict = {}

    # 1. Add the DK-SVD specific weights
    for key, value in new_dk_svd_state_dict_parts.items():
        final_new_state_dict[key] = value

    # 2. Identify original keys that were replaced or are now obsolete
    original_keys_handled_or_obsolete = set()
    original_model_uses_qk_norm = False
    if model_config.num_hidden_layers > 0:
        first_layer_q_norm_key = f'model.layers.0.self_attn.query_norm.weight'
        original_model_uses_qk_norm = first_layer_q_norm_key in original_state_dict
    print(f"Original model appears to {'use' if original_model_uses_qk_norm else 'not use'} QK normalization.")

    for i in range(model_config.num_hidden_layers):
        layer_prefix = f"model.layers.{i}.self_attn."
        original_keys_handled_or_obsolete.add(layer_prefix + "qkv_proj.weight")
        if input_is_quantized:
            original_keys_handled_or_obsolete.add(layer_prefix + "qkv_proj.weight_scaler")
        original_keys_handled_or_obsolete.add(layer_prefix + "o_proj.weight")
        if input_is_quantized:
            original_keys_handled_or_obsolete.add(layer_prefix + "o_proj.weight_scaler")

        # If original model used QK norm, these old norm weights are now obsolete
        # as the DK-SVD model will initialize new ones for the new rank_r dimension
        if original_model_uses_qk_norm:
            original_keys_handled_or_obsolete.add(layer_prefix + "query_norm.weight")
            original_keys_handled_or_obsolete.add(layer_prefix + "key_norm.weight")

    # 3. Copy all other weights from original_state_dict
    print("Copying remaining weights...")
    for key, value in original_state_dict.items():
        if key not in original_keys_handled_or_obsolete:
            final_new_state_dict[key] = value.to(target_dtype)
        # else: this key was part of a replaced module or is obsolete, so don't copy.

    # Prepare updated config for saving
    if 'config' in original_checkpoint and isinstance(original_checkpoint['config'], gemma_config.GemmaConfig):
        updated_model_config_dict = original_checkpoint['config'].__dict__.copy()
    elif 'config' in original_checkpoint and isinstance(original_checkpoint['config'], dict):
        updated_model_config_dict = original_checkpoint['config'].copy()
    else:
        updated_model_config_dict = model_config.__dict__.copy()

    updated_model_config_dict['use_dksvd'] = True
    updated_model_config_dict['dksvd_rank'] = rank_r
    # DK-SVD converted parts are float; if original was quant, this means mixed precision.
    # The 'quant' flag in config should reflect overall quantization strategy.
    # For now, let's set it to False, implying the DK-SVD model is primarily float.
    # A more advanced setup might track per-layer quantization.
    updated_model_config_dict['quant'] = False
    updated_model_config_dict['dtype'] = FLAGS.dtype

    # Ensure use_qk_norm is correctly set for the DK-SVD model.
    # If original Gemma 1b config has use_qk_norm=True, we keep it True.
    # The DK-SVD model will then initialize RMSNorm for rank_r.
    # The get_model_config for "1b" already sets use_qk_norm=True.
    # So, `updated_model_config_dict['use_qk_norm']` will already be True if derived from model_config.

    new_checkpoint_data = {
        'model_state_dict': final_new_state_dict,
        'config': updated_model_config_dict,
        'dksvd_metadata': {
            'is_dksvd_model': True,
            'dksvd_rank': rank_r,
            'original_variant': FLAGS.variant,
            'source_checkpoint_name': os.path.basename(FLAGS.ckpt_path)
        }
    }

    print(f"Saving DK-SVD compressed checkpoint to: {FLAGS.output_path}")
    torch.save(new_checkpoint_data, FLAGS.output_path)
    print("Conversion complete.")
    print(f"Output checkpoint keys ({len(final_new_state_dict.keys())}):")
    # for k in sorted(final_new_state_dict.keys())[:20]: # Print some keys
    #     print(f"  {k}: {final_new_state_dict[k].shape}, {final_new_state_dict[k].dtype}")


if __name__ == '__main__':
    # No need to mark flags as required if they have defaults or are checked in code.
    # FLAGS(sys.argv) # This is how absl.flags typically parses arguments
    app.run(main)