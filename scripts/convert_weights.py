import torch
import gc
from absl import app, flags
import os
import sys

# Ensure the gemma module can be found if script is run from a different directory
# Assuming standard project structure where 'scripts' is a sibling of 'gemma'
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


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
        device_val = torch.device('cpu')
    elif FLAGS.device == 'cuda':
        device_val = torch.device('cuda')
    else:
        device_val = torch.device('cpu')

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
    if rank_r >= head_dim:
        print(f"Warning: rank_r ({rank_r}) is not less than head_dim ({head_dim}). Check if this is intended.")

    if num_key_value_heads != 1:
        print(f"CRITICAL Warning: This script is specifically designed for models with num_key_value_heads=1 "
              f"(like Gemma 1b which has a single GQA group). "
              f"Current model has num_key_value_heads={num_key_value_heads}. "
              f"The logic for forming A_g and handling weights will likely be INCORRECT. Proceed with caution or adapt the script.")
        # For a production script, one might raise an error here if N_kv != 1.
        # For now, let it proceed with the warning if user insists.

    # This is 's', the number of query heads per GQA group.
    # For num_key_value_heads = 1, s = num_attention_heads.
    # If num_key_value_heads > 1, this calculation of s_heads_per_group is correct,
    # but the subsequent reshaping of W_Q_math and handling of W_K_math would need
    # to be done *per group* in an outer loop.
    s_heads_per_group = num_attention_heads // num_key_value_heads


    for i in range(model_config.num_hidden_layers):
        print(f"Processing layer {i} on device {device_val}...")
        layer_prefix = f"model.layers.{i}.self_attn."

        # --- QKV Projection ---
        qkv_weight_key = layer_prefix + "qkv_proj.weight"
        W_qkv_ckpt_raw = original_state_dict[qkv_weight_key].to(device_val) # Move to device_val

        if input_is_quantized:
            qkv_scaler_key = layer_prefix + "qkv_proj.weight_scaler"
            qkv_scaler = original_state_dict[qkv_scaler_key].to(device_val)
            W_qkv_dequant = dequantize_weight(W_qkv_ckpt_raw, qkv_scaler).to(svd_dtype)
        else:
            W_qkv_dequant = W_qkv_ckpt_raw.to(svd_dtype)

        # W_qkv_dequant shape: ((num_heads + 2 * num_kv_heads) * head_dim, hidden_size)

        W_Q_ckpt_part = W_qkv_dequant[:W_Q_total_dim, :]
        W_K_ckpt_part = W_qkv_dequant[W_Q_total_dim : W_Q_total_dim + W_KV_total_dim, :]
        W_V_ckpt_part = W_qkv_dequant[W_Q_total_dim + W_KV_total_dim :, :]

        W_Q_math = W_Q_ckpt_part.T # (hidden_size, W_Q_total_dim)
        W_K_math = W_K_ckpt_part.T # (hidden_size, W_KV_total_dim)
        W_V_math = W_V_ckpt_part.T # (hidden_size, W_KV_total_dim)

        # Correct formation of A_g for a single GQA group (num_key_value_heads=1 assumed here based on Gemma 1b)
        # W_Q_math: (d, s*d_k) where s = num_attention_heads for the single group
        # W_K_math: (d, d_k) for the single group

        d_model = hidden_size # d in the paper
        d_k_val = head_dim    # d_k in the paper

        # Reshape W_Q_math to (s, d, d_k) to sum contributions from s query perspectives
        # W_Q_math shape is (d_model, s_heads_per_group * d_k_val)
        try:
            W_Q_for_sum_calc = W_Q_math.view(d_model, s_heads_per_group, d_k_val).permute(1, 0, 2)
        except RuntimeError as e:
            print(f"Error reshaping W_Q_math: {W_Q_math.shape} with s_heads_per_group={s_heads_per_group}, d_k_val={d_k_val}")
            print("This usually means num_key_value_heads > 1 and the script needs generalization for multiple GQA groups.")
            raise e

        # W_K_math shape is (d_model, d_k_val) since num_key_value_heads=1 means W_KV_total_dim = d_k_val
        W_K_math_T = W_K_math.T # shape (d_k_val, d_model)

        print(f"  Shapes for A_g: W_Q_for_sum_calc (s,d,d_k): {W_Q_for_sum_calc.shape}, W_K_math_T (d_k,d): {W_K_math_T.shape}")
        print(f"  Forming A_g = sum_over_s ( W_Q_s @ W_K^T )...")

        # (s, d, d_k) @ (d_k, d) -> (s, d, d)
        A_g_components = torch.matmul(W_Q_for_sum_calc, W_K_math_T)
        A_g = torch.sum(A_g_components, dim=0) # Sum over s components -> (d, d)

        print(f"  Performing SVD on A_g (shape {A_g.shape})...")
        U_g, s_g_vec, Vh_g = torch.linalg.svd(A_g) # Vh_g is V.T

        U_grg = U_g[:, :rank_r]
        s_grg_select_vec = s_g_vec[:rank_r]
        # Clamp to avoid issues with very small or negative singular values if any (though unlikely for top ones)
        s_grg_vec_sqrt = torch.sqrt(torch.clamp(s_grg_select_vec, min=1e-12))
        Sigma_grg_sqrt_diag = torch.diag(s_grg_vec_sqrt)

        V_grg = Vh_g[:rank_r, :].T # V_grg is (d_model, rank_r)

        WQ_new_g_math = U_grg @ Sigma_grg_sqrt_diag # (d_model, rank_r)
        WK_new_g_math = V_grg @ Sigma_grg_sqrt_diag # (d_model, rank_r)

        new_dk_svd_state_dict_parts[layer_prefix + "q_proj_dksvd.weight"] = WQ_new_g_math.T.to(target_dtype)
        new_dk_svd_state_dict_parts[layer_prefix + "k_proj_dksvd.weight"] = WK_new_g_math.T.to(target_dtype)
        # W_V_math is (d_model, d_k_val) for the single group. v_proj is Linear(d_model, d_k_val)
        new_dk_svd_state_dict_parts[layer_prefix + "v_proj.weight"] = W_V_math.T.to(target_dtype)


        # --- O Projection ---
        o_proj_weight_key = layer_prefix + "o_proj.weight"
        W_o_ckpt_raw = original_state_dict[o_proj_weight_key].to(device_val)

        if input_is_quantized:
            o_proj_scaler_key = layer_prefix + "o_proj.weight_scaler"
            o_proj_scaler = original_state_dict[o_proj_scaler_key].to(device_val)
            W_o_dequant = dequantize_weight(W_o_ckpt_raw, o_proj_scaler).to(svd_dtype)
        else:
            W_o_dequant = W_o_ckpt_raw.to(svd_dtype)

        # W_o_dequant shape: (hidden_size, num_attention_heads * head_dim)
        # New o_proj input dim for single GQA group's output is head_dim (d_v which is head_dim)
        # So, o_proj_dksvd becomes Linear(head_dim, hidden_size)
        W_o_new_ckpt = W_o_dequant[:, :head_dim] # Takes the part of original W_o that corresponded to the first d_v channels
        new_dk_svd_state_dict_parts[layer_prefix + "o_proj_dksvd.weight"] = W_o_new_ckpt.to(target_dtype)

        del A_g, U_g, s_g_vec, Vh_g, U_grg, Sigma_grg_sqrt_diag, V_grg, WQ_new_g_math, WK_new_g_math
        del W_Q_math, W_K_math, W_V_math, W_Q_for_sum_calc, W_K_math_T, A_g_components
        del W_qkv_dequant, W_o_dequant
        if device_val.type == 'cuda':
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
    if model_config.num_hidden_layers > 0: # Check if layers exist before accessing
        # Check based on one of the qk_norm keys.
        # model_config.use_qk_norm directly tells us if the variant expects it.
        original_model_uses_qk_norm = getattr(model_config, 'use_qk_norm', False)

    print(f"Original model variant config indicates QK normalization: {original_model_uses_qk_norm}.")

    for i in range(model_config.num_hidden_layers):
        layer_prefix = f"model.layers.{i}.self_attn."
        original_keys_handled_or_obsolete.add(layer_prefix + "qkv_proj.weight")
        if input_is_quantized:
            original_keys_handled_or_obsolete.add(layer_prefix + "qkv_proj.weight_scaler")
        original_keys_handled_or_obsolete.add(layer_prefix + "o_proj.weight")
        if input_is_quantized:
            original_keys_handled_or_obsolete.add(layer_prefix + "o_proj.weight_scaler")

        if original_model_uses_qk_norm:
            original_keys_handled_or_obsolete.add(layer_prefix + "query_norm.weight")
            original_keys_handled_or_obsolete.add(layer_prefix + "key_norm.weight")

    # 3. Copy all other weights from original_state_dict
    print("Copying remaining weights...")
    for key, value in original_state_dict.items():
        if key not in original_keys_handled_or_obsolete:
            final_new_state_dict[key] = value.to(target_dtype)
        else:
            print(f"  Skipping original key (replaced/obsolete): {key}")

    # Prepare updated config for saving
    # Use the loaded model_config as the base, then update DK-SVD specific fields.
    # This ensures all other config fields from get_model_config() are preserved.
    updated_model_config_dict = model_config.__dict__.copy()

    updated_model_config_dict['use_dksvd'] = True
    updated_model_config_dict['dksvd_rank'] = rank_r
    updated_model_config_dict['quant'] = False # DK-SVD parts are currently float. Future work could re-quantize.
    updated_model_config_dict['dtype'] = FLAGS.dtype

    # Ensure 'config' in original_checkpoint is also updated if it was a dict or GemmaConfig object
    if 'config' in original_checkpoint:
        if isinstance(original_checkpoint['config'], gemma_config.GemmaConfig):
            # If it was a GemmaConfig object, we can create a new one or update dict
            original_checkpoint['config'] = gemma_config.GemmaConfig(**updated_model_config_dict)
        elif isinstance(original_checkpoint['config'], dict):
            original_checkpoint['config'].update(updated_model_config_dict)
        # If 'config' was something else, we might just store our updated_model_config_dict separately.
        # For robust saving, let's ensure we always save a dict for 'config'.
        config_to_save = updated_model_config_dict
    else: # If 'config' was not in original_checkpoint
        config_to_save = updated_model_config_dict


    new_checkpoint_data = {
        'model_state_dict': final_new_state_dict,
        'config': config_to_save, # Save the updated config dict
        'dksvd_metadata': {
            'is_dksvd_model': True,
            'dksvd_rank': rank_r,
            'original_variant': FLAGS.variant,
            'source_checkpoint_name': os.path.basename(FLAGS.ckpt_path)
        }
    }
    # If original checkpoint had other keys (like 'optimizer_state_dict'), they are not preserved here.
    # This script focuses on model weights and config for inference.

    print(f"Saving DK-SVD compressed checkpoint to: {FLAGS.output_path}")
    torch.save(new_checkpoint_data, FLAGS.output_path)
    print("Conversion complete.")
    # print(f"Output checkpoint keys ({len(final_new_state_dict.keys())}):")
    # for k_idx, k in enumerate(sorted(final_new_state_dict.keys())):
    #     if k_idx < 20 or k_idx > len(final_new_state_dict.keys()) - 5 : # Print some keys start/end
    #         print(f"  {k}: {final_new_state_dict[k].shape}, {final_new_state_dict[k].dtype}")


if __name__ == '__main__':
    app.run(main)