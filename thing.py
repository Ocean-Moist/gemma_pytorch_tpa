import torch, math
from gemma import config as gcfg
from gemma import model_dksvd as gdk

layer = 0
cfg = gcfg.get_config_for_1b("float32"); cfg.qk_rank = 32
full  = torch.load("model.ckpt", map_location="cpu")["model_state_dict"]
edks  = torch.load("gemma_edksvd", map_location="cpu")["model_state_dict"]

def kernel(weight, heads, d_k, which):
    """pick 'Q' or 'K' block, return list of (d, d_k) tensors"""
    start = dict(Q=0, K=heads*d_k)[which]
    return [weight[start+i*d_k:start+(i+1)*d_k, :].T
            for i in range(heads)]

# original
W = full[f"model.layers.{layer}.self_attn.qkv_proj.weight"]
Q_full = kernel(W, cfg.num_attention_heads, cfg.head_dim, "Q")
K_full = kernel(W, cfg.num_key_value_heads, cfg.head_dim, "K")[0]

# compressed
K_tilde = edks[f"model.layers.{layer}.self_attn.k_linears.0.weight"].T
Qs_tilde = [edks[f"model.layers.{layer}.self_attn.q_linears.0.{i}.weight"].T
            for i in range(cfg.num_attention_heads)]

for i, (Q0, Qt) in enumerate(zip(Q_full, Qs_tilde)):
    err = (Q0@K_full.T - Qt@K_tilde.T).norm() / (Q0@K_full.T).norm()
    print(f"head {i} -- relative Frobenius error: {err:.3e}")