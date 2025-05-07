# quick_check_edksvd.py  (≈ 45 lines)
# -----------------------------------------------------------
#  * counts true‑int8 tensors
#  * measures the numeric range of a few key weights
#  * prints min eigen‑value per layer (saved by convert_weights.py)
#  * runs a 1‑step forward pass to see whether NaNs appear
# -----------------------------------------------------------
import argparse, torch, gc, math
from gemma import config as gcfg, model_dksvd as gdk

def load_state(path):
    st = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(st, dict) and "model_state_dict" in st:
        st = st["model_state_dict"]
    return st

def tensor_stats(t):
    return float(t.min()), float(t.max()), float(t.abs().mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--rank", type=int, default=32)
    args = ap.parse_args()

    sd = load_state(args.ckpt)
    n_int8 = sum(t.dtype == torch.int8 for t in sd.values())
    print(f"int8 tensors remaining : {n_int8}")

    # Pick three representative weights ----------------------
    for k in [
        "model.layers.0.self_attn.k_linears.0.weight",
        "model.layers.0.self_attn.q_linears.0.0.weight",
        "model.layers.0.mlp.gate_proj.weight",
    ]:
        if k in sd:
            mn, mx, av = tensor_stats(sd[k])
            print(f"{k:55}  min {mn:8.4g}  max {mx:8.4g}  mean|x| {av:8.4g}")

    # Eigen‑value sanity (saved by convert_weights.py >= v2) --
    ev_keys = [k for k in sd if k.endswith(".edksvd_lambda")]
    if ev_keys:
        for k in sorted(ev_keys)[:3]:
            lam = sd[k]
            print(f"{k:55}  smallest λ = {float(lam.min()):.4e}")
    else:
        print("(no eigen‑value tensors found – update convert_weights.py?)")

    # One‑step forward, catch NaNs ---------------------------
    cfg = gcfg.get_config_for_1b("float32")
    cfg.qk_rank = args.rank
    cfg.quant   = False
    m = gdk.GemmaForCausalLMDKSVD(cfg)
    m.load_state_dict(sd, strict=False)
    x = torch.tensor([[1,2,3,4]], dtype=torch.int64)  # dummy ids
    pos = torch.arange(4)
    kv   = []
    for _ in range(cfg.num_hidden_layers):
        k = torch.zeros((1,32,cfg.num_key_value_heads,cfg.qk_rank))
        v = torch.zeros((1,32,cfg.num_key_value_heads,cfg.head_dim))
        kv.append((k,v))
    with torch.no_grad():
        out_tokens, _ = m(
            input_token_ids=x,
            input_positions=pos,
            kv_write_indices=pos,
            kv_caches=kv,
            mask = torch.triu(
                torch.full((1, 1, 4, 32), float("-inf")),
                diagonal=1,
            ),
            output_positions=torch.tensor([3]),
            temperatures=None,
            top_ps=torch.tensor([1.0]),
            top_ks=torch.tensor([1]),
        )
    if torch.isnan(out_tokens.float()).any():
        print("❌  NaNs produced in forward pass")
    else:
        print("✅  forward pass clean")

    del m, sd; gc.collect()

if __name__ == "__main__":
    main()