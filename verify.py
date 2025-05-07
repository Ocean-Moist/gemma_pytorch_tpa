# verify_edksvd.py
# ------------------------------------------------------------
# Quick sanity-checker for a Gemma-E-DK-SVD checkpoint.
#
#  • Counts remaining int8 tensors.
#  • Runs one forward pass to see if the model answers sensibly.
# ------------------------------------------------------------
import argparse, torch, json, os, gc
from gemma import config as gemma_config
from gemma import model_dksvd as gemma_dksvd

def load_state(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    return state

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--prompt", default="Hello, Gemma!")
    ap.add_argument("--out_len", type=int, default=32)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    print(f"✓ loading checkpoint: {args.ckpt}")
    state = load_state(args.ckpt)

    # ---------- 1) weight-type histogram ---------------------
    int8_cnt = sum(t.dtype == torch.int8 for t in state.values())
    fp_cnt   = sum(t.dtype != torch.int8 for t in state.values())
    print(f"→ tensors   fp={fp_cnt:,}   int8={int8_cnt:,}")
    if int8_cnt:
        print("⚠  Some weights are still int8 – de-quantisation incomplete!")

    # ---------- 2) tiny generation test ----------------------
    cfg = gemma_config.get_config_for_1b("float32")
    cfg.qk_rank = args.rank
    cfg.quant   = False                      # we expect pure fp weights
    model = gemma_dksvd.GemmaForCausalLMDKSVD(cfg)
    missing, _ = model.load_state_dict(state, strict=False)
    if missing:
        print("⚠  Missing keys:", len(missing))
    model = model.to(args.device).eval()

    out = model.generate(
        prompts=args.prompt,
        device=args.device,
        output_len=args.out_len,
        temperature=0.0,  # greedy
        top_p=1.0,
        top_k=1,
    )
    print("\n=== PROMPT ===")
    print(args.prompt)
    print("\n=== COMPLETION ===")
    print(out)

    del model, state
    gc.collect()

if __name__ == "__main__":
    main()
