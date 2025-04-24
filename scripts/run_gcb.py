#!/usr/bin/env python3
import argparse, torch
from gemma import config as gcfg
from gemma_gcb.model import GemmaForCausalLM_GCB

# --------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', required=True, help='gauged Gemma .pt (from convert_weights)')
parser.add_argument('--meta', required=True, help='.pkl file from convert_weights')
parser.add_argument('--prompt', default="The quick brown fox")
parser.add_argument('--device', default="cuda")
parser.add_argument('--out_len', type=int, default=64)
parser.add_argument('--temperature', type=float, default=1.0)
parser.add_argument('--top_p', type=float, default=0.95)
parser.add_argument('--top_k', type=int, default=64)
args = parser.parse_args()

# --------------------------------------------------------------------------
cfg = gcfg.get_model_config('1b')
model = GemmaForCausalLM_GCB(cfg, args.meta, args.ckpt).to(args.device)

out = model.generate(
    args.prompt,
    device=args.device,
    out_len=args.out_len,
    temperature=args.temperature,
    top_p=args.top_p,
    top_k=args.top_k,
)

print("\n---  RESULT  ---------------------------------------------------")
print(out)
print("----------------------------------------------------------------")