# debug_wrap.py  –  drop this next to run_edksvd.py and import it
import contextlib, torch
from types import SimpleNamespace

def _probe(name):
    "Return a hook that prints statistics of the tensor it sees."
    def hook(_, __, out):
        if isinstance(out, tuple):                       # (next_tokens, logits) head
            out = out[1]
        t = out.detach()
        print(f"{name:30} | shape {tuple(t.shape):18} "
              f"min {t.min():8.3g}  max {t.max():8.3g}  "
              f"mean {t.mean():8.3g}  std {t.std():8.3g}")
    return hook

@contextlib.contextmanager
def attach_probes(model, every_layer=False):
    """Add hooks to key blocks; disable with `with` scope exit."""
    h = []                                               # keep handles to remove later
    # --- high-level logits out of the sampler ----------
    h.append(model.sampler.register_forward_hook(_probe("sampler-logits")))

    # --- embeddings -----------------------------------
    h.append(model.embedder.register_forward_hook(_probe("embedder")))

    # --- per transformer layer -------------------------
    for li, layer in enumerate(model.model.layers):
        tag = f"L{li:02d}"
        h.append(layer.input_layernorm.register_forward_hook(_probe(f"{tag} input-ln")))
        h.append(layer.self_attn.register_forward_hook(_probe(f"{tag} attn-out")))
        h.append(layer.post_attention_layernorm.register_forward_hook(_probe(f"{tag} post-attn-ln")))
        if getattr(layer, "pre_feedforward_layernorm", None):
            h.append(layer.pre_feedforward_layernorm.register_forward_hook(_probe(f"{tag} pre-ff-ln")))
        h.append(layer.mlp.register_forward_hook(_probe(f"{tag} mlp-out")))
        if every_layer:
            break    # comment out to print *all* layers (gets very verbose)

    try:
        yield
    finally:                                            # remove hooks cleanly
        for handle in h:
            handle.remove()
