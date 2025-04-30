# -*- coding: utf-8 -*-
"""Pickle-able containers for all per-layer / per-head analytical data."""
import pickle, torch
from dataclasses import dataclass, field

from typing import Optional

@dataclass
class HeadAux:
    A:      torch.Tensor            # (d_k , d_k) - Orthogonal gauge matrix
    A_invT: torch.Tensor            # (d_k , d_k) - Should be same as A for orthogonal gauge
    P_r:    torch.Tensor            # (d_k , r_k) - Core projector (from right singular vectors of gauged interaction)
    P_a:    Optional[torch.Tensor]  # (r_k , r_a) - CP Factor A, Optional (None if SKIP_CP)
    P_b:    Optional[torch.Tensor]  # (r_k , r_b) - CP Factor B, Optional (None if SKIP_CP)
    alpha:  float
    lam:    float

@dataclass
class LayerAux:
    Z_r: torch.Tensor         # (d_v , r_v)

@dataclass
class GCBMeta:
    heads:  dict = field(default_factory=dict)   # key = (layer , head)
    layers: dict = field(default_factory=dict)   # key =  layer

    # ---- helpers ----------------------------------------------------
    def add_head(self, l, h, *args): self.heads[(l, h)] = HeadAux(*args)
    def save(self, path):  pickle.dump(self, open(path, "wb"))
    @staticmethod
    def load(path):        return pickle.load(open(path, "rb"))