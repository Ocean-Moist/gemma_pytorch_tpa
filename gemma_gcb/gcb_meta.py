# -*- coding: utf-8 -*-
"""Pickle-able containers for all per-layer / per-head analytical data."""
import pickle, torch
from dataclasses import dataclass, field

@dataclass
class HeadAux:
    A:      torch.Tensor      # (d_k , d_k)
    A_invT: torch.Tensor      # (d_k , d_k)
    P_r:    torch.Tensor      # (d_k , r_k)
    P_a:    torch.Tensor      # (r_k , r_a)
    P_b:    torch.Tensor      # (r_k , r_b)
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