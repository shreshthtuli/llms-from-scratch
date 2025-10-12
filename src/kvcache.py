import torch
from dataclasses import dataclass

@dataclass
class KVCache:
    k: torch.Tensor  # (B,H,T,D)
    v: torch.Tensor  # (B,H,T,D)
    @property
    def T(self):
        return self.k.size(2)