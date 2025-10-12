from __future__ import annotations
import math, torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RoPECache, apply_rope_single
from .kvcache import KVCache

class CausalSelfAttentionMLA(nn.Module):
    """
    Multi-Head Latent Attention:
      - Queries are standard multi-head (H heads, dim Dh)
      - Keys/Values live in a compact latent space (Lh heads, dim Dl << Dh)
      - Only latent K/V are cached; at use-time we expand latent K/V -> per-head dims
    """
    def __init__(
        self,
        d_model: int, n_head: int,
        d_head: int | None = None,  # if None, d_model // n_head
        d_latent: int = 64,         # must be even for RoPE on latent keys
        dropout: float = 0.0, rope: bool = True, max_pos: int = 32768,
    ):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.n_head = n_head
        self.d_head = d_head or (d_model // n_head)
        assert d_model == self.n_head * self.d_head, "d_model must equal n_head*d_head"
        self.d_latent = d_latent

        # Projections: Q in full head space; K/V in single latent head
        self.wq = nn.Linear(d_model, self.n_head * self.d_head, bias=False)
        self.wk_lat = nn.Linear(d_model, self.d_latent, bias=False)
        self.wv_lat = nn.Linear(d_model, self.d_latent, bias=False)

        # Per-head expansion from latent -> per-head dim
        self.k_expand = nn.Parameter(torch.empty(self.n_head, self.d_latent, self.d_head))
        self.v_expand = nn.Parameter(torch.empty(self.n_head, self.d_latent, self.d_head))
        nn.init.xavier_uniform_(self.k_expand)
        nn.init.xavier_uniform_(self.v_expand)

        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

        # RoPE
        self.use_rope = rope
        self.max_pos = max_pos
        self.rope_cache_q: RoPECache | None = None  # Dh
        self.rope_cache_k: RoPECache | None = None  # Dl

    def _maybe_init_rope(self, device):
        if not self.use_rope:
            return
        if self.rope_cache_q is None:
            assert self.d_head % 2 == 0, "d_head must be even for RoPE"
            self.rope_cache_q = RoPECache(self.d_head, self.max_pos, device=device)
        if self.rope_cache_k is None:
            assert self.d_latent % 2 == 0, "d_latent must be even for RoPE on latent keys"
            self.rope_cache_k = RoPECache(self.d_latent, self.max_pos, device=device)

    def _expand_latent_to_heads(self, k_lat: torch.Tensor, v_lat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        k_lat, v_lat: (B, 1, T, Dl)
        Returns: (B, H, T, Dh)
        """
        B, _, T, Dl = k_lat.shape
        # repeat latent across heads
        k_rep = k_lat.expand(B, self.n_head, T, Dl)  # (B,H,T,Dl)
        v_rep = v_lat.expand(B, self.n_head, T, Dl)  # (B,H,T,Dl)
        # per-head linear
        k_exp = torch.einsum("bhtd,hdm->bhtm", k_rep, self.k_expand)  # (B,H,T,Dh)
        v_exp = torch.einsum("bhtd,hdm->bhtm", v_rep, self.v_expand)  # (B,H,T,Dh)
        return k_exp, v_exp

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, start_pos: int = 0):
        """
        x: (B,T,C=n_embd)
        kv_cache: stores LATENT K/V only (B, 1, Tpast, Dl)
        """
        B, T, C = x.shape
        self._maybe_init_rope(x.device)

        # Projections
        q = self.wq(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)  # (B,H,T,Dh)
        k_lat = self.wk_lat(x).view(B, T, 1, self.d_latent).transpose(1, 2)  # (B,1,T,Dl)
        v_lat = self.wv_lat(x).view(B, T, 1, self.d_latent).transpose(1, 2)  # (B,1,T,Dl)

        # RoPE
        if self.use_rope:
            pos = torch.arange(start_pos, start_pos + T, device=x.device)
            cos_q, sin_q = self.rope_cache_q.get(pos)  # (T, Dh/2)
            q = apply_rope_single(q, cos_q, sin_q)
            cos_k, sin_k = self.rope_cache_k.get(pos)  # (T, Dl/2)
            k_lat = apply_rope_single(k_lat, cos_k, sin_k)

        # Concatenate latent cache
        if kv_cache is not None:
            k_lat_all = torch.cat([kv_cache.k, k_lat], dim=2)  # (B,1,Tpast+T,Dl)
            v_lat_all = torch.cat([kv_cache.v, v_lat], dim=2)
        else:
            k_lat_all, v_lat_all = k_lat, v_lat

        # Expand to per-head for attention
        k_attn, v_attn = self._expand_latent_to_heads(k_lat_all, v_lat_all)  # (B,H,Tk,Dh)

        # Scaled dot-product attention
        is_causal = kv_cache is None  # follow your original convention
        y = F.scaled_dot_product_attention(q, k_attn, v_attn, attn_mask=None, dropout_p=self.dropout.p if self.training else 0.0, is_causal=is_causal)  # (B,H,T,Dh)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.proj(y)

        # Update latent cache
        if kv_cache is not None:
            k_new = torch.cat([kv_cache.k, k_lat], dim=2)
            v_new = torch.cat([kv_cache.v, v_lat], dim=2)
        else:
            k_new, v_new = k_lat, v_lat

        return y, KVCache(k_new, v_new)