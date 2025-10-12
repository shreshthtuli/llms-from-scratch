import torch
import torch.nn as nn
from torch.nn import functional as F
from .swiglu import SwiGLU

class MoE(nn.Module):
    """A sparse Top-k Mixture of Experts layer."""
    def __init__(self, dim: int, num_experts: int, num_experts_per_tok: int, mult: float = 2.68, dropout: float = 0.0):
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(dim, mult, dropout) for _ in range(num_experts)])
        self.shared_expert = SwiGLU(dim, mult, dropout)
        self.num_experts_per_tok = num_experts_per_tok

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (B*T, C)
        router_logits = self.gate(x_flat)
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        # --- Start: Auxiliary Loss Calculation ---
        S, E = routing_weights.size()
        importance = routing_weights.mean(0)
        hard_indices = top_k_indices[:, 0]
        load = F.one_hot(hard_indices, num_classes=E).float().mean(0)
        aux_loss = E * (importance * load).sum()
        # Normalize the weights of the top-k experts
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        # Initialize final output tensor
        sparse_output = torch.zeros_like(x_flat)
        # Get expert outputs and combine them
        expert_outputs = torch.stack([self.experts[i](x_flat) for i in range(len(self.experts))])
        # Create a mask for indexing
        idx = torch.arange(x_flat.shape[0]).to(x.device)
        # Weighted sum of the expert outputs
        for i in range(self.num_experts_per_tok):
            expert_idx = top_k_indices[:, i]
            weight = top_k_weights[:, i].unsqueeze(-1)
            sparse_output += weight * expert_outputs[expert_idx, idx]
        shared_output = self.shared_expert(x_flat)
        final_output = sparse_output + shared_output
        # Return the main output and the auxiliary loss
        return final_output.view(B, T, C), aux_loss