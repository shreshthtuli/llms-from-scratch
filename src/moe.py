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
        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float32)
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        # --- Start: Auxiliary Loss Calculation ---
        S, E = routing_weights.size()
        importance = routing_weights.mean(0)
        hard_indices = top_k_indices[:, 0]
        load = F.one_hot(hard_indices, num_classes=E).float().mean(0)
        aux_loss = E * (importance * load).sum()
        # Normalize the weights of the top-k experts
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(x_flat.dtype)

        # Compute all expert activations: (S, num_experts, C)
        expert_outputs = torch.stack([expert(x_flat) for expert in self.experts], dim=1)

        # Gather the selected experts per token and apply routing weights
        gather_indices = top_k_indices.unsqueeze(-1).expand(-1, -1, C)
        selected_expert_outputs = expert_outputs.gather(1, gather_indices)
        weighted_expert_output = (selected_expert_outputs * top_k_weights.unsqueeze(-1)).sum(dim=1)

        shared_output = self.shared_expert(x_flat)
        final_output = weighted_expert_output + shared_output
        # Return the main output and the auxiliary loss
        return final_output.view(B, T, C), aux_loss
