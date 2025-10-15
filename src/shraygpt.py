import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from .rmsnorm import RMSNorm
from .mla import CausalSelfAttentionMLA
from .moe import MoE

class Block(nn.Module):
    def __init__(self, d_model, n_head, d_head, max_pos, num_experts, num_experts_per_tok, dropout=0.0):
        super(Block, self).__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.mha = CausalSelfAttentionMLA(d_model, n_head, d_head, d_latent=64, max_pos=max_pos, dropout=dropout)
        self.moe = MoE(d_model, num_experts=num_experts, num_experts_per_tok=num_experts_per_tok, dropout=dropout)
    
    def forward(self, x, kv_cache=None, start_pos=0): # post-norm as done in the original Attention is All You Need paper
        a, kv_cache = self.mha(self.norm1(x), kv_cache=kv_cache, start_pos=start_pos)
        x = x + a
        moe_output, aux_loss = self.moe(self.norm2(x))
        x = x + moe_output
        return x, kv_cache, aux_loss

def split_params_for_muon(model):
    matrix_params = []   # use Muon
    other_params  = []   # use AdamW (bias, norms, embeddings, LM head, etc.)

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # Embeddings & LM head are typically excluded from Muon, even though embeddings are 2D.
        is_embedding = any(k in name.lower() for k in ["embed", "embedding"])
        is_lm_head   = name.lower().endswith("head.weight") or "lm_head" in name.lower()

        if (p.ndim == 2) and (not is_embedding) and (not is_lm_head):
            matrix_params.append(p)   # hidden-layer weight matrices: Wq, Wk, Wv, Wo, MLP weights, etc.
        else:
            other_params.append(p)    # biases, (RMS)Norm weights (1D), embeddings, lm_head, scalars/vectors
    return matrix_params, other_params

class ShrayGPT(L.LightningModule):
    def __init__(self, vocab_size, block_size, d_model, n_head, d_head, n_layers, num_experts, num_experts_per_tok, dropout=0.0):
        super(ShrayGPT, self).__init__()
        self.save_hyperparameters()

        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            Block(d_model, n_head, d_head, block_size, num_experts, num_experts_per_tok, dropout) for _ in range(n_layers)
        ])
        self.ln_f = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, kv_cache_list=None, start_pos=0):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence length exceeds block size"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)  # (1, T)
        x = self.tok_emb(idx)  # (B, T, d_model), embed the tokens
        x = self.dropout(x) # (B, T, d_model), apply dropout 

        new_caches = []
        total_aux_loss = 0.0
        for i, layer in enumerate(self.layers):
            cache = None if kv_cache_list is None else kv_cache_list[i]
            x, cache, aux_loss = layer(x, kv_cache=cache, start_pos=start_pos)  # (B, T, d_model)
            total_aux_loss += aux_loss
            new_caches.append(cache)
        
        x = self.ln_f(x)                   # (B, T, d_model)
        logits = self.head(x)              # (B, T, vocab_size)
        return logits, new_caches, total_aux_loss / len(self.layers)

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens, temperature=1.0, top_k=None):
        self.eval() 
        if prompt.size(1) > self.block_size:
            prompt = prompt[:, -self.block_size:]

        logits, kv_caches, _ = self(prompt, kv_cache_list=None, start_pos=0)  # prefill start_pos = 0
        cur_pos = prompt.size(1)

        for _ in range(max_new_tokens):
            last_token = prompt[:, -1:]                      # (1,1)
            step_logits, kv_caches, _ = self(last_token, kv_cache_list=kv_caches, start_pos=cur_pos)
            cur_pos += 1

            # sample from the last position
            logits_step = step_logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits_step, min(top_k, logits_step.size(-1)))
                logits_step[logits_step < v[:, [-1]]] = -float('Inf')
            probs = F.softmax(logits_step, dim=-1)
            prompt_next = torch.multinomial(probs, num_samples=1)  # (1,1)

            # append to sequence
            prompt = torch.cat([prompt, prompt_next], dim=1)

        return prompt

    @torch.no_grad()
    def generate_nocache(self, prompt: torch.Tensor, max_new_tokens=200, temperature=1.0, top_k=50):
        self.eval()
        B = prompt.size(0)
        device = prompt.device

        for _ in range(max_new_tokens):
            # Condition on last block_size tokens to respect positional tables
            if prompt.size(1) > self.block_size:
                prompt_cond = prompt[:, -self.block_size:]
            else:
                prompt_cond = prompt

            # No cache, start_pos=0 for the window
            logits, _, _ = self(prompt_cond, kv_cache_list=None, start_pos=0)  # (B, Tcond, V)
            logits = logits[:, -1, :] / max(temperature, 1e-6)          # (B, V)

            if top_k is not None:
                k = min(top_k, logits.size(-1))
                v, _ = torch.topk(logits, k)
                logits[logits < v[:, [-1]]] = -float('inf')

            probs = F.softmax(logits, dim=-1)
            prompt_next = torch.multinomial(probs, num_samples=1)          # (B, 1)
            prompt = torch.cat([prompt, prompt_next.to(device)], dim=1)          # (B, T+1)

        return prompt

    def _calculate_loss(self, logits, targets, aux_loss):
        B, T, C = logits.shape
        logits_view = logits.view(B * T, C)
        targets_view = targets.view(B * T)
        loss = F.cross_entropy(logits_view, targets_view)
        total_loss = loss + self.hparams.aux_loss_weight * aux_loss
        return total_loss, loss, aux_loss

    def training_step(self, batch, batch_idx):
        muon_opt, adamw_opt = self.optimizers()
        muon_opt.zero_grad(); adamw_opt.zero_grad()
        x, y = batch
        logits, _, aux_loss_ = self(x)
        total_loss, main_loss, aux_loss = self._calculate_loss(logits, y, aux_loss_)
        self.manual_backward(total_loss)
        muon_opt.step();  adamw_opt.step()
        muon_sched, adamw_sched = self.lr_schedulers()
        muon_sched.step(); adamw_sched.step()
        self.log('train_loss', main_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('train_aux_loss', aux_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits, _, aux_loss_ = self(x)
        total_loss, main_loss, aux_loss = self._calculate_loss(logits, y, aux_loss_)
        self.log('val_loss', main_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log('val_aux_loss', aux_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        return total_loss

    def configure_optimizers(self):
        matrix_params, other_params = split_params_for_muon(self)

        muon_opt = torch.optim.Muon(matrix_params, lr=self.hparams.learning_rate_muon, weight_decay=0.0)
        adamw_opt = torch.optim.AdamW(other_params, lr=self.hparams.learning_rate_adamw, weight_decay=1e-2)

        muon_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(muon_opt, T_0=2000, 
                        eta_min=self.hparams.learning_rate_muon / 2)
        adamw_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(adamw_opt, T_0=2000,
                        eta_min=self.hparams.learning_rate_adamw / 2)

        return (
            [muon_opt, adamw_opt],
            [
                {"scheduler": muon_sched, "interval": "epoch"},
                {"scheduler": adamw_sched, "interval": "epoch"},
            ],
        )