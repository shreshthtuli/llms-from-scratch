import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, dim, mult=2.68, dropout=0.0):
        super().__init__()
        inner = int(mult * dim)
        self.w1 = nn.Linear(dim, inner, bias=False)
        self.w2 = nn.Linear(dim, inner, bias=False)
        self.w3 = nn.Linear(inner, dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        a = self.w1(x)
        b = self.act(self.w2(x))
        return self.dropout(self.w3(a * b))