from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class VARExpert(nn.Module):
\
\
\
       
    def __init__(self, in_features: int, hidden_features: int, fc1_bias: bool = True, fc2_bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=fc1_bias)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=fc2_bias)

    def hidden(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc1(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.hidden(x))


class D2MRouter(nn.Module):
\
\
       
    def __init__(self, in_features: int, n_experts: int, router_temp: float = 1.0, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(in_features, n_experts, bias=bias)
        self.router_temp = float(router_temp)
        self.last_logits = None
        self.last_probs = None
        self.last_indices = None
        self.last_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x) / max(self.router_temp, 1e-6)


class VARD2MFFN(nn.Module):
\
\
\
\
       
    def __init__(
        self,
        in_features: int,
        shared_hidden: int,
        expert_hidden: int,
        n_experts: int,
        topk: int,
        drop: float = 0.0,
        hard_mode: bool = True,
        norm_topk_prob: bool = False,
        router_temp: float = 1.0,
        router_bias: bool = False,
    ):
        super().__init__()
        self.in_features = int(in_features)
        self.shared_hidden = int(shared_hidden)
        self.expert_hidden = int(expert_hidden)
        self.n_experts = int(n_experts)
        self.topk = int(topk)
        self.hard_mode = bool(hard_mode)
        self.norm_topk_prob = bool(norm_topk_prob)

        self.shared = VARExpert(in_features, shared_hidden, fc1_bias=True, fc2_bias=False)
        self.experts = nn.ModuleList(
            [VARExpert(in_features, expert_hidden, fc1_bias=True, fc2_bias=False) for _ in range(n_experts)]
        )
        self.gate = D2MRouter(in_features, n_experts, router_temp=router_temp, bias=router_bias)
        self.drop = nn.Dropout(p=float(drop)) if drop and drop > 0 else nn.Identity()

        self.out_bias = nn.Parameter(torch.zeros(in_features), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            x2 = x.reshape(-1, orig_shape[-1])
        else:
            x2 = x

        y = self.shared(x2)
        logits = self.gate(x2)

        if self.hard_mode:
            indices = torch.topk(logits.float(), k=self.topk, dim=-1).indices          
            self.gate.last_logits = None
            self.gate.last_probs = None
            self.gate.last_indices = indices.detach()
            self.gate.last_weights = None
            out = torch.zeros_like(y)
            for e in range(self.n_experts):
                mask = (indices == e)
                if not mask.any():
                    continue
                token_idx, _ = mask.nonzero(as_tuple=True)
                out[token_idx] += self.experts[e](x2[token_idx])          
            y = y + out
        else:
            probs = torch.softmax(logits, dim=-1, dtype=torch.float32)

                                                  
            topk_p, topk_i = torch.topk(probs, k=self.topk, dim=-1)                  

            if self.norm_topk_prob:
                topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

            self.gate.last_logits = logits
            self.gate.last_probs = probs
            self.gate.last_indices = topk_i.detach()
            self.gate.last_weights = topk_p.detach()

            out = torch.zeros_like(y)
            for e in range(self.n_experts):
                mask = (topk_i == e)
                if not mask.any():
                    continue
                token_idx, slot_idx = mask.nonzero(as_tuple=True)
                w = topk_p[token_idx, slot_idx].to(dtype=y.dtype).unsqueeze(-1)
                out[token_idx] += self.experts[e](x2[token_idx]) * w
            y = y + out

        y = self.drop(y)
        y = y + self.out_bias.to(dtype=y.dtype, device=y.device)

        if x.dim() == 3:
            return y.reshape(orig_shape)
        return y
