                  
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class VARExpert(nn.Module):
\
\
\
       
    def __init__(self, in_features: int, hidden_features: int,
                 fc1_bias: bool = True, fc2_bias: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features, bias=fc1_bias)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, in_features, bias=fc2_bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class D2MRouter(nn.Module):
\
\
\
\
\
\
       
    def __init__(self, hidden_size: int, n_experts: int, topk: int, hard_mode: bool = False, bias: bool = False):
        super().__init__()
        assert 0 < topk <= n_experts
        self.n_experts = n_experts
        self.topk = topk
        self.hard_mode = hard_mode

                                                                                        
        self.proj = nn.Linear(hidden_size, n_experts, bias=bias)

                                                          
        self.last_logits = None
        self.last_probs = None
        self.last_indices = None
        self.last_weights = None

    def forward_logits(self, x: torch.Tensor) -> torch.Tensor:
                                                                 
                                                                             
        logits = self.proj(x)
        self.last_logits = logits
        return logits

    def forward_topk(self, x: torch.Tensor, hard_mode: Optional[bool] = None
                     ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if hard_mode is None:
            hard_mode = self.hard_mode

        logits = self.forward_logits(x)                 
        if hard_mode:
                                                           
            indices = torch.topk(logits.float(), self.topk, dim=-1).indices
            weights = None
            self.last_indices = indices.detach()
            self.last_weights = None
            self.last_probs = None
            return weights, indices

        probs = F.softmax(logits.float(), dim=-1)              
        indices = torch.topk(probs, self.topk, dim=-1).indices            
        w = probs.gather(1, indices)            

                                                    
        w = w / (w.sum(dim=-1, keepdim=True).clamp_min(1e-9))
        weights = w.to(dtype=x.dtype)

        self.last_probs = probs
        self.last_indices = indices.detach()
        self.last_weights = weights.detach()
        return weights, indices

    def forward(self, x: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
                                                      
        return self.forward_topk(x)


class VARD2MFFN(nn.Module):
\
\
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
        hard_mode: bool = False,
        router_bias: bool = False,
    ):
        super().__init__()
        assert n_experts > 0 and 0 < topk <= n_experts
        self.in_features = in_features
        self.shared_hidden = shared_hidden
        self.expert_hidden = expert_hidden
        self.n_experts = n_experts
        self.topk = topk
        self.hard_mode = hard_mode

        self.gate = D2MRouter(in_features, n_experts, topk=topk, hard_mode=hard_mode, bias=router_bias)
        self.shared = VARExpert(in_features, shared_hidden, fc1_bias=True, fc2_bias=False)
        self.experts = nn.ModuleList(
            [VARExpert(in_features, expert_hidden, fc1_bias=True, fc2_bias=False) for _ in range(n_experts)]
        )

        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
        self.register_buffer("out_bias", torch.zeros(in_features, dtype=torch.float32), persistent=True)

        self.last_counts = None                  

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        if x.dim() == 3:
            B, L, C = x.shape
            x_flat = x.reshape(B * L, C)
        elif x.dim() == 2:
            x_flat = x
        else:
            raise ValueError(f"Unsupported x shape: {x.shape}")

                     
        y = self.shared(x_flat)

                               
        weights, indices = self.gate.forward_topk(x_flat, hard_mode=self.hard_mode)
                                             
        counts = torch.bincount(indices.flatten(), minlength=self.n_experts).to(dtype=torch.float32)
        self.last_counts = counts.detach()

        out = torch.zeros_like(y)
                             
        for i in range(self.n_experts):
            mask = (indices == i)
            if not mask.any():
                continue
            token_idx, top_pos = mask.nonzero(as_tuple=True)
            out_i = self.experts[i](x_flat[token_idx])            
            if weights is not None:
                out_i = out_i * weights[token_idx, top_pos].unsqueeze(-1)
            out[token_idx] += out_i

        y = y + out
        y = y + self.out_bias.to(dtype=y.dtype)
        y = self.drop(y)

        if len(orig_shape) == 3:
            return y.reshape(B, L, C)
        return y
