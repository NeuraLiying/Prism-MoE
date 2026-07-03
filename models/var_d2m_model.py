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
                                                                                   

    def __init__(
        self,
        hidden_size: int,
        n_experts: int,
        topk: int,
        hard_mode: bool = False,
        bias: bool = False,
        router_temp: float = 1.0,
        delta_hidden_mult: float = 0.25,
        init_alpha: float = 0.0,
        context_mode: str = "none",
        context_num_stages: int = 0,
        context_init_alpha: float = 0.1,
        context_interaction_rank: int = 0,
        context_interaction_init_alpha: float = 0.1,
        context_cosine: bool = False,
        context_cosine_init_alpha: float = 0.0,
        token_cosine: bool = False,
        token_cosine_init_alpha: float = 0.0,
        logit_mode: str = "linear",
        cosine_tau: float = 10.0,
    ):
        super().__init__()
        assert 0 < topk <= n_experts
        assert context_mode in {"none", "cond", "cond_stage", "cond_stage_branch"}
        assert logit_mode in {"linear", "cosine"}
        self.n_experts = n_experts
        self.topk = topk
        self.hard_mode = hard_mode
        self.router_temp = float(router_temp)
        self.context_mode = context_mode
        self.logit_mode = str(logit_mode)
        self.cosine_tau = float(cosine_tau)

                                                                               
                                               
        self.proj = nn.Linear(hidden_size, n_experts, bias=bias)
        hidden = max(16, int(hidden_size * float(delta_hidden_mult)))
        self.delta = nn.Sequential(
            nn.Linear(hidden_size, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, n_experts, bias=False),
        )
        nn.init.zeros_(self.delta[-1].weight)
        self.alpha = nn.Parameter(torch.tensor(float(init_alpha)))

        use_cond = context_mode in {"cond", "cond_stage", "cond_stage_branch"}
        use_stage = context_mode in {"cond_stage", "cond_stage_branch"}
        use_branch = context_mode == "cond_stage_branch"
        self.cond_proj = nn.Linear(hidden_size, n_experts, bias=False) if use_cond else None
        self.stage_embed = nn.Embedding(int(context_num_stages), n_experts) if use_stage else None
        self.branch_embed = nn.Embedding(2, n_experts) if use_branch else None
        if self.cond_proj is not None:
            nn.init.zeros_(self.cond_proj.weight)
        if self.stage_embed is not None:
            nn.init.zeros_(self.stage_embed.weight)
        if self.branch_embed is not None:
            nn.init.zeros_(self.branch_embed.weight)
        self.context_alpha = nn.Parameter(torch.tensor(float(context_init_alpha))) if context_mode != "none" else None

        interaction_rank = int(context_interaction_rank)
        use_interaction = use_cond and interaction_rank > 0
        self.cond_token_proj = nn.Linear(hidden_size, interaction_rank, bias=False) if use_interaction else None
        self.cond_context_proj = nn.Linear(hidden_size, interaction_rank, bias=False) if use_interaction else None
        self.cond_interaction_out = nn.Linear(interaction_rank, n_experts, bias=False) if use_interaction else None
        self.cond_interaction_alpha = (
            nn.Parameter(torch.tensor(float(context_interaction_init_alpha))) if use_interaction else None
        )
        if self.cond_interaction_out is not None:
            nn.init.zeros_(self.cond_interaction_out.weight)

        use_cosine = use_cond and bool(context_cosine)
        self.cond_cosine_proto = nn.Parameter(torch.empty(n_experts, hidden_size)) if use_cosine else None
        self.cond_cosine_alpha = nn.Parameter(torch.tensor(float(context_cosine_init_alpha))) if use_cosine else None
        if self.cond_cosine_proto is not None:
            nn.init.normal_(self.cond_cosine_proto, std=hidden_size ** -0.5)

        self.token_cosine_proto = nn.Parameter(torch.empty(n_experts, hidden_size)) if bool(token_cosine) else None
        self.token_cosine_alpha = nn.Parameter(torch.tensor(float(token_cosine_init_alpha))) if bool(token_cosine) else None
        if self.token_cosine_proto is not None:
            nn.init.normal_(self.token_cosine_proto, std=hidden_size ** -0.5)

        self.register_buffer("dynamic_bias", torch.zeros(n_experts), persistent=True)
        self.register_buffer(
            "dynamic_stage_bias",
            torch.zeros(max(0, int(context_num_stages)), n_experts),
            persistent=True,
        )

        self.last_logits = None
        self.last_probs = None
        self.last_indices = None
        self.last_weights = None

    def forward_logits(
        self,
        x: torch.Tensor,
        context_logits: Optional[torch.Tensor] = None,
        dynamic_context_logits: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.logit_mode == "cosine":
            x_norm = F.normalize(x.float(), dim=-1)
            w_norm = F.normalize(self.proj.weight.float(), dim=-1)
            logits = F.linear(x_norm, w_norm).to(dtype=x.dtype) * self.cosine_tau
            delta_logits = torch.tanh(self.delta(x).float()).to(dtype=x.dtype)
            logits = logits + torch.tanh(self.alpha).to(dtype=x.dtype) * delta_logits
        else:
            logits = self.proj(x) + self.alpha.to(dtype=x.dtype) * self.delta(x)
        if self.token_cosine_proto is not None and self.token_cosine_alpha is not None:
            token_norm = F.normalize(x.float(), dim=-1)
            proto_norm = F.normalize(self.token_cosine_proto.float(), dim=-1)
            cosine_logits = token_norm @ proto_norm.t()
            logits = logits + self.token_cosine_alpha.to(dtype=x.dtype) * cosine_logits.to(dtype=x.dtype)
        if self.dynamic_bias.numel() > 0:
            logits = logits + self.dynamic_bias.to(dtype=x.dtype).view(1, -1)
        if dynamic_context_logits is not None:
            logits = logits + dynamic_context_logits.to(dtype=x.dtype)
        if context_logits is not None and self.context_alpha is not None:
            logits = logits + self.context_alpha.to(dtype=x.dtype) * context_logits.to(dtype=x.dtype)
        logits = logits / max(self.router_temp, 1e-6)
        self.last_logits = logits
        return logits

    def forward_topk(self, x: torch.Tensor, hard_mode: Optional[bool] = None,
                     context_logits: Optional[torch.Tensor] = None,
                     dynamic_context_logits: Optional[torch.Tensor] = None,
                     ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        if hard_mode is None:
            hard_mode = self.hard_mode

        logits = self.forward_logits(
            x,
            context_logits=context_logits,
            dynamic_context_logits=dynamic_context_logits,
        )                 
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
        mode: str = "moe",                       
        router_bias: bool = False,
        norm_topk_prob: bool = True,
        router_temp: float = 1.0,
        delta_hidden_mult: float = 0.25,
        init_alpha: float = 0.0,
        router_context_mode: str = "none",
        router_context_num_stages: int = 0,
        router_context_init_alpha: float = 0.1,
        router_context_interaction_rank: int = 0,
        router_context_interaction_init_alpha: float = 0.1,
        router_context_cosine: bool = False,
        router_context_cosine_init_alpha: float = 0.0,
        router_token_cosine: bool = False,
        router_token_cosine_init_alpha: float = 0.0,
        router_logit_mode: str = "linear",
        router_cosine_tau: float = 10.0,
        router_capture_input_sample_tokens: int = 0,
    ):
        super().__init__()
        assert n_experts > 0 and 0 < topk <= n_experts
        self.in_features = in_features
        self.shared_hidden = shared_hidden
        self.expert_hidden = expert_hidden
        self.n_experts = n_experts
        self.topk = topk
        self.hard_mode = hard_mode
        self.mode = mode                          
        self.norm_topk_prob = bool(norm_topk_prob)

        self.gate = D2MRouter(
            in_features,
            n_experts,
            topk=topk,
            hard_mode=hard_mode,
            bias=router_bias,
            router_temp=router_temp,
            delta_hidden_mult=delta_hidden_mult,
            init_alpha=init_alpha,
            context_mode=router_context_mode,
            context_num_stages=router_context_num_stages,
            context_init_alpha=router_context_init_alpha,
            context_interaction_rank=router_context_interaction_rank,
            context_interaction_init_alpha=router_context_interaction_init_alpha,
            context_cosine=router_context_cosine,
            context_cosine_init_alpha=router_context_cosine_init_alpha,
            token_cosine=router_token_cosine,
            token_cosine_init_alpha=router_token_cosine_init_alpha,
            logit_mode=router_logit_mode,
            cosine_tau=router_cosine_tau,
        )
        self.shared = VARExpert(in_features, shared_hidden, fc1_bias=True, fc2_bias=False)
        self.experts = nn.ModuleList(
            [VARExpert(in_features, expert_hidden, fc1_bias=True, fc2_bias=False) for _ in range(n_experts)]
        )

        self.drop = nn.Dropout(drop, inplace=True) if drop > 0 else nn.Identity()
        self.out_bias = nn.Parameter(torch.zeros(in_features, dtype=torch.float32))

        self.last_counts = None                  
        self.router_cond = None
        self.router_stage_ids = None
        self.router_branch_ids = None
        self.router_label_ids = None
        self.last_stage_ids_flat = None
        self.last_branch_ids_flat = None
        self.last_label_ids_flat = None
        self.router_capture_input_sample_tokens = int(router_capture_input_sample_tokens)
        self.last_input_sample = None
        self.last_input_sample_indices = None
        self.last_all_expert_outputs = None

    def set_mode(self, mode: str):
\
\
\
\
\
           
        assert mode in ("moe", "shared_only"), f"mode must be 'moe' or 'shared_only', got {mode}"
        self.mode = mode

    def set_router_context(
        self,
        cond: Optional[torch.Tensor] = None,
        stage_ids: Optional[torch.Tensor] = None,
        branch_ids: Optional[torch.Tensor] = None,
        label_ids: Optional[torch.Tensor] = None,
    ) -> None:
        self.router_cond = cond
        self.router_stage_ids = stage_ids
        self.router_branch_ids = branch_ids
        self.router_label_ids = label_ids

    def _expand_token_ids(
        self,
        ids: Optional[torch.Tensor],
        B: int,
        L: int,
        device: torch.device,
    ) -> Optional[torch.Tensor]:
        if ids is None:
            return None
        if not torch.is_tensor(ids):
            ids = torch.tensor(ids, device=device, dtype=torch.long)
        ids = ids.to(device=device, dtype=torch.long)
        if ids.ndim == 0:
            return ids.view(1, 1).expand(B, L).reshape(B * L)
        if ids.ndim == 1:
            if ids.shape[0] == L:
                return ids.view(1, L).expand(B, L).reshape(B * L)
            if ids.shape[0] == B:
                return ids.view(B, 1).expand(B, L).reshape(B * L)
        if ids.ndim == 2 and ids.shape == (B, L):
            return ids.reshape(B * L)
        return None

    def _context_logits(
        self,
        x_flat: torch.Tensor,
        orig_shape: torch.Size,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if len(orig_shape) != 3:
            self.last_stage_ids_flat = None
            self.last_branch_ids_flat = None
            self.last_label_ids_flat = None
            return None, None
        B, L, _ = orig_shape
        self.last_stage_ids_flat = self._expand_token_ids(self.router_stage_ids, B, L, x_flat.device)
        self.last_branch_ids_flat = self._expand_token_ids(self.router_branch_ids, B, L, x_flat.device)
        self.last_label_ids_flat = self._expand_token_ids(self.router_label_ids, B, L, x_flat.device)
        terms = []
        dynamic_terms = []

        if self.gate.dynamic_stage_bias.numel() > 0 and self.last_stage_ids_flat is not None:
            stage_ids = self.last_stage_ids_flat.clamp(0, self.gate.dynamic_stage_bias.shape[0] - 1)
            dynamic_terms.append(self.gate.dynamic_stage_bias.index_select(0, stage_ids).to(dtype=x_flat.dtype))

        if self.gate.cond_proj is not None and self.router_cond is not None:
            cond = self.router_cond
            if cond.shape[0] == B:
                cond = cond.to(dtype=x_flat.dtype)
                cond_logits = self.gate.cond_proj(cond)
                terms.append(cond_logits.unsqueeze(1).expand(B, L, -1).reshape(B * L, self.n_experts))
                if self.gate.cond_interaction_out is not None:
                    cond_flat = cond.unsqueeze(1).expand(B, L, -1).reshape(B * L, -1)
                    token_factor = self.gate.cond_token_proj(x_flat)
                    cond_factor = self.gate.cond_context_proj(cond_flat)
                    interaction = self.gate.cond_interaction_out(F.gelu(token_factor * cond_factor, approximate="tanh"))
                    scale = self.gate.cond_interaction_alpha.to(dtype=x_flat.dtype)
                    terms.append(scale * interaction)
                if self.gate.cond_cosine_proto is not None:
                    cond_norm = F.normalize(cond.float(), dim=-1)
                    proto_norm = F.normalize(self.gate.cond_cosine_proto.float(), dim=-1)
                    cosine_logits = cond_norm @ proto_norm.t()
                    scale = self.gate.cond_cosine_alpha.to(dtype=x_flat.dtype)
                    terms.append(
                        scale
                        * cosine_logits.to(dtype=x_flat.dtype).unsqueeze(1).expand(B, L, -1).reshape(B * L, self.n_experts)
                    )

        if self.gate.stage_embed is not None and self.router_stage_ids is not None:
            stage_ids = self.last_stage_ids_flat
            if stage_ids is not None:
                terms.append(self.gate.stage_embed(stage_ids))

        if self.gate.branch_embed is not None and self.router_branch_ids is not None:
            branch_ids = self.last_branch_ids_flat
            if branch_ids is not None:
                terms.append(self.gate.branch_embed(branch_ids))

        if self.last_stage_ids_flat is not None:
            self.last_stage_ids_flat = self.last_stage_ids_flat.detach()
        if self.last_branch_ids_flat is not None:
            self.last_branch_ids_flat = self.last_branch_ids_flat.detach()
        if self.last_label_ids_flat is not None:
            self.last_label_ids_flat = self.last_label_ids_flat.detach()

        context_logits = torch.stack(terms, dim=0).sum(dim=0) if terms else None
        dynamic_context_logits = torch.stack(dynamic_terms, dim=0).sum(dim=0) if dynamic_terms else None
        return context_logits, dynamic_context_logits

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

                                                             
        if self.mode == "shared_only":
                                                         
                                                        
            self.last_counts = None
            self.gate.last_logits = None
            self.gate.last_probs = None
            self.gate.last_indices = None
            self.gate.last_weights = None
            self.last_input_sample = None
            self.last_input_sample_indices = None
            self.last_all_expert_outputs = None
            y = y + self.out_bias.to(dtype=y.dtype)
            y = self.drop(y)
            if len(orig_shape) == 3:
                return y.reshape(B, L, C)
            return y

                                                
                               
        context_logits, dynamic_context_logits = self._context_logits(x_flat, orig_shape)
        weights, indices = self.gate.forward_topk(
            x_flat,
            hard_mode=self.hard_mode,
            context_logits=context_logits,
            dynamic_context_logits=dynamic_context_logits,
        )
        sample_tokens = int(self.router_capture_input_sample_tokens)
        if sample_tokens > 0:
            n_tokens = x_flat.shape[0]
            k_tokens = min(n_tokens, sample_tokens)
            if k_tokens == n_tokens:
                sample_idx = torch.arange(n_tokens, device=x_flat.device)
            else:
                sample_idx = torch.linspace(0, n_tokens - 1, k_tokens, device=x_flat.device).round().long()
            self.last_input_sample_indices = sample_idx.detach()
            self.last_input_sample = x_flat.index_select(0, sample_idx).detach()
            if self.training:
                self.last_all_expert_outputs = torch.stack(
                    [expert(self.last_input_sample) for expert in self.experts],
                    dim=1,
                )
            else:
                self.last_all_expert_outputs = None
        else:
            self.last_input_sample = None
            self.last_input_sample_indices = None
            self.last_all_expert_outputs = None
        if weights is not None and not self.norm_topk_prob:
            probs = self.gate.last_probs
            weights = probs.gather(1, indices).to(dtype=x_flat.dtype)
                                        
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
