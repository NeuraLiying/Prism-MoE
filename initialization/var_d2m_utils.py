                  
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, List, Optional, Tuple, Any, Dict
from tqdm import tqdm

from var_d2m_model import VARD2MFFN, D2MRouter


LossFn = Callable[[torch.Tensor, Any], torch.Tensor]                                                   


def _unpack_token_payload(payload: Any):
\
\
\
\
       
    if isinstance(payload, dict):
        tokens = payload.get("tokens_BLCv")
        targets = payload.get("gt_BL")
        seed = payload.get("seed", 0)
        if torch.is_tensor(tokens) and torch.is_tensor(targets):
            return tokens, targets, int(seed)

    if isinstance(payload, tuple):
        if len(payload) == 3 and torch.is_tensor(payload[0]) and torch.is_tensor(payload[1]):
            return payload[0], payload[1], int(payload[2])
        if len(payload) == 2 and torch.is_tensor(payload[0]) and torch.is_tensor(payload[1]):
            return payload[0], payload[1], 0

    return None, None, int(payload)


def build_var_loss_fn(var, vae, mode: str, args) -> LossFn:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    if mode == "custom_stub":
        def _stub(label_B: torch.Tensor, seed: int) -> torch.Tensor:
            raise RuntimeError(
                "No differentiable loss is available. "
                "You must implement build_var_loss_fn() using your VAR training forward "
                "(e.g., CE on next tokens / teacher-forcing)."
            )
        return _stub

    if mode == "model":
                             
        def _model_loss(label_B: torch.Tensor, seed: int) -> torch.Tensor:
                                                                                 
                                                    
            if hasattr(var, "forward_loss"):
                return var.forward_loss(label_B=label_B, g_seed=seed, cfg=getattr(args, "cfg", 1.5))
            if hasattr(var, "compute_loss"):
                return var.compute_loss(label_B=label_B, g_seed=seed, cfg=getattr(args, "cfg", 1.5))
            raise RuntimeError(
                "mode=model selected but var.forward_loss / var.compute_loss not found. "
                "Switch to --loss_mode trainer and bind to your training pipeline."
            )
        return _model_loss

    if mode == "trainer":
                                                                      
                                                                          
                                                                      
                                                        
                                                    
                                             
                                     
                                                                      
        import torch.nn as nn
        loss_fn_ce = nn.CrossEntropyLoss(reduction='mean')
        
        def _trainer_loss(label_B: torch.Tensor, seed_or_payload: Any) -> torch.Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
               
            x_BLCv_wo_first_l, gt_BL, seed = _unpack_token_payload(seed_or_payload)
            B = label_B.shape[0]
            device = label_B.device
            label_B = label_B.to(device)
            
                                                                            
            torch.manual_seed(seed)

            if x_BLCv_wo_first_l is None or gt_BL is None:
                                                                                 
                                                                                
                H, W = 256, 256
                dummy_images = torch.rand(B, 3, H, W, device=device)
                gt_idx_Bl = vae.img_to_idxBl(dummy_images)
                gt_BL = torch.cat(gt_idx_Bl, dim=1)
                x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            else:
                x_BLCv_wo_first_l = x_BLCv_wo_first_l.to(device=device, non_blocking=True)
                gt_BL = gt_BL.to(device=device, non_blocking=True)
            
                                                              
            original_prog_si = var.prog_si
            var.prog_si = -1                  
            
                                                           
                                                                                                 
            logits_BLV = var(label_B, x_BLCv_wo_first_l)             
            
                                      
            var.prog_si = original_prog_si
            
                                      
                                                     
            V = logits_BLV.shape[-1]
            loss = loss_fn_ce(logits_BLV.view(-1, V), gt_BL.view(-1))
            
            return loss
        
        return _trainer_loss

    raise ValueError(f"Unknown loss mode: {mode}")


def compute_hidden_neuron_importance_taylor(
    model,
    layer_idx: int,
    calib_pairs: List[Tuple[torch.Tensor, int]],
    loss_fn: LossFn,
    device: torch.device,
) -> torch.Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    block = model.blocks[layer_idx]
    ffn = block.ffn
    fc1 = ffn.fc1
    fc2 = ffn.fc2

    H = fc1.out_features

                                                                  
                                                                           
                                                                  
                                          
    original_training = model.training
    original_requires_grad = {}
    for name, p in model.named_parameters():
        original_requires_grad[name] = p.requires_grad
    
                                                                          
                                                                       
    model.eval()
    
                                                     
    for p in model.parameters():
        p.requires_grad_(False)
    for p in fc1.parameters():
        p.requires_grad_(True)
    for p in fc2.parameters():
        p.requires_grad_(True)

    imp = torch.zeros(H, device=device, dtype=torch.float32)

    for (label_B, seed) in calib_pairs:
        model.zero_grad(set_to_none=True)
        loss = loss_fn(label_B, seed)
        if loss.dim() != 0:
            loss = loss.mean()
                                                                        
        loss.backward()                                                      

                                 
        if fc1.weight.grad is None or fc2.weight.grad is None:
            raise RuntimeError("Gradients not found. Ensure your loss_fn is differentiable and uses this FFN.")

        g1 = fc1.weight.grad.detach()
        w1 = fc1.weight.detach()
        imp_fc1 = (g1 * w1).abs().sum(dim=1)       

        imp_bias = 0.0
        if fc1.bias is not None and fc1.bias.grad is not None:
            imp_bias = (fc1.bias.grad.detach() * fc1.bias.detach()).abs()       

                                                      
        g2 = fc2.weight.grad.detach()
        w2 = fc2.weight.detach()
        imp_fc2 = (g2 * w2).abs().sum(dim=0)       

        imp += (imp_fc1 + imp_fc2 + imp_bias.float())

                                                                  
                                                              
                                                                  
    if original_training:
        model.train()
    else:
        model.eval()
    
                                           
    for name, p in model.named_parameters():
        p.requires_grad_(original_requires_grad[name])
    
                                                               
    return imp.detach().cpu()


def _split_rest_to_experts(
    rest_ordered: np.ndarray,
    n_experts: int,
    expert_assignment: str = "contiguous",
    ffn_fc1_weight: Optional[torch.Tensor] = None,
    assignment_features: Optional[torch.Tensor] = None,
    importance: Optional[torch.Tensor] = None,
    kmeans_iters: int = 8,
    kmeans_restarts: int = 1,
) -> List[np.ndarray]:
    remain = len(rest_ordered)
    assert remain % n_experts == 0, "Remaining hidden must be divisible by n_experts"
    expert_hidden = remain // n_experts

    if expert_assignment == "contiguous":
        return [
            rest_ordered[i * expert_hidden:(i + 1) * expert_hidden].astype(np.int64)
            for i in range(n_experts)
        ]

    if expert_assignment == "round_robin":
                                                                               
                                                                                
                                                                            
        return [
            rest_ordered[i::n_experts][:expert_hidden].astype(np.int64)
            for i in range(n_experts)
        ]

    if expert_assignment in {"balanced_kmeans", "trajectory_profile_kmeans"}:
        if importance is None:
            raise ValueError(f"{expert_assignment} requires importance")
        if expert_assignment == "trajectory_profile_kmeans":
            if assignment_features is None:
                raise ValueError("trajectory_profile_kmeans requires assignment_features")
            feature_matrix = assignment_features
        else:
            if ffn_fc1_weight is None:
                raise ValueError("balanced_kmeans requires ffn_fc1_weight")
            feature_matrix = ffn_fc1_weight
        return _split_rest_to_experts_balanced_kmeans(
            rest_ordered=rest_ordered,
            n_experts=n_experts,
            expert_hidden=expert_hidden,
            feature_matrix=feature_matrix,
            importance=importance,
            kmeans_iters=kmeans_iters,
            kmeans_restarts=kmeans_restarts,
        )

    raise ValueError(f"Unknown expert_assignment: {expert_assignment}")


def _split_rest_to_experts_balanced_kmeans(
    rest_ordered: np.ndarray,
    n_experts: int,
    expert_hidden: int,
    feature_matrix: torch.Tensor,
    importance: torch.Tensor,
    kmeans_iters: int = 8,
    kmeans_restarts: int = 1,
) -> List[np.ndarray]:
\
\
\
\
\
\
\
\
\
       
    rest_idx = torch.as_tensor(rest_ordered, dtype=torch.long, device="cpu")
    feature_cpu = feature_matrix.detach().float().cpu()
    X = feature_cpu.index_select(0, rest_idx)
    X = torch.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = F.normalize(X, p=2, dim=1)

    imp_all = importance.detach().float().cpu().index_select(0, rest_idx)
    imp_all = imp_all.clamp_min(0)
    if float(imp_all.max()) <= 0:
        imp_all = torch.ones_like(imp_all)
    imp_weight = torch.sqrt(imp_all / imp_all.mean().clamp_min(1e-12))

    N = X.shape[0]
    assert N == n_experts * expert_hidden

    def init_centers(first_idx: int) -> torch.Tensor:
                                                                              
                                                                              
        centers = [int(first_idx)]
        min_dist = torch.full((N,), 2.0, dtype=torch.float32)
        for _ in range(1, n_experts):
            last = centers[-1]
            sim = X @ X[last].unsqueeze(1)
            dist = (1.0 - sim.squeeze(1)).clamp_min(0)
            min_dist = torch.minimum(min_dist, dist)
            score = min_dist * imp_weight
            score[torch.as_tensor(centers, dtype=torch.long)] = -1.0
            centers.append(int(torch.argmax(score).item()))
        return F.normalize(X[torch.as_tensor(centers, dtype=torch.long)].clone(), p=2, dim=1)

    def run_capacity_kmeans(initial_centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, float]:
        C = initial_centers
        assignment = torch.full((N,), -1, dtype=torch.long)
        order = torch.arange(N, dtype=torch.long)                                 

        for _ in range(max(1, int(kmeans_iters))):
            sims = X @ C.t()
            assignment.fill_(-1)
            remaining = [expert_hidden for _ in range(n_experts)]

            for idx_t in order.tolist():
                pref = torch.argsort(sims[idx_t], descending=True).tolist()
                for e in pref:
                    if remaining[e] > 0:
                        assignment[idx_t] = e
                        remaining[e] -= 1
                        break

            new_centers = []
            for e in range(n_experts):
                mask = assignment == e
                if not mask.any():
                    new_centers.append(C[e])
                    continue
                x_e = X[mask]
                w_e = imp_weight[mask].unsqueeze(1)
                c = (x_e * w_e).sum(dim=0)
                c = F.normalize(c, p=2, dim=0)
                new_centers.append(c)
            C = torch.stack(new_centers, dim=0)

        sims = X @ C.t()
        assigned_sims = sims[torch.arange(N), assignment]
        compactness = float((assigned_sims * imp_weight).mean().item())
        return assignment.clone(), C, compactness

    n_restarts = max(1, int(kmeans_restarts))
    if n_restarts == 1:
        first_indices = [0]
    else:
        top_span = max(n_experts, min(N, n_experts * max(1, n_restarts)))
        first_indices = [0]
        for r in range(1, n_restarts):
            pos = int(round(r * (top_span - 1) / max(n_restarts - 1, 1)))
            if pos not in first_indices:
                first_indices.append(pos)
        pos = top_span
        while len(first_indices) < n_restarts and pos < N:
            first_indices.append(pos)
            pos += top_span

    best_assignment = None
    best_score = -float("inf")
    for first_idx in first_indices:
        assignment, _, score = run_capacity_kmeans(init_centers(first_idx))
        if score > best_score:
            best_score = score
            best_assignment = assignment

    assignment = best_assignment
    if assignment is None:
        raise RuntimeError("balanced_kmeans failed to produce an assignment")

    expert_idx_list = []
    for e in range(n_experts):
        local = torch.nonzero(assignment == e, as_tuple=True)[0]
        if local.numel() != expert_hidden:
            raise RuntimeError(
                f"balanced_kmeans produced expert {e} with {local.numel()} neurons; "
                f"expected {expert_hidden}"
            )
        expert_idx_list.append(rest_idx.index_select(0, local).numpy().astype(np.int64))
    return expert_idx_list


def split_hidden_by_importance(
    importance: torch.Tensor,
    shared_hidden: int,
    n_experts: int,
    expert_assignment: str = "contiguous",
    ffn_fc1_weight: Optional[torch.Tensor] = None,
    assignment_features: Optional[torch.Tensor] = None,
    kmeans_iters: int = 8,
    kmeans_restarts: int = 1,
) -> Tuple[np.ndarray, List[np.ndarray]]:
\
\
\
\
\
\
\
       
    imp = importance.numpy()
    order = np.argsort(-imp)              

    shared_idx = order[:shared_hidden]
    rest = order[shared_hidden:]

    expert_idx_list = _split_rest_to_experts(
        rest_ordered=rest,
        n_experts=n_experts,
        expert_assignment=expert_assignment,
        ffn_fc1_weight=ffn_fc1_weight,
        assignment_features=assignment_features,
        importance=importance,
        kmeans_iters=kmeans_iters,
        kmeans_restarts=kmeans_restarts,
    )

    return shared_idx.astype(np.int64), [x.astype(np.int64) for x in expert_idx_list]


def init_router_from_expert_weights(
    router: D2MRouter,
    ffn_fc1_weight: torch.Tensor,
    expert_indices_list: List[np.ndarray],
    device: torch.device,
    importance: Optional[torch.Tensor] = None,
) -> None:
\
\
\
\
\
       
    W = ffn_fc1_weight.to(device).float()         
    rows = []
    for idx in expert_indices_list:
        w = W[idx]            
        w = F.normalize(w, p=2, dim=1)
        if importance is not None:
            idx_t = torch.tensor(idx, device=device, dtype=torch.long)
            imp = importance.to(device=device, dtype=torch.float32).index_select(0, idx_t)
            imp = imp.clamp_min(0)
            if float(imp.max().item()) <= 0:
                imp = torch.ones_like(imp)
            imp = torch.sqrt(imp / imp.mean().clamp_min(1e-12)).view(-1, 1)
            c = (w * imp).sum(dim=0, keepdim=True)
        else:
            c = w.mean(dim=0, keepdim=True)
        c = F.normalize(c, p=2, dim=1)
        rows.append(c.squeeze(0))
    R = torch.stack(rows, dim=0)         
    with torch.no_grad():
        router.proj.weight.data = R.to(dtype=router.proj.weight.dtype).clone()
        if router.proj.bias is not None:
            router.proj.bias.zero_()


def _stable_solve_ridge(
    A: torch.Tensor,
    B: torch.Tensor,
    ridge_lambda: float,
    regularize_last: bool = False,
) -> torch.Tensor:
    D = A.shape[0]
    if regularize_last and D > 1:
        diag_for_scale = A.diag()[:-1]
    else:
        diag_for_scale = A.diag()
    diag_mean = diag_for_scale.mean().clamp_min(1e-8)
    reg = float(ridge_lambda) * diag_mean * torch.eye(D, device=A.device, dtype=A.dtype)
    if regularize_last and D > 1:
        reg[-1, -1] = 0.0
    A_reg = A + reg
    try:
        L = torch.linalg.cholesky(A_reg, upper=False)
        y = torch.linalg.solve_triangular(L, B, upper=False)
        return torch.linalg.solve_triangular(L.t(), y, upper=True)
    except RuntimeError:
        try:
            return torch.linalg.solve(A_reg, B)
        except RuntimeError:
            return torch.linalg.lstsq(A_reg, B).solution


def _scores_to_topk_membership(scores: torch.Tensor, topk: int) -> torch.Tensor:
                                                                                   
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got shape {tuple(scores.shape)}")
    k = max(1, min(int(topk), scores.shape[1]))
    topk_idx = torch.topk(scores, k=k, dim=1).indices
    target = torch.zeros_like(scores)
    target.scatter_(1, topk_idx, 1.0)
    return target


@torch.no_grad()
def fit_router_from_activation_energy(
    model,
    layer_idx: int,
    calib_pairs: List[Tuple[torch.Tensor, Any]],
    vae,
    expert_indices_list: List[np.ndarray],
    device: torch.device,
    max_tokens: int = 8192,
    ridge_lambda: float = 1e-2,
    target_transform: str = "log",
    target_metric: str = "activation_norm",
    topk: Optional[int] = None,
    fit_bias: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    if target_metric not in {"activation_norm", "output_norm", "activation_topk", "output_topk"}:
        raise ValueError(f"Unknown router target metric: {target_metric}")
    if target_metric.endswith("_topk") and topk is None:
        raise ValueError(f"topk must be provided for router target metric {target_metric}")

    block = model.blocks[layer_idx]
    ffn = block.ffn
    fc1 = ffn.fc1
    fc2 = ffn.fc2
    C = fc1.in_features
    E = len(expert_indices_list)

    D = C + (1 if fit_bias else 0)
    A = torch.zeros((D, D), device=device, dtype=torch.float32)
    B_stat = torch.zeros((D, E), device=device, dtype=torch.float32)
    total_tokens = 0
    tokens_per_call = max_tokens if max_tokens and max_tokens > 0 else 0

    expert_idx_t = [
        torch.tensor(idx, device=device, dtype=torch.long)
        for idx in expert_indices_list
    ]
    fc2_col_norm = fc2.weight.detach().float().to(device).norm(dim=0)

    def hook_ffn(_module, inp, _out):
        nonlocal total_tokens
        if len(inp) == 0:
            return
        x = inp[0].detach().reshape(-1, C).float()
        if tokens_per_call > 0 and x.shape[0] > tokens_per_call:
            token_idx = torch.randperm(x.shape[0], device=x.device)[:tokens_per_call]
            x = x.index_select(0, token_idx)

        targets = []
        for idx_t in expert_idx_t:
            h = F.linear(
                x,
                fc1.weight.detach().float().to(device).index_select(0, idx_t),
                fc1.bias.detach().float().to(device).index_select(0, idx_t) if fc1.bias is not None else None,
            )
            h = ffn.act(h)
            if target_metric in {"activation_norm", "activation_topk"}:
                                                                           
                                  
                norm = fc2_col_norm.index_select(0, idx_t).view(1, -1)
                score = (h.abs() * norm).sum(dim=1)
            elif target_metric in {"output_norm", "output_topk"}:
                fc2_slice = fc2.weight.detach().float().to(device).index_select(1, idx_t)
                score = F.linear(h, fc2_slice).norm(dim=1)
            else:
                raise AssertionError(target_metric)
            targets.append(score)

        T = torch.stack(targets, dim=1)
        if target_metric.endswith("_topk"):
            T = _scores_to_topk_membership(T, topk=topk)
        else:
            if target_transform == "log":
                T = torch.log1p(T)
            elif target_transform == "sqrt":
                T = torch.sqrt(T.clamp_min(0))
            elif target_transform == "none":
                pass
            else:
                raise ValueError(f"Unknown router target transform: {target_transform}")

                                                                                
        T = T - T.mean(dim=1, keepdim=True)

        if fit_bias:
            A[:C, :C].add_(x.t().mm(x))
            x_sum = x.sum(dim=0)
            A[:C, C].add_(x_sum)
            A[C, :C].add_(x_sum)
            A[C, C].add_(float(x.shape[0]))
            B_stat[:C].add_(x.t().mm(T))
            B_stat[C].add_(T.sum(dim=0))
        else:
            A.add_(x.t().mm(x))
            B_stat.add_(x.t().mm(T))
        total_tokens += int(x.shape[0])

    handle = ffn.register_forward_hook(hook_ffn)
    original_prog_si = model.prog_si

    try:
        model.eval()
        for label_B, seed_or_payload in calib_pairs:
            x_BLCv_wo_first_l, _gt_BL, seed = _unpack_token_payload(seed_or_payload)
            label_B = label_B.to(device=device, non_blocking=True)
            torch.manual_seed(seed)

            if x_BLCv_wo_first_l is None:
                batch_size = label_B.shape[0]
                dummy_images = torch.rand(batch_size, 3, 256, 256, device=device)
                gt_idx_Bl = vae.img_to_idxBl(dummy_images)
                x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            else:
                x_BLCv_wo_first_l = x_BLCv_wo_first_l.to(device=device, non_blocking=True)

            model.prog_si = -1
            _ = model(label_B, x_BLCv_wo_first_l)
    finally:
        model.prog_si = original_prog_si
        handle.remove()

    if total_tokens <= 0:
        raise RuntimeError(f"No router calibration tokens collected for layer {layer_idx}")

    W_t = _stable_solve_ridge(
        A=A,
        B=B_stat,
        ridge_lambda=ridge_lambda,
        regularize_last=fit_bias,
    )
    if fit_bias:
        W = W_t[:C].t().contiguous()
        bias = W_t[C].contiguous()
        row_scale = torch.sqrt(W.pow(2).sum(dim=1) + bias.pow(2)).clamp_min(1e-12)
        W = W / row_scale.view(-1, 1)
        bias = bias / row_scale
        return W.detach().cpu(), bias.detach().cpu()
    W = W_t.t().contiguous()
    W = F.normalize(W, p=2, dim=1)
    return W.detach().cpu(), None


@torch.no_grad()
def _autoregressive_run_for_router_stats(
    var_model,
    B: int,
    label_B: torch.LongTensor,
    g_seed: int,
    cfg: float,
    top_k: int,
    top_p: float,
    stage_state: Dict[str, int],
) -> None:
\
\
\
\
\
\
       
    from models.helpers import sample_with_top_k_top_p_

    var_model.rng.manual_seed(g_seed)
    rng = var_model.rng

    if label_B is None:
        label_B = torch.multinomial(
            var_model.uniform_prob,
            num_samples=B,
            replacement=True,
            generator=rng,
        ).reshape(B)
    elif isinstance(label_B, int):
        label_B = torch.full(
            (B,),
            fill_value=var_model.num_classes if label_B < 0 else label_B,
            device=var_model.lvl_1L.device,
        )

    label_B = label_B.to(device=var_model.lvl_1L.device, dtype=torch.long)
    sos = cond_BD = var_model.class_emb(
        torch.cat(
            (
                label_B,
                torch.full_like(label_B, fill_value=var_model.num_classes),
            ),
            dim=0,
        )
    )

    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC
    next_token_map = (
        sos.unsqueeze(1).expand(2 * B, var_model.first_l, -1)
        + var_model.pos_start.expand(2 * B, var_model.first_l, -1)
        + lvl_pos[:, :var_model.first_l]
    )

    cur_L = 0
    f_hat = sos.new_zeros(
        B,
        var_model.Cvae,
        var_model.patch_nums[-1],
        var_model.patch_nums[-1],
    )

    for block in var_model.blocks:
        block.attn.kv_caching(True)

    try:
        for si, pn in enumerate(var_model.patch_nums):
            stage_state["si"] = int(si)
            stage_state["pn"] = int(pn)

            ratio = si / var_model.num_stages_minus_1
            cur_L += pn * pn
            cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
            x = next_token_map

            for block in var_model.blocks:
                x = block(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

            logits_BlV = var_model.get_logits(x, cond_BD)
            t = cfg * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            if torch.any(torch.isnan(logits_BlV)) or torch.any(torch.isinf(logits_BlV)):
                logits_BlV = torch.nan_to_num(logits_BlV, nan=0.0, posinf=1e6, neginf=-1e6)
                logits_BlV = logits_BlV.clamp(min=-1e6, max=1e6)

            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=top_k,
                top_p=top_p,
                num_samples=1,
            )[:, :, 0]

            h_BChw = var_model.vae_quant_proxy[0].embedding(idx_Bl)
            h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)
            f_hat, next_token_map = var_model.vae_quant_proxy[0].get_next_autoregressive_input(
                si,
                len(var_model.patch_nums),
                f_hat,
                h_BChw,
            )
            if si != var_model.num_stages_minus_1:
                next_token_map = next_token_map.view(B, var_model.Cvae, -1).transpose(1, 2)
                next_token_map = (
                    var_model.word_embed(next_token_map)
                    + lvl_pos[:, cur_L:cur_L + var_model.patch_nums[si + 1] ** 2]
                )
                next_token_map = next_token_map.repeat(2, 1, 1)
    finally:
        stage_state["si"] = -1
        stage_state["pn"] = 1
        for block in var_model.blocks:
            block.attn.kv_caching(False)


@torch.no_grad()
def fit_routers_from_autoregressive_activation_energy(
    model,
    layer_expert_indices: Dict[int, List[np.ndarray]],
    device: torch.device,
    nsamples: int,
    batch_size: int,
    calib_seed: int,
    num_classes: int,
    cfg: float = 4.0,
    top_k: int = 900,
    top_p: float = 0.96,
    max_tokens_per_call: int = 8192,
    ridge_lambda: float = 1e-2,
    target_transform: str = "log",
    stage_weight: str = "uniform",
    target_metric: str = "activation_norm",
    topk: Optional[int] = None,
    fit_bias: bool = False,
) -> Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]]:
\
\
\
\
\
\
\
       
    if nsamples <= 0:
        raise ValueError("nsamples must be positive for trajectory router calibration")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive for trajectory router calibration")
    if stage_weight not in {"token", "uniform", "sqrt"}:
        raise ValueError(f"Unknown trajectory stage_weight: {stage_weight}")
    if target_metric not in {"activation_norm", "output_norm", "activation_topk", "output_topk"}:
        raise ValueError(f"Unknown router target metric: {target_metric}")
    if target_metric.endswith("_topk") and topk is None:
        raise ValueError(f"topk must be provided for router target metric {target_metric}")

    layer_ids = sorted(layer_expert_indices.keys())
    stats: Dict[int, Dict[str, Any]] = {}
    handles = []
    stage_state = {"si": -1, "pn": 1}
    tokens_per_call = max_tokens_per_call if max_tokens_per_call and max_tokens_per_call > 0 else 0

    def make_hook(layer_idx: int):
        ffn = model.blocks[layer_idx].ffn
        fc1 = ffn.fc1
        fc2 = ffn.fc2
        C = fc1.in_features
        expert_idx_t = [
            torch.tensor(idx, device=device, dtype=torch.long)
            for idx in layer_expert_indices[layer_idx]
        ]
        fc2_col_norm = fc2.weight.detach().float().to(device).norm(dim=0)
        stats[layer_idx] = {
            "A": torch.zeros((C + (1 if fit_bias else 0), C + (1 if fit_bias else 0)), device=device, dtype=torch.float32),
            "B": torch.zeros((C + (1 if fit_bias else 0), len(expert_idx_t)), device=device, dtype=torch.float32),
            "tokens": 0,
        }

        def hook_ffn(_module, inp, _out):
            if len(inp) == 0:
                return
            x = inp[0].detach().reshape(-1, C).float()
            if tokens_per_call > 0 and x.shape[0] > tokens_per_call:
                token_idx = torch.randperm(x.shape[0], device=x.device)[:tokens_per_call]
                x = x.index_select(0, token_idx)

            targets = []
            for idx_t in expert_idx_t:
                h = F.linear(
                    x,
                    fc1.weight.detach().float().to(device).index_select(0, idx_t),
                    fc1.bias.detach().float().to(device).index_select(0, idx_t)
                    if fc1.bias is not None else None,
                )
                h = ffn.act(h)
                if target_metric in {"activation_norm", "activation_topk"}:
                    norm = fc2_col_norm.index_select(0, idx_t).view(1, -1)
                    score = (h.abs() * norm).sum(dim=1)
                elif target_metric in {"output_norm", "output_topk"}:
                    fc2_slice = fc2.weight.detach().float().to(device).index_select(1, idx_t)
                    score = F.linear(h, fc2_slice).norm(dim=1)
                else:
                    raise AssertionError(target_metric)
                targets.append(score)
            T = torch.stack(targets, dim=1)

            if target_metric.endswith("_topk"):
                T = _scores_to_topk_membership(T, topk=topk)
            else:
                if target_transform == "log":
                    T = torch.log1p(T)
                elif target_transform == "sqrt":
                    T = torch.sqrt(T.clamp_min(0))
                elif target_transform == "none":
                    pass
                else:
                    raise ValueError(f"Unknown router target transform: {target_transform}")

            T = T - T.mean(dim=1, keepdim=True)

            pn = max(1, int(stage_state.get("pn", 1)))
            if stage_weight == "uniform":
                scale = 1.0 / float(pn * pn)
            elif stage_weight == "sqrt":
                scale = 1.0 / float(pn)
            else:
                scale = 1.0

            if fit_bias:
                A = stats[layer_idx]["A"]
                B = stats[layer_idx]["B"]
                A[:C, :C].add_(x.t().mm(x) * scale)
                x_sum = x.sum(dim=0) * scale
                A[:C, C].add_(x_sum)
                A[C, :C].add_(x_sum)
                A[C, C].add_(float(x.shape[0]) * scale)
                B[:C].add_(x.t().mm(T) * scale)
                B[C].add_(T.sum(dim=0) * scale)
            else:
                stats[layer_idx]["A"].add_(x.t().mm(x) * scale)
                stats[layer_idx]["B"].add_(x.t().mm(T) * scale)
            stats[layer_idx]["tokens"] += int(x.shape[0])

        return hook_ffn

    try:
        model.eval()
        for layer_idx in layer_ids:
            handles.append(model.blocks[layer_idx].ffn.register_forward_hook(make_hook(layer_idx)))

        g = torch.Generator(device="cpu")
        g.manual_seed(calib_seed)
        n_batches = (nsamples + batch_size - 1) // batch_size
        for bi in tqdm(range(n_batches), desc="Trajectory router calibration"):
            cur_bs = min(batch_size, nsamples - bi * batch_size)
            label_B = torch.randint(0, num_classes, (cur_bs,), generator=g).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            _autoregressive_run_for_router_stats(
                var_model=model,
                B=cur_bs,
                label_B=label_B,
                g_seed=calib_seed + bi,
                cfg=cfg,
                top_k=top_k,
                top_p=top_p,
                stage_state=stage_state,
            )
    finally:
        for handle in handles:
            handle.remove()
        stage_state["si"] = -1
        stage_state["pn"] = 1

    router_weights: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]] = {}
    for layer_idx in layer_ids:
        if stats[layer_idx]["tokens"] <= 0:
            raise RuntimeError(f"No trajectory router tokens collected for layer {layer_idx}")
        W_t = _stable_solve_ridge(
            A=stats[layer_idx]["A"],
            B=stats[layer_idx]["B"],
            ridge_lambda=ridge_lambda,
            regularize_last=fit_bias,
        )
        if fit_bias:
            C = model.blocks[layer_idx].ffn.fc1.in_features
            W = W_t[:C].t().contiguous()
            bias = W_t[C].contiguous()
            row_scale = torch.sqrt(W.pow(2).sum(dim=1) + bias.pow(2)).clamp_min(1e-12)
            W = W / row_scale.view(-1, 1)
            bias = bias / row_scale
            router_weights[layer_idx] = (W.detach().cpu(), bias.detach().cpu())
        else:
            W = F.normalize(W_t.t().contiguous(), p=2, dim=1)
            router_weights[layer_idx] = (W.detach().cpu(), None)
        print(
            f"  [Trajectory router] Layer {layer_idx}: "
            f"tokens={stats[layer_idx]['tokens']}, stage_weight={stage_weight}, "
            f"target_metric={target_metric}, fit_bias={fit_bias}"
        )

    return router_weights


def _router_balance_update_from_counts(
    counts: torch.Tensor,
    bias: Optional[torch.Tensor],
    strength: float,
    max_abs_bias: float,
    delta_linf_cap: float = 0.0,
    target_probs: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    counts_f = counts.detach().float().cpu().clamp_min(0)
    total = float(counts_f.sum().item())
    if total <= 0:
        raise RuntimeError("No router selections collected for balance calibration")

    E = counts_f.numel()
    freq = counts_f / max(total, 1.0)
    if target_probs is None:
        target_f = torch.full_like(freq, 1.0 / float(E))
    else:
        target_f = target_probs.detach().float().cpu().clamp_min(0)
        if target_f.numel() != E:
            raise ValueError(f"target_probs must have {E} entries, got {target_f.numel()}")
        target_total = float(target_f.sum().item())
        if target_total <= 0:
            raise RuntimeError("Router balance target probabilities sum to zero")
        target_f = target_f / target_total
    eps = 1.0 / max(total, 1.0)

    delta = float(strength) * (torch.log(target_f + eps) - torch.log(freq + eps))
    delta = delta - delta.mean()
    raw_delta_linf = float(delta.abs().max().item())
    delta_scale = 1.0
    if delta_linf_cap and float(delta_linf_cap) > 0 and raw_delta_linf > float(delta_linf_cap):
        delta_scale = float(delta_linf_cap) / max(raw_delta_linf, 1e-12)
        delta = delta * delta_scale

    if bias is None:
        new_bias = torch.zeros(E, dtype=torch.float32)
    else:
        new_bias = bias.detach().float().cpu().clone()
    new_bias = new_bias + delta
    new_bias = new_bias - new_bias.mean()
    if max_abs_bias > 0:
        new_bias = new_bias.clamp(min=-float(max_abs_bias), max=float(max_abs_bias))
        new_bias = new_bias - new_bias.mean()

    std = float(freq.std(unbiased=False).item())
    mean = float(freq.mean().item())
    target_std = float(target_f.std(unbiased=False).item())
    target_mean = float(target_f.mean().item())
    stats = {
        "total": total,
        "min_freq": float(freq.min().item()),
        "max_freq": float(freq.max().item()),
        "cv": std / max(mean, 1e-12),
        "target_min_freq": float(target_f.min().item()),
        "target_max_freq": float(target_f.max().item()),
        "target_cv": target_std / max(target_mean, 1e-12),
        "delta_linf": float(delta.abs().max().item()),
        "raw_delta_linf": raw_delta_linf,
        "delta_scale": float(delta_scale),
        "delta_linf_cap": float(delta_linf_cap),
        "effective_strength": float(strength) * float(delta_scale),
        "bias_linf": float(new_bias.abs().max().item()),
    }
    return new_bias, stats


def _build_ar_label_schedule(
    nsamples: int,
    num_classes: int,
    calib_seed: int,
    mode: str,
) -> Tuple[Optional[torch.Tensor], torch.Generator]:
\
\
\
\
\
\
       
    if mode not in {"random", "cycle", "stratified"}:
        raise ValueError(f"Unknown autoregressive label sampling mode: {mode}")
    g = torch.Generator(device="cpu")
    g.manual_seed(calib_seed)
    if mode == "random":
        return None, g
    if mode == "cycle":
        return torch.arange(nsamples, dtype=torch.long) % int(num_classes), g

    chunks = []
    remaining = int(nsamples)
    while remaining > 0:
        perm = torch.randperm(int(num_classes), generator=g, dtype=torch.long)
        take = min(remaining, int(num_classes))
        chunks.append(perm[:take])
        remaining -= take
    return torch.cat(chunks, dim=0), g


@torch.no_grad()
def calibrate_router_bias_from_teacher_forcing_counts(
    model,
    layer_router_weights: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]],
    calib_pairs: List[Tuple[torch.Tensor, Any]],
    device: torch.device,
    vae,
    topk: int,
    max_tokens_per_call: int = 8192,
    strength: float = 0.25,
    max_abs_bias: float = 2.0,
    delta_linf_cap: float = 0.0,
) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Dict[str, float]]]:
\
\
\
\
\
\
       
    if not layer_router_weights:
        return layer_router_weights, {}

    layer_ids = sorted(layer_router_weights.keys())
    counts: Dict[int, torch.Tensor] = {}
    handles = []
    tokens_per_call = max_tokens_per_call if max_tokens_per_call and max_tokens_per_call > 0 else 0

    def make_hook(layer_idx: int):
        ffn = model.blocks[layer_idx].ffn
        C = ffn.fc1.in_features
        W, b = layer_router_weights[layer_idx]
        W_d = W.detach().float().to(device)
        b_d = b.detach().float().to(device) if b is not None else None
        E = W_d.shape[0]
        counts[layer_idx] = torch.zeros(E, device=device, dtype=torch.float32)

        def hook_ffn(_module, inp, _out):
            if len(inp) == 0:
                return
            x = inp[0].detach().reshape(-1, C).float()
            if tokens_per_call > 0 and x.shape[0] > tokens_per_call:
                token_idx = torch.randperm(x.shape[0], device=x.device)[:tokens_per_call]
                x = x.index_select(0, token_idx)
            logits = x.matmul(W_d.t())
            if b_d is not None:
                logits = logits + b_d.view(1, -1)
            idx = torch.topk(logits.float(), k=topk, dim=-1).indices
            counts[layer_idx].add_(torch.bincount(idx.reshape(-1), minlength=E).float())

        return hook_ffn

    original_prog_si = model.prog_si
    try:
        model.eval()
        for layer_idx in layer_ids:
            handles.append(model.blocks[layer_idx].ffn.register_forward_hook(make_hook(layer_idx)))

        for label_B, seed_or_payload in tqdm(calib_pairs, desc="Router balance TF"):
            x_BLCv_wo_first_l, _gt_BL, seed = _unpack_token_payload(seed_or_payload)
            label_B = label_B.to(device=device, non_blocking=True)
            torch.manual_seed(seed)
            if x_BLCv_wo_first_l is None:
                batch_size = label_B.shape[0]
                dummy_images = torch.rand(batch_size, 3, 256, 256, device=device)
                gt_idx_Bl = vae.img_to_idxBl(dummy_images)
                x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            else:
                x_BLCv_wo_first_l = x_BLCv_wo_first_l.to(device=device, non_blocking=True)
            model.prog_si = -1
            _ = model(label_B, x_BLCv_wo_first_l)
    finally:
        model.prog_si = original_prog_si
        for handle in handles:
            handle.remove()

    updated: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    stats: Dict[int, Dict[str, float]] = {}
    for layer_idx in layer_ids:
        W, b = layer_router_weights[layer_idx]
        new_bias, layer_stats = _router_balance_update_from_counts(
            counts=counts[layer_idx],
            bias=b,
            strength=strength,
            max_abs_bias=max_abs_bias,
            delta_linf_cap=delta_linf_cap,
        )
        updated[layer_idx] = (W, new_bias)
        stats[layer_idx] = layer_stats
        print(
            f"  [Router balance TF] Layer {layer_idx}: "
            f"min={layer_stats['min_freq']:.4f}, max={layer_stats['max_freq']:.4f}, "
            f"cv={layer_stats['cv']:.3f}, bias_linf={layer_stats['bias_linf']:.3f}"
        )
    return updated, stats


@torch.no_grad()
def calibrate_router_bias_from_autoregressive_counts(
    model,
    layer_router_weights: Dict[int, Tuple[torch.Tensor, Optional[torch.Tensor]]],
    device: torch.device,
    nsamples: int,
    batch_size: int,
    calib_seed: int,
    num_classes: int,
    topk: int,
    cfg: float = 4.0,
    top_k: int = 900,
    top_p: float = 0.96,
    max_tokens_per_call: int = 8192,
    strength: float = 0.25,
    max_abs_bias: float = 2.0,
    stage_weight: str = "token",
    delta_linf_cap: float = 0.0,
    target_metric: str = "uniform",
    target_transform: str = "log",
    target_mix_uniform: float = 0.0,
    layer_expert_indices: Optional[Dict[int, List[np.ndarray]]] = None,
    layer_expert_weights: Optional[
        Dict[int, List[Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]]]
    ] = None,
    label_sampling: str = "random",
) -> Tuple[Dict[int, Tuple[torch.Tensor, torch.Tensor]], Dict[int, Dict[str, float]]]:
\
\
\
\
\
\
       
    if nsamples <= 0:
        raise ValueError("nsamples must be positive for trajectory router balance")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive for trajectory router balance")
    if stage_weight not in {"token", "uniform", "sqrt"}:
        raise ValueError(f"Unknown router balance stage_weight: {stage_weight}")
    if target_metric not in {"uniform", "activation_norm", "output_norm", "activation_topk", "output_topk"}:
        raise ValueError(f"Unknown router balance target_metric: {target_metric}")
    if target_transform not in {"log", "sqrt", "none"}:
        raise ValueError(f"Unknown router balance target_transform: {target_transform}")
    if label_sampling not in {"random", "cycle", "stratified"}:
        raise ValueError(f"Unknown router balance label_sampling: {label_sampling}")
    if not (0.0 <= float(target_mix_uniform) <= 1.0):
        raise ValueError("target_mix_uniform must be in [0, 1]")
    if target_metric != "uniform" and layer_expert_indices is None and layer_expert_weights is None:
        raise ValueError(
            "Contribution-aware router balance requires layer_expert_indices "
            "or layer_expert_weights"
        )
    if not layer_router_weights:
        return layer_router_weights, {}

    layer_ids = sorted(layer_router_weights.keys())
    counts: Dict[int, torch.Tensor] = {}
    target_scores: Dict[int, torch.Tensor] = {}
    handles = []
    stage_state = {"si": -1, "pn": 1}
    tokens_per_call = max_tokens_per_call if max_tokens_per_call and max_tokens_per_call > 0 else 0

    def make_hook(layer_idx: int):
        ffn = model.blocks[layer_idx].ffn
        C = ffn.fc1.in_features
        W, b = layer_router_weights[layer_idx]
        W_d = W.detach().float().to(device)
        b_d = b.detach().float().to(device) if b is not None else None
        E = W_d.shape[0]
        counts[layer_idx] = torch.zeros(E, device=device, dtype=torch.float32)
        if target_metric != "uniform":
            target_scores[layer_idx] = torch.zeros(E, device=device, dtype=torch.float32)

        expert_idx_t = None
        expert_weight_t = None
        if target_metric != "uniform":
            if layer_expert_weights is not None and layer_idx in layer_expert_weights:
                expert_weight_t = []
                for fc1_w, fc1_b, fc2_w in layer_expert_weights[layer_idx]:
                    expert_weight_t.append(
                        (
                            fc1_w.detach().float().to(device),
                            fc1_b.detach().float().to(device) if fc1_b is not None else None,
                            fc2_w.detach().float().to(device),
                        )
                    )
                if len(expert_weight_t) != E:
                    raise ValueError(
                        f"Layer {layer_idx}: target expert weight count {len(expert_weight_t)} "
                        f"does not match router experts {E}"
                    )
            elif layer_expert_indices is not None and layer_idx in layer_expert_indices:
                expert_idx_t = [
                    torch.tensor(idx, device=device, dtype=torch.long)
                    for idx in layer_expert_indices[layer_idx]
                ]
                if len(expert_idx_t) != E:
                    raise ValueError(
                        f"Layer {layer_idx}: target expert index count {len(expert_idx_t)} "
                        f"does not match router experts {E}"
                    )
            else:
                raise ValueError(f"Layer {layer_idx}: missing contribution target expert data")

        fc2_col_norm = (
            ffn.fc2.weight.detach().float().to(device).norm(dim=0)
            if target_metric.startswith("activation") and expert_weight_t is None
            else None
        )

        def hook_ffn(_module, inp, _out):
            if len(inp) == 0:
                return
            x = inp[0].detach().reshape(-1, C).float()
            if tokens_per_call > 0 and x.shape[0] > tokens_per_call:
                token_idx = torch.randperm(x.shape[0], device=x.device)[:tokens_per_call]
                x = x.index_select(0, token_idx)
            logits = x.matmul(W_d.t())
            if b_d is not None:
                logits = logits + b_d.view(1, -1)
            idx = torch.topk(logits.float(), k=topk, dim=-1).indices
            pn = max(1, int(stage_state.get("pn", 1)))
            if stage_weight == "uniform":
                scale = 1.0 / float(pn * pn)
            elif stage_weight == "sqrt":
                scale = 1.0 / float(pn)
            else:
                scale = 1.0
            counts[layer_idx].add_(
                torch.bincount(idx.reshape(-1), minlength=E).float() * scale
            )

            if target_metric == "uniform":
                return

            scores = []
            if expert_weight_t is not None:
                for fc1_w, fc1_b, fc2_w in expert_weight_t:
                    h = F.linear(x, fc1_w, fc1_b)
                    h = ffn.act(h)
                    if target_metric in {"activation_norm", "activation_topk"}:
                        norm = fc2_w.detach().float().norm(dim=0).view(1, -1)
                        score = (h.abs() * norm).sum(dim=1)
                    elif target_metric in {"output_norm", "output_topk"}:
                        score = F.linear(h, fc2_w).norm(dim=1)
                    else:
                        raise AssertionError(target_metric)
                    scores.append(score)
            else:
                assert expert_idx_t is not None
                fc1_w = ffn.fc1.weight.detach().float().to(device)
                fc1_b = ffn.fc1.bias.detach().float().to(device) if ffn.fc1.bias is not None else None
                fc2_w = ffn.fc2.weight.detach().float().to(device)
                assert fc2_col_norm is not None
                for idx_t in expert_idx_t:
                    h = F.linear(
                        x,
                        fc1_w.index_select(0, idx_t),
                        fc1_b.index_select(0, idx_t) if fc1_b is not None else None,
                    )
                    h = ffn.act(h)
                    if target_metric in {"activation_norm", "activation_topk"}:
                        norm = fc2_col_norm.index_select(0, idx_t).view(1, -1)
                        score = (h.abs() * norm).sum(dim=1)
                    elif target_metric in {"output_norm", "output_topk"}:
                        fc2_slice = fc2_w.index_select(1, idx_t)
                        score = F.linear(h, fc2_slice).norm(dim=1)
                    else:
                        raise AssertionError(target_metric)
                    scores.append(score)

            T = torch.stack(scores, dim=1)
            if target_metric.endswith("_topk"):
                T = _scores_to_topk_membership(T, topk=topk)
            elif target_transform == "log":
                T = torch.log1p(T)
            elif target_transform == "sqrt":
                T = torch.sqrt(T.clamp_min(0))
            elif target_transform == "none":
                pass
            else:
                raise AssertionError(target_transform)
            target_scores[layer_idx].add_(T.sum(dim=0) * scale)

        return hook_ffn

    try:
        model.eval()
        for layer_idx in layer_ids:
            handles.append(model.blocks[layer_idx].ffn.register_forward_hook(make_hook(layer_idx)))

        label_schedule, g = _build_ar_label_schedule(
            nsamples=nsamples,
            num_classes=num_classes,
            calib_seed=calib_seed,
            mode=label_sampling,
        )
        n_batches = (nsamples + batch_size - 1) // batch_size
        for bi in tqdm(range(n_batches), desc="Router balance trajectory"):
            start = bi * batch_size
            cur_bs = min(batch_size, nsamples - bi * batch_size)
            if label_schedule is None:
                label_B = torch.randint(0, num_classes, (cur_bs,), generator=g)
            else:
                label_B = label_schedule[start:start + cur_bs]
            label_B = label_B.to(device=device, dtype=torch.long, non_blocking=True)
            _autoregressive_run_for_router_stats(
                var_model=model,
                B=cur_bs,
                label_B=label_B,
                g_seed=calib_seed + bi,
                cfg=cfg,
                top_k=top_k,
                top_p=top_p,
                stage_state=stage_state,
            )
    finally:
        for handle in handles:
            handle.remove()
        stage_state["si"] = -1
        stage_state["pn"] = 1

    updated: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
    stats: Dict[int, Dict[str, float]] = {}
    for layer_idx in layer_ids:
        W, b = layer_router_weights[layer_idx]
        target_probs = None
        if target_metric != "uniform":
            raw_target = target_scores[layer_idx].detach().float().cpu().clamp_min(0)
            target_total = float(raw_target.sum().item())
            if target_total <= 0:
                raise RuntimeError(f"Layer {layer_idx}: empty contribution-aware route target")
            target_probs = raw_target / target_total
            mix = float(target_mix_uniform)
            if mix > 0:
                uniform = torch.full_like(target_probs, 1.0 / float(target_probs.numel()))
                target_probs = mix * uniform + (1.0 - mix) * target_probs
                target_probs = target_probs / target_probs.sum().clamp_min(1e-12)
        new_bias, layer_stats = _router_balance_update_from_counts(
            counts=counts[layer_idx],
            bias=b,
            strength=strength,
            max_abs_bias=max_abs_bias,
            delta_linf_cap=delta_linf_cap,
            target_probs=target_probs,
        )
        updated[layer_idx] = (W, new_bias)
        layer_stats["target_metric"] = target_metric
        layer_stats["target_transform"] = target_transform if target_metric != "uniform" else None
        layer_stats["target_mix_uniform"] = float(target_mix_uniform) if target_metric != "uniform" else None
        stats[layer_idx] = layer_stats
        print(
            f"  [Router balance trajectory] Layer {layer_idx}: "
            f"min={layer_stats['min_freq']:.4f}, max={layer_stats['max_freq']:.4f}, "
            f"cv={layer_stats['cv']:.3f}, target_min={layer_stats['target_min_freq']:.4f}, "
            f"target_max={layer_stats['target_max_freq']:.4f}, bias_linf={layer_stats['bias_linf']:.3f}, "
            f"stage_weight={stage_weight}, target_metric={target_metric}, "
            f"label_sampling={label_sampling}"
        )
    return updated, stats


@torch.no_grad()
def compute_autoregressive_hidden_contribution_profiles(
    model,
    layer_indices: List[int],
    device: torch.device,
    nsamples: int,
    batch_size: int,
    calib_seed: int,
    num_classes: int,
    cfg: float = 4.0,
    top_k: int = 900,
    top_p: float = 0.96,
    max_tokens_per_call: int = 8192,
    transform: str = "log",
    position_bins: int = 1,
) -> Dict[int, torch.Tensor]:
    profiles, _scores = compute_autoregressive_hidden_contribution_profile_stats(
        model=model,
        layer_indices=layer_indices,
        device=device,
        nsamples=nsamples,
        batch_size=batch_size,
        calib_seed=calib_seed,
        num_classes=num_classes,
        cfg=cfg,
        top_k=top_k,
        top_p=top_p,
        max_tokens_per_call=max_tokens_per_call,
        transform=transform,
        position_bins=position_bins,
    )
    return profiles


@torch.no_grad()
def compute_autoregressive_hidden_contribution_profile_stats(
    model,
    layer_indices: List[int],
    device: torch.device,
    nsamples: int,
    batch_size: int,
    calib_seed: int,
    num_classes: int,
    cfg: float = 4.0,
    top_k: int = 900,
    top_p: float = 0.96,
    max_tokens_per_call: int = 8192,
    transform: str = "log",
    shared_score_mode: str = "sum",
    position_bins: int = 1,
) -> Tuple[Dict[int, torch.Tensor], Dict[int, torch.Tensor]]:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    if nsamples <= 0:
        raise ValueError("nsamples must be positive for trajectory profile collection")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive for trajectory profile collection")
    if transform not in {"log", "sqrt", "none"}:
        raise ValueError(f"Unknown trajectory profile transform: {transform}")
    if shared_score_mode not in {"sum", "max", "mean", "stable_contribution"}:
        raise ValueError(f"Unknown trajectory shared score mode: {shared_score_mode}")
    if position_bins <= 0:
        raise ValueError("position_bins must be positive")

    layer_ids = sorted(layer_indices)
    num_stages = len(model.patch_nums)
    num_bins = int(position_bins)
    stage_state = {"si": -1, "pn": 1}
    tokens_per_call = max_tokens_per_call if max_tokens_per_call and max_tokens_per_call > 0 else 0
    stats: Dict[int, Dict[str, Any]] = {}
    handles = []

    def make_hook(layer_idx: int):
        ffn = model.blocks[layer_idx].ffn
        fc1 = ffn.fc1
        fc2 = ffn.fc2
        C = fc1.in_features
        H = fc1.out_features
        fc2_col_norm_sq = fc2.weight.detach().float().to(device).pow(2).sum(dim=0)
        stats[layer_idx] = {
            "energy": torch.zeros((num_stages, num_bins, H), device=device, dtype=torch.float32),
            "tokens": torch.zeros((num_stages, num_bins), device=device, dtype=torch.float32),
        }

        def hook_ffn(module, inp, _out):
            if len(inp) == 0:
                return
            si = int(stage_state.get("si", -1))
            if si < 0 or si >= num_stages:
                return

            x_raw = inp[0].detach()
            if x_raw.dim() == 3:
                _, L, _ = x_raw.shape
                flat_pos = torch.arange(L, device=x_raw.device).view(1, L).expand(x_raw.shape[0], L).reshape(-1)
                x = x_raw.reshape(-1, C).float()
            else:
                x = x_raw.reshape(-1, C).float()
                L = max(1, int(stage_state.get("pn", 1)) ** 2)
                flat_pos = torch.arange(x.shape[0], device=x.device) % L
            if tokens_per_call > 0 and x.shape[0] > tokens_per_call:
                idx = torch.randperm(x.shape[0], device=x.device)[:tokens_per_call]
                x = x.index_select(0, idx)
                flat_pos = flat_pos.index_select(0, idx)

            h = F.linear(
                x,
                module.fc1.weight.detach().float(),
                module.fc1.bias.detach().float() if module.fc1.bias is not None else None,
            )
            h = module.act(h)
            token_energy = h.float().pow(2) * fc2_col_norm_sq.view(1, -1)
            if num_bins == 1:
                stats[layer_idx]["energy"][si, 0].add_(token_energy.sum(dim=0))
                stats[layer_idx]["tokens"][si, 0].add_(float(x.shape[0]))
            else:
                bin_idx = torch.div(flat_pos.clamp_min(0) * num_bins, max(1, int(L)), rounding_mode="floor")
                bin_idx = bin_idx.clamp_(0, num_bins - 1).long()
                stats[layer_idx]["energy"][si].index_add_(0, bin_idx, token_energy)
                stats[layer_idx]["tokens"][si].index_add_(
                    0,
                    bin_idx,
                    torch.ones_like(bin_idx, device=device, dtype=torch.float32),
                )

        return hook_ffn

    try:
        model.eval()
        for layer_idx in layer_ids:
            handles.append(model.blocks[layer_idx].ffn.register_forward_hook(make_hook(layer_idx)))

        g = torch.Generator(device="cpu")
        g.manual_seed(calib_seed)
        n_batches = (nsamples + batch_size - 1) // batch_size
        for bi in tqdm(range(n_batches), desc="Trajectory contribution profiles"):
            cur_bs = min(batch_size, nsamples - bi * batch_size)
            label_B = torch.randint(0, num_classes, (cur_bs,), generator=g).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
            _autoregressive_run_for_router_stats(
                var_model=model,
                B=cur_bs,
                label_B=label_B,
                g_seed=calib_seed + bi,
                cfg=cfg,
                top_k=top_k,
                top_p=top_p,
                stage_state=stage_state,
            )
    finally:
        for handle in handles:
            handle.remove()
        stage_state["si"] = -1
        stage_state["pn"] = 1

    profiles: Dict[int, torch.Tensor] = {}
    shared_scores: Dict[int, torch.Tensor] = {}
    for layer_idx in layer_ids:
        token_count = stats[layer_idx]["tokens"]
        if int((token_count > 0).sum().item()) == 0:
            raise RuntimeError(f"No trajectory profile tokens collected for layer {layer_idx}")
        profile = stats[layer_idx]["energy"] / token_count.clamp_min(1.0).unsqueeze(2)
        if transform == "log":
            profile = torch.log1p(profile)
        elif transform == "sqrt":
            profile = torch.sqrt(profile.clamp_min(0))
        elif transform == "none":
            pass
        profile_t = profile.reshape(num_stages * num_bins, -1).t().contiguous()
        token_count_flat = token_count.reshape(-1)
        if shared_score_mode == "sum":
            shared_score = profile_t.sum(dim=1)
        elif shared_score_mode == "max":
            shared_score = profile_t.max(dim=1).values
        elif shared_score_mode == "mean":
            stage_mask = token_count_flat > 0
            shared_score = profile_t[:, stage_mask].mean(dim=1)
        elif shared_score_mode == "stable_contribution":
            stage_mask = token_count_flat > 0
            profile_active = profile_t[:, stage_mask]
            mean = profile_active.mean(dim=1)
            std = profile_active.std(dim=1, unbiased=False)
            shared_score = mean / (std + mean.abs() + 1e-12)
        else:
            raise AssertionError(shared_score_mode)

        profile_t = profile_t / profile_t.norm(dim=1, keepdim=True).clamp_min(1e-12)
        profiles[layer_idx] = profile_t.detach().cpu().float()
        shared_scores[layer_idx] = shared_score.detach().cpu().float()
        print(
            f"  [Trajectory profiles] Layer {layer_idx}: "
            f"active_stage_bins={int((token_count > 0).sum().item())}/{num_stages * num_bins}, "
            f"tokens={int(token_count.sum().item())}"
        )

    return profiles, shared_scores


def build_var_d2m_from_ffn(
    ffn: nn.Module,
    shared_indices: np.ndarray,
    expert_indices_list: List[np.ndarray],
    n_experts: int,
    topk: int,
    hard_mode: bool,
    device: torch.device,
    router_bias: bool = False,
) -> nn.Module:
\
\
\
\
\
\
\
\
\
\
\
       
    fc1 = ffn.fc1
    fc2 = ffn.fc2
    C = fc1.in_features
    H = fc1.out_features

    shared_hidden = len(shared_indices)
    remain = H - shared_hidden
    assert len(expert_indices_list) == n_experts
    assert remain % n_experts == 0
    expert_hidden = remain // n_experts

    drop_rate = 0.0
    if hasattr(ffn, "drop") and isinstance(ffn.drop, nn.Dropout):
        drop_rate = float(ffn.drop.p)

    moe = VARD2MFFN(
        in_features=C,
        shared_hidden=shared_hidden,
        expert_hidden=expert_hidden,
        n_experts=n_experts,
        topk=topk,
        drop=drop_rate,
        hard_mode=hard_mode,
        router_bias=router_bias,
    ).to(device)

                           
    w1 = fc1.weight.data.to(device)
    b1 = fc1.bias.data.to(device) if fc1.bias is not None else None
    w2 = fc2.weight.data.to(device)
    b2 = fc2.bias.data.to(device) if fc2.bias is not None else None

    sidx = torch.tensor(shared_indices, device=device, dtype=torch.long)
    with torch.no_grad():
        moe.shared.fc1.weight.data.copy_(w1.index_select(0, sidx))
        if b1 is not None:
            moe.shared.fc1.bias.data.copy_(b1.index_select(0, sidx))
        moe.shared.fc2.weight.data.copy_(w2.index_select(1, sidx))

                            
    for i, idx_np in enumerate(expert_indices_list):
        eidx = torch.tensor(idx_np, device=device, dtype=torch.long)
        with torch.no_grad():
            moe.experts[i].fc1.weight.data.copy_(w1.index_select(0, eidx))
            if b1 is not None:
                moe.experts[i].fc1.bias.data.copy_(b1.index_select(0, eidx))
            moe.experts[i].fc2.weight.data.copy_(w2.index_select(1, eidx))

                           
    if b2 is not None:
        with torch.no_grad():
                                                                              
            moe.out_bias.copy_(b2.detach().float().to(device=moe.out_bias.device))

    return moe

def compute_hidden_activation_stability(
    model,
    layer_idx: int,
    calib_pairs: List[Tuple[torch.Tensor, int]],
    vae,
    device: torch.device,
) -> torch.Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    block = model.blocks[layer_idx]
    ffn = block.ffn
    H = ffn.fc1.out_features
    
    activations_list = []
    
                                        
    def activation_hook(module, inp, out):
                                   
                                            
                                  
        x_ffn_inp = inp[0].detach()             
        B, L, C = x_ffn_inp.shape
        x_flat = x_ffn_inp.reshape(B * L, C)            
        
                           
        preact = F.linear(x_flat, module.fc1.weight, module.fc1.bias)            
                                    
        h = module.act(preact)            
        activations_list.append(h.cpu())
    
             
    handle = ffn.register_forward_hook(activation_hook)
    
    try:
        model.eval()
        with torch.no_grad():
                                                            
            for label_B, seed_or_payload in calib_pairs:
                x_BLCv_wo_first_l, _gt_BL, seed = _unpack_token_payload(seed_or_payload)
                B = label_B.shape[0]
                label_B = label_B.to(device=device, non_blocking=True)
                torch.manual_seed(seed)

                if x_BLCv_wo_first_l is None:
                    H_img, W_img = 256, 256
                    dummy_images = torch.rand(B, 3, H_img, W_img, device=device)
                    gt_idx_Bl = vae.img_to_idxBl(dummy_images)
                    x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
                else:
                    x_BLCv_wo_first_l = x_BLCv_wo_first_l.to(device=device, non_blocking=True)
                
                                               
                original_prog_si = model.prog_si
                model.prog_si = -1                  
                _ = model(label_B, x_BLCv_wo_first_l)           
                model.prog_si = original_prog_si
    finally:
        handle.remove()
    
    if len(activations_list) == 0:
        raise RuntimeError(f"No activations collected for layer {layer_idx}")
    
             
    activations = torch.cat(activations_list, dim=0)                
    
                                          
                                
                                               
    abs_activations = activations.abs()                
    mean_abs = abs_activations.mean(dim=0)       
    std_abs = abs_activations.std(dim=0)       
    eps = 1e-6
    stability = mean_abs / (std_abs + eps)       
    
    return stability.float()


@torch.no_grad()
def compute_hidden_contribution_energy(
    model,
    layer_idx: int,
    calib_pairs: List[Tuple[torch.Tensor, int]],
    vae,
    device: torch.device,
    max_tokens_per_call: int = 8192,
    transform: str = "log",
) -> torch.Tensor:
\
\
\
\
\
\
\
\
\
\
\
\
       
    block = model.blocks[layer_idx]
    ffn = block.ffn
    fc1 = ffn.fc1
    fc2 = ffn.fc2
    H = fc1.out_features
    C = fc1.in_features

    energy_sum = torch.zeros(H, device=device, dtype=torch.float32)
    token_count = 0
    tokens_per_call = max_tokens_per_call if max_tokens_per_call and max_tokens_per_call > 0 else 0
    fc2_col_norm_sq = fc2.weight.detach().float().to(device).pow(2).sum(dim=0)

    def contribution_hook(module, inp, _out):
        nonlocal token_count
        if len(inp) == 0:
            return
        x = inp[0].detach().reshape(-1, C).float()
        if tokens_per_call > 0 and x.shape[0] > tokens_per_call:
            idx = torch.randperm(x.shape[0], device=x.device)[:tokens_per_call]
            x = x.index_select(0, idx)

        h = F.linear(
            x,
            module.fc1.weight.detach().float(),
            module.fc1.bias.detach().float() if module.fc1.bias is not None else None,
        )
        h = module.act(h)
        energy_sum.add_(h.float().pow(2).sum(dim=0) * fc2_col_norm_sq)
        token_count += int(x.shape[0])

    handle = ffn.register_forward_hook(contribution_hook)
    original_prog_si = model.prog_si

    try:
        model.eval()
        with torch.no_grad():
            for label_B, seed_or_payload in calib_pairs:
                x_BLCv_wo_first_l, _gt_BL, seed = _unpack_token_payload(seed_or_payload)
                B = label_B.shape[0]
                label_B = label_B.to(device=device, non_blocking=True)
                torch.manual_seed(seed)

                if x_BLCv_wo_first_l is None:
                    dummy_images = torch.rand(B, 3, 256, 256, device=device)
                    gt_idx_Bl = vae.img_to_idxBl(dummy_images)
                    x_BLCv_wo_first_l = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
                else:
                    x_BLCv_wo_first_l = x_BLCv_wo_first_l.to(device=device, non_blocking=True)

                model.prog_si = -1
                _ = model(label_B, x_BLCv_wo_first_l)
    finally:
        model.prog_si = original_prog_si
        handle.remove()

    if token_count <= 0:
        raise RuntimeError(f"No contribution tokens collected for layer {layer_idx}")

    score = energy_sum / float(token_count)
    if transform == "log":
        score = torch.log1p(score)
    elif transform == "sqrt":
        score = torch.sqrt(score.clamp_min(0))
    elif transform == "none":
        pass
    else:
        raise ValueError(f"Unknown contribution transform: {transform}")

    return score.detach().cpu().float()


def split_hidden_by_importance_two_stage(
    importance: torch.Tensor,
    stability: torch.Tensor,
    shared_hidden: int,
    n_experts: int,
    candidate_multiplier: float = 2.0,
    shared_selection_mode: str = "second_score",
    shared_importance_weight: float = 0.0,
    expert_assignment: str = "contiguous",
    ffn_fc1_weight: Optional[torch.Tensor] = None,
    assignment_features: Optional[torch.Tensor] = None,
    kmeans_iters: int = 8,
    kmeans_restarts: int = 1,
) -> Tuple[np.ndarray, List[np.ndarray]]:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    H = importance.shape[0]
    assert stability.shape[0] == H, f"Importance and stability shape mismatch: {importance.shape} vs {stability.shape}"
    if shared_selection_mode not in {"second_score", "rank_fusion"}:
        raise ValueError(f"Unknown shared_selection_mode: {shared_selection_mode}")
    if not (0.0 <= float(shared_importance_weight) <= 1.0):
        raise ValueError("shared_importance_weight must be in [0, 1]")
    
                                                 
    imp_np = importance.numpy()
    order_t = np.argsort(-imp_np)                                   
    M = int(round(candidate_multiplier * shared_hidden))
    M = min(M, H)           
    candidate_pool = order_t[:M]       
    
                                                             
    stab_np = stability.numpy()
    candidate_stability = stab_np[candidate_pool]       
    if shared_selection_mode == "second_score":
        order_s = np.argsort(-candidate_stability)                              
    elif shared_selection_mode == "rank_fusion":
        def _rank01_desc(values: np.ndarray) -> np.ndarray:
            values = np.nan_to_num(values.astype(np.float64), nan=-np.inf, posinf=np.inf, neginf=-np.inf)
            order = np.argsort(-values, kind="mergesort")
            ranks = np.empty(values.shape[0], dtype=np.float64)
            ranks[order] = np.arange(values.shape[0], dtype=np.float64)
            if values.shape[0] <= 1:
                return np.ones(values.shape[0], dtype=np.float64)
            return 1.0 - ranks / float(values.shape[0] - 1)

        imp_rank = _rank01_desc(imp_np[candidate_pool])
        second_rank = _rank01_desc(candidate_stability)
        w_imp = float(shared_importance_weight)
        fused_score = w_imp * imp_rank + (1.0 - w_imp) * second_rank
        order_s = np.argsort(-fused_score, kind="mergesort")
    else:
        raise AssertionError(shared_selection_mode)
    shared_idx_in_pool = order_s[:shared_hidden]
    shared_idx = candidate_pool[shared_idx_in_pool]                   
    
                                                            
    rest_mask = np.ones(H, dtype=bool)
    rest_mask[shared_idx] = False
    rest = np.where(rest_mask)[0]              
    
                                 
    rest_imp = imp_np[rest]
    rest_order = np.argsort(-rest_imp)              
    
    rest_ordered = rest[rest_order]
    expert_idx_list = _split_rest_to_experts(
        rest_ordered=rest_ordered,
        n_experts=n_experts,
        expert_assignment=expert_assignment,
        ffn_fc1_weight=ffn_fc1_weight,
        assignment_features=assignment_features,
        importance=importance,
        kmeans_iters=kmeans_iters,
        kmeans_restarts=kmeans_restarts,
    )
    
    return shared_idx.astype(np.int64), [x.astype(np.int64) for x in expert_idx_list]


def build_hybrid_assignment_features(
    ffn_fc1_weight: torch.Tensor,
    ffn_fc2_weight: Optional[torch.Tensor],
    trajectory_profile: torch.Tensor,
    fc1_weight: float = 0.5,
    fc2_weight: float = 0.0,
    profile_weight: float = 1.0,
    stage_onehot_weight: float = 0.0,
) -> torch.Tensor:
\
\
\
\
\
\
\
\
\
\
       
    if fc1_weight < 0 or fc2_weight < 0 or profile_weight < 0 or stage_onehot_weight < 0:
        raise ValueError("feature weights must be non-negative")
    if fc1_weight == 0 and fc2_weight == 0 and profile_weight == 0 and stage_onehot_weight == 0:
        raise ValueError("at least one feature weight must be positive")

    W = ffn_fc1_weight.detach().float().cpu()
    P = trajectory_profile.detach().float().cpu()
    if W.shape[0] != P.shape[0]:
        raise ValueError(f"Feature row mismatch: fc1={W.shape}, profile={P.shape}")
    P = torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

    blocks = []
    if fc1_weight > 0:
        blocks.append(F.normalize(W, p=2, dim=1) * float(fc1_weight))
    if fc2_weight > 0:
        if ffn_fc2_weight is None:
            raise ValueError("ffn_fc2_weight is required when fc2_weight > 0")
        W2 = ffn_fc2_weight.detach().float().cpu().t().contiguous()
        if W2.shape[0] != P.shape[0]:
            raise ValueError(f"Feature row mismatch: fc2={W2.shape}, profile={P.shape}")
        blocks.append(F.normalize(W2, p=2, dim=1) * float(fc2_weight))
    if profile_weight > 0:
        blocks.append(F.normalize(P, p=2, dim=1) * float(profile_weight))
    if stage_onehot_weight > 0:
        dominant_stage = P.argmax(dim=1)
        onehot = F.one_hot(dominant_stage, num_classes=P.shape[1]).float()
        blocks.append(onehot * float(stage_onehot_weight))
    return torch.cat(blocks, dim=1).float()
