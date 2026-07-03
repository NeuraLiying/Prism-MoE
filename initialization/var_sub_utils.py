                  
\
\
\
\
\
\
\
\
\
   

from typing import Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

import os
import sys
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).resolve().parents[0]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.path_utils import add_var_root

from var_sub_model import VARD2MFFN


class _StepState:
    step: int = -1


STEP_STATE = _StepState()


def load_var_weights_into_model(var_model: nn.Module, ckpt_path: str) -> None:
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "trainer" in sd and "var_wo_ddp" in sd["trainer"]:
        sd = sd["trainer"]["var_wo_ddp"]
    elif isinstance(sd, dict) and "var_wo_ddp" in sd:
        sd = sd["var_wo_ddp"]
    elif isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    var_model.load_state_dict(sd, strict=False)


def ensure_all_layers_are_moe(var_model: nn.Module, moe_cls=VARD2MFFN) -> None:
    for li, blk in enumerate(var_model.blocks):
        if not isinstance(blk.ffn, moe_cls):
            raise RuntimeError(
                f"Layer {li} is not MoE FFN ({moe_cls.__name__}). Found: {type(blk.ffn)}"
            )


def autoregressive_run_with_step_tracking(
    var_model,
    B: int,
    label_B: torch.LongTensor,
    g_seed: int,
    cfg: float,
    top_k: int = 900,
    top_p: float = 0.96,
    more_smooth: bool = False,
    forced_token_indices: Optional[List[torch.Tensor]] = None,
):
\
\
\
\
\
\
\
       
    add_var_root()
    from models.helpers import gumbel_softmax_with_rng, sample_with_top_k_top_p_

    var_model.rng.manual_seed(g_seed)
    rng = var_model.rng

    sos = cond_BD = var_model.class_emb(
        torch.cat((label_B, torch.full_like(label_B, fill_value=var_model.num_classes)), dim=0)
    )
    lvl_pos = var_model.lvl_embed(var_model.lvl_1L) + var_model.pos_1LC

    next_token_map = (
        sos.unsqueeze(1).expand(2 * B, var_model.first_l, -1)
        + var_model.pos_start.expand(2 * B, var_model.first_l, -1)
        + lvl_pos[:, :var_model.first_l]
    )

    cur_L = 0
    f_hat = sos.new_zeros(B, var_model.Cvae, var_model.patch_nums[-1], var_model.patch_nums[-1])

    for blk in var_model.blocks:
        blk.attn.kv_caching(True)

    for si, pn in enumerate(var_model.patch_nums):
        STEP_STATE.step = int(si)

        ratio = si / var_model.num_stages_minus_1
        cur_L += pn * pn

        cond_BD_or_gss = var_model.shared_ada_lin(cond_BD)
        x = next_token_map

        for blk in var_model.blocks:
            x = blk(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)

        logits_BlV = var_model.get_logits(x, cond_BD)
        t = cfg * ratio
        logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]

                                   
        if torch.any(torch.isnan(logits_BlV)) or torch.any(torch.isinf(logits_BlV)):
                                 
            logits_BlV = torch.where(
                torch.isnan(logits_BlV) | torch.isinf(logits_BlV),
                torch.zeros_like(logits_BlV),
                logits_BlV
            )
                            
            logits_BlV = torch.clamp(logits_BlV, min=-1e6, max=1e6)

                                              
        if forced_token_indices is not None and si < len(forced_token_indices):
            idx_Bl = forced_token_indices[si]          
        else:
                                   
            if torch.any(torch.isnan(logits_BlV)) or torch.any(torch.isinf(logits_BlV)):
                import warnings
                warnings.warn(
                    f"Stage {si}: logits still contain inf/nan after cleaning. "
                    f"Using uniform sampling as fallback."
                )
                                
                V = logits_BlV.shape[-1]
                idx_Bl = torch.randint(0, V, (B, logits_BlV.shape[1]), device=logits_BlV.device)
            else:
                idx_Bl = sample_with_top_k_top_p_(
                    logits_BlV, rng=rng, top_k=top_k, top_p=top_p, num_samples=1
                )[:, :, 0]

        if not more_smooth:
            h_BChw = var_model.vae_quant_proxy[0].embedding(idx_Bl)
        else:
            gum_t = max(0.27 * (1 - ratio * 0.95), 0.005)
            h_BChw = (
                gumbel_softmax_with_rng(
                    logits_BlV.mul(1 + ratio), tau=gum_t, hard=False, dim=-1, rng=rng
                )
                @ var_model.vae_quant_proxy[0].embedding.weight.unsqueeze(0)
            )

        h_BChw = h_BChw.transpose_(1, 2).reshape(B, var_model.Cvae, pn, pn)
        f_hat, next_token_map = var_model.vae_quant_proxy[0].get_next_autoregressive_input(
            si, len(var_model.patch_nums), f_hat, h_BChw
        )

        if si != var_model.num_stages_minus_1:
            next_token_map = next_token_map.view(B, var_model.Cvae, -1).transpose(1, 2)
            next_token_map = (
                var_model.word_embed(next_token_map)
                + lvl_pos[:, cur_L:cur_L + var_model.patch_nums[si + 1] ** 2]
            )
            next_token_map = next_token_map.repeat(2, 1, 1)

    for blk in var_model.blocks:
        blk.attn.kv_caching(False)

    STEP_STATE.step = -1


@torch.no_grad()
def _run_student_calibration_passes(
    student,
    calib_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    cfg: float,
    desc: str,
    use_student_trajectory: bool,
    trajectory_seed: int = 42,
    trajectory_top_k: int = 900,
    trajectory_top_p: float = 0.96,
) -> None:
\
\
\
\
\
\
\
       
    for bi, (label_B, tokens_BLCv) in enumerate(tqdm(
        calib_pairs,
        desc=desc,
        leave=True,
        ncols=100,
    )):
        if use_student_trajectory:
            label_B = label_B.to(device=student.lvl_1L.device, dtype=torch.long, non_blocking=True)
            autoregressive_run_with_step_tracking(
                var_model=student,
                B=int(label_B.shape[0]),
                label_B=label_B,
                g_seed=int(trajectory_seed) + bi,
                cfg=cfg,
                top_k=trajectory_top_k,
                top_p=trajectory_top_p,
                more_smooth=False,
            )
        else:
            student(label_B, tokens_BLCv)


def solve_delta_ridge(
    A: torch.Tensor, 
    B: torch.Tensor, 
    W_old: torch.Tensor,
    ridge_lambda: float,
    max_delta_norm: float = 0.1,
    layer_name: str = "unknown",
    use_adaptive_lambda: bool = True,
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
       
    D = A.shape[0]
    C = W_old.shape[0]
    device = A.device
    dtype = A.dtype
    
                                         
                     
                                                  
                                                  
    W_old_T = W_old.t()          
    G = B - A @ W_old_T          
    
                             
                                    
                         
    if use_adaptive_lambda:
        diag_mean = A.diag().mean().item()
        adaptive_lambda = float(ridge_lambda) * diag_mean
    else:
        adaptive_lambda = float(ridge_lambda)
    
                 
                                     
    A_reg = A + adaptive_lambda * torch.eye(D, device=device, dtype=dtype)
    
    try:
        L = torch.linalg.cholesky(A_reg, upper=False)
        y = torch.linalg.solve_triangular(L, G, upper=False, unitriangular=False)
        delta_T = torch.linalg.solve_triangular(L.t(), y, upper=True, unitriangular=False)
    except RuntimeError:
        try:
            delta_T = torch.linalg.solve(A_reg, G)
        except RuntimeError:
            delta_T = torch.linalg.lstsq(A_reg, G).solution
    
    delta = delta_T.t()          
    
                                     
                                                  
                                  
    norm_delta = delta.norm().item()
    norm_old = W_old.norm().item()
    
    if norm_old > 1e-6:
        ratio = norm_delta / norm_old
        if ratio > max_delta_norm:
            scale = max_delta_norm / ratio
            delta = delta * scale
            print(f"  [Safety] {layer_name} delta clipped! Ratio {ratio:.4f} -> {max_delta_norm:.4f}. "
                  f"Norm: {norm_delta:.2e} -> {delta.norm().item():.2e}")
        else:
            print(f"  [Info] {layer_name} delta ratio: {ratio:.4f} (Safe, threshold={max_delta_norm:.4f})")
    else:
                                   
        if norm_delta > max_delta_norm:
            delta = delta * (max_delta_norm / norm_delta)
            print(f"  [Safety] {layer_name} delta clipped! Norm {norm_delta:.2e} -> {max_delta_norm:.2e} "
                  f"(W_old norm too small: {norm_old:.2e})")
    
    W_new = W_old + delta
    return W_new.contiguous()


def solve_residual_delta_ridge(
    A: torch.Tensor,
    B: torch.Tensor,
    reference_W: torch.Tensor,
    ridge_lambda: float,
    max_delta_norm: float = 0.1,
    layer_name: str = "unknown",
    use_adaptive_lambda: bool = True,
) -> torch.Tensor:
\
\
\
\
\
\
\
\
       
    D = A.shape[0]
    device = A.device
    dtype = A.dtype

    if use_adaptive_lambda:
        diag_mean = A.diag().mean().item()
        adaptive_lambda = float(ridge_lambda) * diag_mean
    else:
        adaptive_lambda = float(ridge_lambda)

    A_reg = A + adaptive_lambda * torch.eye(D, device=device, dtype=dtype)

    try:
        L = torch.linalg.cholesky(A_reg, upper=False)
        y = torch.linalg.solve_triangular(L, B, upper=False, unitriangular=False)
        delta_T = torch.linalg.solve_triangular(L.t(), y, upper=True, unitriangular=False)
    except RuntimeError:
        try:
            delta_T = torch.linalg.solve(A_reg, B)
        except RuntimeError:
            delta_T = torch.linalg.lstsq(A_reg, B).solution

    delta = delta_T.t().contiguous()

    norm_delta = delta.norm().item()
    norm_ref = reference_W.norm().item()
    if norm_ref > 1e-6:
        ratio = norm_delta / norm_ref
        if ratio > max_delta_norm:
            scale = max_delta_norm / ratio
            delta = delta * scale
            print(
                f"  [Safety] {layer_name} residual delta clipped! "
                f"Ratio {ratio:.4f} -> {max_delta_norm:.4f}. "
                f"Norm: {norm_delta:.2e} -> {delta.norm().item():.2e}"
            )
        else:
            print(f"  [Info] {layer_name} residual delta ratio: {ratio:.4f} (threshold={max_delta_norm:.4f})")
    else:
        if norm_delta > max_delta_norm:
            delta = delta * (max_delta_norm / norm_delta)
            print(
                f"  [Safety] {layer_name} residual delta clipped! "
                f"Norm {norm_delta:.2e} -> {max_delta_norm:.2e} "
                f"(reference norm too small: {norm_ref:.2e})"
            )

    return delta


def solve_ridge_from_stats(
    A: torch.Tensor, 
    B: torch.Tensor, 
    ridge_lambda: float,
    use_adaptive_lambda: bool = True,
    use_diag_precondition: bool = True,
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
       
    D = A.shape[0]
    
                             
    if use_adaptive_lambda:
                                 
        diag_mean = A.diag().mean().item()
        adaptive_lambda = float(ridge_lambda) * diag_mean
    else:
        adaptive_lambda = float(ridge_lambda)
    
                        
    if use_diag_precondition:
                           
        diag_sqrt = A.diag().sqrt().clamp_min(1e-8)       
        diag_sqrt_inv = 1.0 / diag_sqrt       
        
                                       
                                 
        D_mat = torch.diag(diag_sqrt_inv)          
        A_precond = D_mat @ A @ D_mat          
        B_precond = D_mat @ B          
        
                   
        A_reg = A_precond + adaptive_lambda * torch.eye(D, device=A.device, dtype=A.dtype)
    else:
        A_reg = A + adaptive_lambda * torch.eye(D, device=A.device, dtype=A.dtype)
        B_precond = B
        diag_sqrt_inv = torch.ones(D, device=A.device, dtype=A.dtype)
    
                      
    try:
        L = torch.linalg.cholesky(A_reg, upper=False)
        y = torch.linalg.solve_triangular(L, B_precond, upper=False, unitriangular=False)
        W_T_precond = torch.linalg.solve_triangular(L.t(), y, upper=True, unitriangular=False)
    except RuntimeError:
        try:
            W_T_precond = torch.linalg.solve(A_reg, B_precond)
        except RuntimeError:
            W_T_precond = torch.linalg.lstsq(A_reg, B_precond).solution
    
                                
    if use_diag_precondition:
        W_T = torch.diag(diag_sqrt_inv) @ W_T_precond
    else:
        W_T = W_T_precond
    
    return W_T.t().contiguous()


def _subsample_tokens_pair(
    x2: torch.Tensor,
    y2: torch.Tensor,
    max_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
\
\
\
\
       
    if max_tokens is None or max_tokens <= 0:
        return x2, y2
    N = x2.shape[0]
    if N <= max_tokens:
        return x2, y2
    idx = torch.randperm(N, device=x2.device)[:max_tokens]
    return x2.index_select(0, idx), y2.index_select(0, idx)


@torch.no_grad()
def verify_out_bias_consistency(
    teacher: nn.Module,
    student: nn.Module,
    layer_idx: int,
    tolerance: float = 1e-5,
    verbose: bool = True,
) -> bool:
\
\
\
\
\
       
    ffn_t = teacher.blocks[layer_idx].ffn
    ffn_s = student.blocks[layer_idx].ffn
    
    if not isinstance(ffn_s, VARD2MFFN):
        return False
    
                            
    teacher_bias = None
    if hasattr(ffn_t, 'fc2') and hasattr(ffn_t.fc2, 'bias') and ffn_t.fc2.bias is not None:
        teacher_bias = ffn_t.fc2.bias.detach().float()
    
                           
    student_bias = ffn_s.out_bias.detach().float()
    
                                   
    student_bias_norm = student_bias.norm().item()
    if student_bias_norm < tolerance:
        if verbose:
            print(f"  ⚠️  Layer {layer_idx}: student.out_bias is all zeros (norm={student_bias_norm:.2e})")
            print(f"     This may indicate out_bias was not loaded correctly from checkpoint.")
            if teacher_bias is not None:
                teacher_norm = teacher_bias.norm().item()
                print(f"     Teacher has bias with norm={teacher_norm:.6f}, but student.out_bias is zero!")
        return False
    
                              
    if teacher_bias is not None:
        diff = (student_bias - teacher_bias).abs().max().item()
        if diff > tolerance:
            if verbose:
                print(f"  ⚠️  Layer {layer_idx}: out_bias mismatch!")
                print(f"     Teacher bias norm: {teacher_bias.norm().item():.6f}")
                print(f"     Student bias norm: {student_bias_norm:.6f}")
                print(f"     Max difference: {diff:.6f}")
            return False
        elif verbose:
            print(f"  ✓ Layer {layer_idx}: out_bias matches teacher (norm={student_bias_norm:.6f}, diff={diff:.2e})")
    else:
                                                  
        if student_bias_norm > tolerance:
            if verbose:
                print(f"  ⚠️  Layer {layer_idx}: teacher has no bias, but student.out_bias is non-zero (norm={student_bias_norm:.6f})")
            return False
        elif verbose:
            print(f"  ✓ Layer {layer_idx}: teacher has no bias, student.out_bias is zero (as expected)")
    
    return True


def _compute_experts_output(x2: torch.Tensor, ffn_s: VARD2MFFN, dtype: torch.dtype) -> torch.Tensor:
\
\
\
\
\
\
       
    C = ffn_s.in_features
    E = ffn_s.n_experts
    logits = ffn_s.gate(x2)

    if ffn_s.hard_mode:
        indices = torch.topk(logits.float(), k=ffn_s.topk, dim=-1).indices          
        y_exp = torch.zeros((x2.shape[0], C), device=x2.device, dtype=dtype)

        for e in range(E):
                                                   
            mask_any = (indices == e).any(dim=1)       
            if not mask_any.any():
                continue
            token_idx = mask_any.nonzero(as_tuple=True)[0]                          
            he = ffn_s.experts[e].hidden(x2[token_idx]).to(dtype=dtype)
            y_e = F.linear(he, ffn_s.experts[e].fc2.weight)
            y_exp[token_idx] += y_e
        return y_exp

    probs = torch.softmax(logits, dim=-1)
    topk_p, topk_i = torch.topk(probs, k=ffn_s.topk, dim=-1)
    topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    p = torch.zeros_like(probs)
    p.scatter_(1, topk_i, topk_p)
    p = p.to(dtype=dtype)

    y_exp = torch.zeros((x2.shape[0], C), device=x2.device, dtype=dtype)
    for e in range(E):
        pe = p[:, e:e+1]
        if torch.all(pe == 0):
            continue
        he = ffn_s.experts[e].hidden(x2).to(dtype=dtype)
        y_e = F.linear(he, ffn_s.experts[e].fc2.weight)
        y_exp += y_e * pe
    return y_exp


@torch.no_grad()
def collect_dense_ffn_io_stats_for_shared(
    teacher,
    student,
    layer_idx: int,
    calib_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    cfg: float,
    dtype: torch.dtype = torch.float32,
    max_tokens_per_call: int = 0,
    use_student_trajectory: bool = True,
    trajectory_seed: int = 42,
    trajectory_top_k: int = 900,
    trajectory_top_p: float = 0.96,
) -> Dict[str, torch.Tensor]:
\
\
       
    ffn_t = teacher.blocks[layer_idx].ffn
    ffn_s: VARD2MFFN = student.blocks[layer_idx].ffn

    C = ffn_s.in_features
    S = ffn_s.shared_hidden
    E = ffn_s.n_experts
    device = ffn_s.out_bias.device

    A = torch.zeros((S, S), device=device, dtype=dtype)
    B = torch.zeros((S, C), device=device, dtype=dtype)

    out_bias = ffn_s.out_bias.detach().to(dtype=dtype)
    merge_device = ffn_s.shared.fc1.weight.device

    def _compute_teacher_ffn_output(x2: torch.Tensor) -> torch.Tensor:
\
\
\
           
        y_t = ffn_t(x2)
        return y_t.to(dtype=dtype)

    def hook_student_ffn(_module, inp, out):
        if len(inp) == 0:
            return

        x = inp[0].detach()
        x2 = x.reshape(-1, x.shape[-1])

        if max_tokens_per_call and max_tokens_per_call > 0:
            N = x2.shape[0]
            if N > max_tokens_per_call:
                idx = torch.randperm(N, device=x2.device)[:max_tokens_per_call]
                x2 = x2.index_select(0, idx)

        x2 = x2.to(dtype=dtype)

        y_teacher = _compute_teacher_ffn_output(x2)
        y_teacher = y_teacher - out_bias.view(1, -1)

        y_experts = _compute_experts_output(x2, ffn_s, dtype)

        target = y_teacher - y_experts

        h_shared = ffn_s.shared.hidden(x2).to(dtype=dtype)

        A.add_(h_shared.t().mm(h_shared))
        B.add_(h_shared.t().mm(target))

    handle_student = ffn_s.register_forward_hook(hook_student_ffn)

    try:
        teacher.eval()
        student.eval()
        _run_student_calibration_passes(
            student=student,
            calib_pairs=calib_pairs,
            cfg=cfg,
            desc=f"L{layer_idx} Shared fc2 stats",
            use_student_trajectory=use_student_trajectory,
            trajectory_seed=trajectory_seed,
            trajectory_top_k=trajectory_top_k,
            trajectory_top_p=trajectory_top_p,
        )
    finally:
        handle_student.remove()

    return {"A": A, "B": B}


@torch.no_grad()
def collect_dense_ffn_io_stats_for_experts(
    teacher,
    student,
    layer_idx: int,
    calib_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    cfg: float,
    dtype: torch.dtype = torch.float32,
    max_tokens_per_call: int = 0,
    use_student_trajectory: bool = True,
    trajectory_seed: int = 42,
    trajectory_top_k: int = 900,
    trajectory_top_p: float = 0.96,
) -> Dict[int, Dict[str, torch.Tensor]]:
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
       
    ffn_t = teacher.blocks[layer_idx].ffn
    ffn_s: VARD2MFFN = student.blocks[layer_idx].ffn

    C = ffn_s.in_features
    H = ffn_s.expert_hidden                 
    E = ffn_s.n_experts
    K = ffn_s.topk
    device = ffn_s.out_bias.device

                       
    expert_stats = {}
    for e in range(E):
        expert_stats[e] = {
            "A": torch.zeros((H, H), device=device, dtype=dtype),
            "B": torch.zeros((H, C), device=device, dtype=dtype),
            "token_count": 0,
        }

    out_bias = ffn_s.out_bias.detach().to(dtype=dtype)

    def _compute_teacher_ffn_output(x2: torch.Tensor) -> torch.Tensor:
                            
        y_t = ffn_t(x2)
        return y_t.to(dtype=dtype)

    def hook_student_ffn(_module, inp, out):
        if len(inp) == 0:
            return

        x = inp[0].detach()
        x2 = x.reshape(-1, x.shape[-1])

        if max_tokens_per_call and max_tokens_per_call > 0:
            N = x2.shape[0]
            if N > max_tokens_per_call:
                idx = torch.randperm(N, device=x2.device)[:max_tokens_per_call]
                x2 = x2.index_select(0, idx)

        x2 = x2.to(dtype=dtype)
        N = x2.shape[0]

                                                                                        
        y_teacher = _compute_teacher_ffn_output(x2)
        y_teacher = y_teacher - out_bias.view(1, -1)          

                                                       
        h_shared = ffn_s.shared.hidden(x2).to(dtype=dtype)          
        y_shared = F.linear(h_shared, ffn_s.shared.fc2.weight)          

                                                          
        y_experts = _compute_experts_output(x2, ffn_s, dtype)          

                  
        r = y_teacher - y_shared - y_experts          

                       
        logits = ffn_s.gate(x2)          

        if ffn_s.hard_mode:
                                  
            indices = torch.topk(logits.float(), k=K, dim=-1).indices          
            target_piece = r / float(K)                                 

                                              
            for e in range(E):
                                                       
                mask_any = (indices == e).any(dim=1)       
                if not mask_any.any():
                    continue

                token_idx = mask_any.nonzero(as_tuple=True)[0]          
                M = token_idx.shape[0]

                                      
                h_i = ffn_s.experts[e].hidden(x2[token_idx]).to(dtype=dtype)          

                       
                expert_stats[e]["A"].add_(h_i.t().mm(h_i))          
                expert_stats[e]["B"].add_(h_i.t().mm(target_piece[token_idx]))          
                expert_stats[e]["token_count"] += M

        else:
                                   
            probs = torch.softmax(logits, dim=-1)          
            topk_p, topk_i = torch.topk(probs, k=K, dim=-1)                  
            topk_p = topk_p / topk_p.sum(dim=-1, keepdim=True).clamp_min(1e-9)               

                               
            p = torch.zeros((N, E), device=x2.device, dtype=dtype)          
            p.scatter_(1, topk_i, topk_p)          

                          
            for e in range(E):
                pe = p[:, e]       
                mask_active = pe > 1e-8       
                if not mask_active.any():
                    continue

                token_idx = mask_active.nonzero(as_tuple=True)[0]          
                M = token_idx.shape[0]

                                    
                alpha_i = pe[token_idx].unsqueeze(1)          
                target_i = r[token_idx] * alpha_i          

                                      
                h_i = ffn_s.experts[e].hidden(x2[token_idx]).to(dtype=dtype)          

                       
                expert_stats[e]["A"].add_(h_i.t().mm(h_i))          
                expert_stats[e]["B"].add_(h_i.t().mm(target_i))          
                expert_stats[e]["token_count"] += M

    handle_student = ffn_s.register_forward_hook(hook_student_ffn)

    try:
        teacher.eval()
        student.eval()
        _run_student_calibration_passes(
            student=student,
            calib_pairs=calib_pairs,
            cfg=cfg,
            desc=f"L{layer_idx} Experts fc2 stats",
            use_student_trajectory=use_student_trajectory,
            trajectory_seed=trajectory_seed,
            trajectory_top_k=trajectory_top_k,
            trajectory_top_p=trajectory_top_p,
        )
    finally:
        handle_student.remove()

    return expert_stats


@torch.no_grad()
def collect_expert_sequential_cache(
    teacher,
    student,
    layer_idx: int,
    calib_pairs: List[Tuple[torch.Tensor, torch.Tensor]],
    cfg: float,
    dtype: torch.dtype = torch.float32,
    max_tokens_per_call: int = 0,
    use_student_trajectory: bool = True,
    trajectory_seed: int = 42,
    trajectory_top_k: int = 900,
    trajectory_top_p: float = 0.96,
) -> Dict[str, torch.Tensor]:
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
       
    ffn_t = teacher.blocks[layer_idx].ffn
    ffn_s: VARD2MFFN = student.blocks[layer_idx].ffn

    C = ffn_s.in_features
    E = ffn_s.n_experts
    K = ffn_s.topk
    merge_device = ffn_s.shared.fc1.weight.device

    out_bias = ffn_s.out_bias.detach().to(dtype=dtype)

                         
    all_x2_list = []
    all_r0_list = []
    all_indices_sorted_list = []
    all_h_cache_list = []                                                      

    def _compute_teacher_ffn_output(x2: torch.Tensor) -> torch.Tensor:
                            
        y_t = ffn_t(x2)
        return y_t.to(dtype=dtype)

    def hook_student_ffn(_module, inp, out):
        if len(inp) == 0:
            return

        x = inp[0].detach()
        x2 = x.reshape(-1, x.shape[-1])

        if max_tokens_per_call and max_tokens_per_call > 0:
            N = x2.shape[0]
            if N > max_tokens_per_call:
                idx = torch.randperm(N, device=x2.device)[:max_tokens_per_call]
                x2 = x2.index_select(0, idx)

        x2 = x2.to(dtype=dtype)
        N = x2.shape[0]

                                                                                    
        y_teacher = _compute_teacher_ffn_output(x2)
        y_teacher = y_teacher - out_bias.view(1, -1)          

                                            
        h_shared = ffn_s.shared.hidden(x2).to(dtype=dtype)          
        y_shared = F.linear(h_shared, ffn_s.shared.fc2.weight)          

                                                          
        y_experts = _compute_experts_output(x2, ffn_s, dtype)          

              
        r0 = y_teacher - y_shared - y_experts          

                                         
        logits = ffn_s.gate(x2)          
        _, indices_sorted = torch.topk(logits.float(), k=K, dim=-1, sorted=True)                             

                                             
        h_cache = {}
        for e in range(E):
                                            
            mask_any = (indices_sorted == e).any(dim=1)       
            if mask_any.any():
                token_idx_e = mask_any.nonzero(as_tuple=True)[0]            
                h_e = ffn_s.experts[e].hidden(x2[token_idx_e]).to(dtype=dtype)          
                h_cache[e] = (token_idx_e, h_e)

                                                                                           
        all_x2_list.append(x2.detach().cpu())
        all_r0_list.append(r0.detach().cpu())
        all_indices_sorted_list.append(indices_sorted.detach().cpu())
        all_h_cache_list.append({
            e: (token_idx.detach().cpu(), h.detach().cpu())
            for e, (token_idx, h) in h_cache.items()
        })

    handle_student = ffn_s.register_forward_hook(hook_student_ffn)

    try:
        teacher.eval()
        student.eval()
                    
        _run_student_calibration_passes(
            student=student,
            calib_pairs=calib_pairs,
            cfg=cfg,
            desc=f"L{layer_idx} Collecting sequential cache",
            use_student_trajectory=use_student_trajectory,
            trajectory_seed=trajectory_seed,
            trajectory_top_k=trajectory_top_k,
            trajectory_top_p=trajectory_top_p,
        )
    finally:
        handle_student.remove()

                    
    if len(all_x2_list) == 0:
        return {}

    batch_token_counts = [t.shape[0] for t in all_x2_list]
    x2_all = torch.cat(all_x2_list, dim=0)                                                          
    r0_all = torch.cat(all_r0_list, dim=0).to(merge_device, dtype=dtype, non_blocking=True)
    indices_sorted_all = torch.cat(all_indices_sorted_list, dim=0).to(merge_device, non_blocking=True)
    del all_r0_list, all_indices_sorted_list
    if merge_device.type == "cuda":
        torch.cuda.empty_cache()

                                   
    h_cache_all = {}
    offset = 0
    for batch_idx, h_cache_batch in enumerate(all_h_cache_list):
        Nb = batch_token_counts[batch_idx]
        for e, (token_idx_local, h_local) in h_cache_batch.items():
                          
            token_idx_global = token_idx_local + offset
            if e not in h_cache_all:
                h_cache_all[e] = []
            h_cache_all[e].append((token_idx_global, h_local))
        offset += Nb

                                                                    
                                                                          
                                                                            
    for e in list(h_cache_all.keys()):
        if len(h_cache_all[e]) > 0:
            token_indices_list = []
            h_list = []
            for token_idx_global, h_e in h_cache_all[e]:
                token_indices_list.append(token_idx_global)
                h_list.append(h_e)
            token_idx_merged = torch.cat(token_indices_list, dim=0)
            h_merged = torch.cat(h_list, dim=0)
            h_cache_all[e] = (
                token_idx_merged.cpu(),
                h_merged.cpu().to(dtype=dtype),
            )
        else:
            del h_cache_all[e]

    return {
        "x2_all": x2_all,
        "r0_all": r0_all,
        "indices_sorted": indices_sorted_all,
        "h_cache": h_cache_all,
    }

"""
python run_var_sub.py \
    --moe_ckpt outputs/var_d2m_d16_75_two_stage.pth \
    --dense_ckpt /home/liying/pretrained/model_zoo/var_d16.pth \
    --vae_ckpt /home/liying/pretrained/model_zoo/vae_ch160v4096z32.pth \
    --save_refined_ckpt ./var_d2m_d16_75_two_stage_refined.pth \
    --depth 16 \
    --num_classes 1000 \
    --num_calib 200 \
    --calib_bs 4 \
    --calib_seed 42 \
    --ridge_lambda_shared 10.0 \
    --max_delta_norm 0.05 \
    --refine_experts \
    --ridge_lambda_expert 20.0 \
    --max_delta_norm_expert 0.02 \
    --min_tokens_per_expert 4096 \
    --use_images \
    --imagenet_dir /home/liying/datasets/imagenet
"""
