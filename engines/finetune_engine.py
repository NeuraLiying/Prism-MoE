                      
                       

                                                                                

import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.distributed as tdist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from common import dist
from common.path_utils import add_var_root
from utils.data import build_dataset


PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


def is_dist() -> bool:
    return dist.initialized()


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if is_dist():
        tdist.all_reduce(x, op=tdist.ReduceOp.SUM)
        x /= dist.get_world_size()
    return x


def unwrap_model(model):
    if isinstance(model, DDP):
        model = model.module
    if hasattr(model, "_orig_mod"):
        model = model._orig_mod
    return model


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_checkpoint_state(path: str) -> Tuple[dict, dict]:
    ckpt = torch.load(path, map_location="cpu")
    config = {}
    config_path = path.replace(".pth", "_config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config.update(json.load(f))
    if isinstance(ckpt, dict):
        if "config" in ckpt:
            config.update(ckpt["config"])
        if "trainer" in ckpt and "var_wo_ddp" in ckpt["trainer"]:
            return ckpt["trainer"]["var_wo_ddp"], config
        if "var_wo_ddp" in ckpt:
            return ckpt["var_wo_ddp"], config
        if "state_dict" in ckpt:
            return ckpt["state_dict"], config
    return ckpt, config


def infer_moe_config_from_state(
    state_dict: dict,
    init_config: dict,
    train_topk: int,
    hard_mode: bool,
    norm_topk_prob: bool,
) -> dict:
    gate_key = "blocks.0.ffn.gate.proj.weight"
    shared_key = "blocks.0.ffn.shared.fc1.weight"
    expert_key = "blocks.0.ffn.experts.0.fc1.weight"
    if gate_key not in state_dict or shared_key not in state_dict or expert_key not in state_dict:
        raise RuntimeError("Cannot infer MoE config from checkpoint; expected blocks.0.ffn shared/expert/gate keys.")
    n_experts = int(state_dict[gate_key].shape[0])
    shared_hidden = int(state_dict[shared_key].shape[0])
    expert_hidden = int(state_dict[expert_key].shape[0])
    total_hidden = shared_hidden + expert_hidden * n_experts
    shared_ratio = shared_hidden / float(total_hidden)
    router_bias = any(k.endswith(".ffn.gate.proj.bias") for k in state_dict)
    if bool(init_config.get("router_bias", False)) and not router_bias:
        raise RuntimeError("Init config declares router_bias=true, but checkpoint has no gate.proj.bias tensors.")
    return {
        "nexperts": n_experts,
        "topk": int(train_topk),
        "shared_ratio": shared_ratio,
        "hard_mode": bool(hard_mode),
        "router_bias": bool(router_bias),
        "norm_topk_prob": bool(norm_topk_prob),
        "router_temp": 1.0,
        "init_alpha": 0.0,
        "router_context_mode": "none",
        "router_context_init_alpha": 0.1,
        "router_context_cosine": False,
        "router_context_cosine_init_alpha": 0.0,
        "router_token_cosine": False,
        "router_token_cosine_init_alpha": 0.0,
        "router_capture_input_sample_tokens": 0,
    }


def load_moe_init_into_trainable_model(model, init_state: dict) -> None:
    model_state = model.state_dict()
    loadable = {}
    skipped_shape = []
    dropped_init_keys = []
    for k, v in init_state.items():
        if k in model_state and tuple(model_state[k].shape) == tuple(v.shape):
            loadable[k] = v
        elif k in model_state:
            skipped_shape.append((k, tuple(v.shape), tuple(model_state[k].shape)))
        elif ".ffn." in k:
            dropped_init_keys.append(k)
    ret = model.load_state_dict(loadable, strict=False)
    unexpected = [k for k in ret.unexpected_keys]
    missing_allowed = []
    missing_bad = []
    for k in ret.missing_keys:
        if (
            ".ffn.gate.delta." in k
            or k.endswith(".ffn.gate.alpha")
            or ".ffn.gate.cond_proj." in k
            or ".ffn.gate.stage_embed." in k
            or ".ffn.gate.branch_embed." in k
            or ".ffn.gate.cond_token_proj." in k
            or ".ffn.gate.cond_context_proj." in k
            or ".ffn.gate.cond_interaction_out." in k
            or k.endswith(".ffn.gate.cond_cosine_proto")
            or k.endswith(".ffn.gate.token_cosine_proto")
            or k.endswith(".ffn.gate.dynamic_bias")
            or k.endswith(".ffn.gate.dynamic_stage_bias")
            or k.endswith(".ffn.gate.context_alpha")
            or k.endswith(".ffn.gate.cond_interaction_alpha")
            or k.endswith(".ffn.gate.cond_cosine_alpha")
            or k.endswith(".ffn.gate.token_cosine_alpha")
        ):
            missing_allowed.append(k)
        else:
            missing_bad.append(k)
    if skipped_shape:
        raise RuntimeError(f"Shape-mismatched checkpoint keys: {skipped_shape[:10]}")
    if dropped_init_keys:
        raise RuntimeError(f"Initialized MoE keys not present in trainable model: {dropped_init_keys[:20]}")
    if unexpected:
        raise RuntimeError(f"Unexpected keys while loading init checkpoint: {unexpected[:20]}")
    if missing_bad:
        raise RuntimeError(f"Missing non-router-delta keys while loading init checkpoint: {missing_bad[:20]}")
    if dist.is_master():
        print(f"Loaded initialized MoE weights: {len(loadable)} tensors")
        print(f"Initialized new trainable router tensors: {len(missing_allowed)}")


@torch.no_grad()
def reset_trainable_router_deltas(
    model,
    init_alpha: float,
    context_init_alpha: float = 0.1,
    context_interaction_init_alpha: float = 0.1,
    context_cosine_init_alpha: float = 0.0,
    token_cosine_init_alpha: float = 0.0,
) -> None:
    for block in model.blocks:
        ffn = block.ffn
        gate = getattr(ffn, "gate", None)
        if gate is None:
            continue
        if hasattr(gate, "delta"):
            last = gate.delta[-1]
            if hasattr(last, "weight") and last.weight is not None:
                last.weight.zero_()
            if hasattr(last, "bias") and last.bias is not None:
                last.bias.zero_()
        if hasattr(gate, "alpha"):
            gate.alpha.fill_(float(init_alpha))
        if hasattr(gate, "cond_proj") and gate.cond_proj is not None:
            gate.cond_proj.weight.zero_()
        if hasattr(gate, "stage_embed") and gate.stage_embed is not None:
            gate.stage_embed.weight.zero_()
        if hasattr(gate, "branch_embed") and gate.branch_embed is not None:
            gate.branch_embed.weight.zero_()
        if hasattr(gate, "context_alpha") and gate.context_alpha is not None:
            gate.context_alpha.fill_(float(context_init_alpha))
        if hasattr(gate, "cond_interaction_out") and gate.cond_interaction_out is not None:
            gate.cond_interaction_out.weight.zero_()
        if hasattr(gate, "cond_interaction_alpha") and gate.cond_interaction_alpha is not None:
            gate.cond_interaction_alpha.fill_(float(context_interaction_init_alpha))
        if hasattr(gate, "cond_cosine_proto") and gate.cond_cosine_proto is not None:
            gate.cond_cosine_proto.copy_(gate.proj.weight.detach())
        if hasattr(gate, "cond_cosine_alpha") and gate.cond_cosine_alpha is not None:
            gate.cond_cosine_alpha.fill_(float(context_cosine_init_alpha))
        if hasattr(gate, "token_cosine_proto") and gate.token_cosine_proto is not None:
            gate.token_cosine_proto.copy_(gate.proj.weight.detach())
        if hasattr(gate, "token_cosine_alpha") and gate.token_cosine_alpha is not None:
            gate.token_cosine_alpha.fill_(float(token_cosine_init_alpha))
        if hasattr(gate, "dynamic_bias"):
            gate.dynamic_bias.zero_()
        if hasattr(gate, "dynamic_stage_bias"):
            gate.dynamic_stage_bias.zero_()


def build_models(args, device):
    add_var_root(args.var_root)
    from models import build_vae_var

    init_state, init_config = load_checkpoint_state(args.moe_init_ckpt)
    moe_config = infer_moe_config_from_state(
        init_state,
        init_config,
        train_topk=args.train_topk,
        hard_mode=args.hard_mode,
        norm_topk_prob=args.norm_topk_prob,
    )
    moe_config["router_temp"] = args.router_temp
    moe_config["init_alpha"] = args.router_init_alpha
    moe_config["delta_hidden_mult"] = args.router_delta_hidden_mult
    moe_config["router_context_mode"] = args.router_context_mode
    moe_config["router_context_init_alpha"] = args.router_context_init_alpha
    moe_config["router_context_interaction_rank"] = args.router_context_interaction_rank
    moe_config["router_context_interaction_init_alpha"] = args.router_context_interaction_init_alpha
    moe_config["router_context_cosine"] = bool(args.router_context_cosine)
    moe_config["router_context_cosine_init_alpha"] = args.router_context_cosine_init_alpha
    moe_config["router_token_cosine"] = bool(args.router_token_cosine)
    moe_config["router_token_cosine_init_alpha"] = args.router_token_cosine_init_alpha
    moe_config["router_logit_mode"] = args.router_logit_mode
    moe_config["router_cosine_tau"] = args.router_cosine_tau
    moe_config["router_capture_input_sample_tokens"] = args.router_capture_input_sample_tokens

    vae, student = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=PATCH_NUMS,
        num_classes=args.num_classes,
        depth=args.depth,
        shared_aln=args.shared_aln,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
        use_moe=True,
        moe_config=moe_config,
    )
    _, teacher = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=PATCH_NUMS,
        num_classes=args.num_classes,
        depth=args.depth,
        shared_aln=args.shared_aln,
        flash_if_available=args.fuse,
        fused_if_available=args.fuse,
        use_moe=False,
        moe_config=None,
    )

    vae.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu"), strict=True)
    dense_state, _ = load_checkpoint_state(args.dense_ckpt)
    teacher.load_state_dict(dense_state, strict=True)
    load_moe_init_into_trainable_model(student, init_state)
    if not args.preserve_router_delta_from_init:
        reset_trainable_router_deltas(
            student,
            args.router_init_alpha,
            args.router_context_init_alpha,
            args.router_context_interaction_init_alpha,
            args.router_context_cosine_init_alpha,
            args.router_token_cosine_init_alpha,
        )

    teacher.eval()
    vae.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    for p in vae.parameters():
        p.requires_grad_(False)

    active_ratio = moe_config["shared_ratio"] + (args.train_topk / moe_config["nexperts"]) * (1.0 - moe_config["shared_ratio"])
    meta = {
        "init_config": init_config,
        "moe_config": moe_config,
        "active_ratio": active_ratio,
        "patch_nums": PATCH_NUMS,
    }
    return vae, teacher, student, meta


def set_trainable(model, train_scope: str) -> None:
    model = unwrap_model(model)
    for p in model.parameters():
        p.requires_grad_(False)

    for block in model.blocks:
        ffn = block.ffn
        if not hasattr(ffn, "experts"):
            continue
        if train_scope in {"router", "router_experts", "all_moe"}:
            for p in ffn.gate.parameters():
                p.requires_grad_(True)
        if train_scope in {"router_experts", "all_moe"}:
            for expert in ffn.experts:
                for p in expert.parameters():
                    p.requires_grad_(True)
        if train_scope == "all_moe":
            for p in ffn.shared.parameters():
                p.requires_grad_(True)
            ffn.out_bias.requires_grad_(True)
    if train_scope == "full":
        for p in model.parameters():
            p.requires_grad_(True)


def apply_trainable_config_overrides(model, args) -> None:
    model = unwrap_model(model)
    if args.router_logit_mode == "cosine":
        for block in model.blocks:
            ffn = getattr(block, "ffn", None)
            gate = getattr(ffn, "gate", None)
            proj = getattr(gate, "proj", None)
            if proj is not None and proj.bias is not None:
                proj.bias.requires_grad_(False)


def make_optimizer(model, args):
    wd_params, nowd_params = [], []
    for name, p in unwrap_model(model).named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith("bias") or "pos_" in name or "lvl_embed" in name or "class_emb" in name:
            nowd_params.append(p)
        else:
            wd_params.append(p)
    return torch.optim.AdamW(
        [
            {"params": wd_params, "weight_decay": args.weight_decay},
            {"params": nowd_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
        fused=args.fused_adamw,
    )


def lr_factor(step: int, total_steps: int, warmup_steps: int, min_factor: float) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return max(min_factor, (step + 1) / warmup_steps)
    if total_steps <= warmup_steps:
        return 1.0
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_factor + (1.0 - min_factor) * 0.5 * (1.0 + math.cos(math.pi * t))


def ramp_factor(global_step: int, warmup_steps: int, ramp_steps: int) -> float:
    if global_step < warmup_steps:
        return 0.0
    if ramp_steps > 0:
        return min(1.0, (global_step - warmup_steps + 1) / ramp_steps)
    return 1.0


def parse_topk_schedule(spec: str, epochs: int, default_topk: int) -> list:
    if not spec:
        return [int(default_topk)] * int(epochs)
    schedule = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "x" in part:
            k_str, n_str = part.split("x", 1)
            k, n = int(k_str), int(n_str)
        elif ":" in part:
            k_str, n_str = part.split(":", 1)
            k, n = int(k_str), int(n_str)
        else:
            k, n = int(part), 1
        if k <= 0 or n <= 0:
            raise ValueError(f"Invalid top-k schedule segment: {part!r}")
        schedule.extend([k] * n)
    if not schedule:
        raise ValueError(f"Empty top-k schedule from {spec!r}")
    if len(schedule) < epochs:
        schedule.extend([schedule[-1]] * (epochs - len(schedule)))
    return schedule[:epochs]


@torch.no_grad()
def set_model_runtime_topk(model, topk: int) -> None:
    model = unwrap_model(model)
    for block in model.blocks:
        ffn = getattr(block, "ffn", None)
        gate = getattr(ffn, "gate", None)
        if gate is None:
            continue
        k = max(1, min(int(topk), int(ffn.n_experts)))
        ffn.topk = k
        gate.topk = k


def expert_recon_alignment_loss(student, teacher, args) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
\
\
\
\
       
    device = next(unwrap_model(student).parameters()).device
    erc_total = torch.zeros((), device=device)
    combo_total = torch.zeros((), device=device)
    stats = {
        "router_erc_loss": 0.0,
        "selected_combo_kd_loss": 0.0,
        "router_erc_target_entropy": 0.0,
        "expert_recon_layers": 0.0,
        "expert_recon_tokens": 0.0,
        "recon_marginal_target_entropy": 0.0,
        "recon_marginal_target_top1_prob": 0.0,
        "recon_marginal_score_cv": 0.0,
        "recon_marginal_selected_target_mass": 0.0,
        "recon_marginal_selected_top1_hit": 0.0,
        "selected_combo_rel_l2": 0.0,
    }
    if args.router_erc_weight <= 0 and args.selected_combo_kd_weight <= 0:
        return erc_total, combo_total, stats

    s_model = unwrap_model(student)
    t_model = unwrap_model(teacher)
    layer_filter_raw = getattr(args, "expert_recon_layers", None)
    layer_filter = {int(v) for v in layer_filter_raw} if layer_filter_raw else None

    for layer_idx, (s_block, t_block) in enumerate(zip(s_model.blocks, t_model.blocks)):
        if layer_filter is not None and layer_idx not in layer_filter:
            continue
        s_ffn = getattr(s_block, "ffn", None)
        t_ffn = getattr(t_block, "ffn", None)
        gate = getattr(s_ffn, "gate", None)
        if (
            s_ffn is None
            or t_ffn is None
            or gate is None
            or getattr(s_ffn, "last_input_sample", None) is None
            or gate.last_probs is None
            or gate.last_indices is None
            or getattr(s_ffn, "experts", None) is None
            or getattr(s_ffn, "last_all_expert_outputs", None) is None
        ):
            continue

        sample_x = s_ffn.last_input_sample.to(device=device)
        sample_idx = s_ffn.last_input_sample_indices
        if sample_idx is None or sample_x.numel() == 0:
            continue
        sample_idx = sample_idx.to(device=gate.last_probs.device, dtype=torch.long)
        max_tokens = int(args.expert_recon_sample_tokens)
        if max_tokens > 0 and sample_idx.numel() > max_tokens:
            keep = torch.linspace(0, sample_idx.numel() - 1, max_tokens, device=sample_x.device).round().long()
            sample_x = sample_x.index_select(0, keep)
            sample_idx = sample_idx.index_select(0, keep)
            expert_outs = s_ffn.last_all_expert_outputs.index_select(0, keep).float()
        else:
            expert_outs = s_ffn.last_all_expert_outputs.float()

        with torch.no_grad():
            dense_out = t_ffn(sample_x).float()
            shared_out = s_ffn.shared(sample_x).float()
            dense_gap = dense_out - shared_out

        route_probs = gate.last_probs.index_select(0, sample_idx).float()
        selected_indices = gate.last_indices.index_select(0, sample_idx).to(device=expert_outs.device, dtype=torch.long)
        selected_probs = route_probs.gather(1, selected_indices)
        selected_weights = selected_probs / selected_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)

        selected_expert_outs = expert_outs.gather(
            1,
            selected_indices.unsqueeze(-1).expand(-1, -1, expert_outs.shape[-1]),
        )
        combo_out = (selected_expert_outs * selected_weights.unsqueeze(-1)).sum(dim=1)
        combo_err = combo_out.float() - dense_gap
        stats["selected_combo_rel_l2"] += float(
            (combo_err.norm(dim=-1) / dense_gap.norm(dim=-1).clamp_min(1e-6)).mean().detach().item()
        )
        if args.selected_combo_kd_weight > 0:
            combo_total = combo_total + combo_err.pow(2).mean()

        if args.router_erc_weight > 0:
            with torch.no_grad():
                expert_detached = expert_outs.detach().float()
                gap_detached = dense_gap.detach().float()
                score = (
                    2.0 * (expert_detached * gap_detached.unsqueeze(1)).sum(dim=-1)
                    - expert_detached.pow(2).sum(dim=-1)
                )
                score_mean_keep = score.mean(dim=1, keepdim=True)
                score_std_keep = score.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-6)
                erc_logits = (score - score_mean_keep) / score_std_keep
                erc_logits = erc_logits / max(float(args.router_erc_target_temperature), 1e-6)
                erc_target = F.softmax(erc_logits, dim=-1)
                score_mean = score.abs().mean(dim=1)
                score_std = score.std(dim=1, unbiased=False)
                stats["recon_marginal_score_cv"] += float(
                    (score_std / score_mean.clamp_min(1e-6)).mean().item()
                )
            erc_loss = -(erc_target * route_probs.clamp_min(1e-9).log()).sum(dim=-1).mean()
            erc_total = erc_total + erc_loss
            erc_entropy = -(erc_target.clamp_min(1e-9) * erc_target.clamp_min(1e-9).log()).sum(dim=-1).mean()
            stats["router_erc_target_entropy"] += float(erc_entropy.detach().item())
            stats["recon_marginal_target_entropy"] += float(erc_entropy.detach().item())
            stats["recon_marginal_target_top1_prob"] += float(erc_target.max(dim=-1).values.mean().item())
            stats["recon_marginal_selected_target_mass"] += float(
                erc_target.gather(1, selected_indices).sum(dim=-1).mean().item()
            )
            target_top1 = erc_target.argmax(dim=-1, keepdim=True)
            stats["recon_marginal_selected_top1_hit"] += float(
                (selected_indices == target_top1).any(dim=1).float().mean().item()
            )

        stats["expert_recon_layers"] += 1.0
        stats["expert_recon_tokens"] += float(sample_x.shape[0])

    if stats["expert_recon_layers"] > 0:
        n = stats["expert_recon_layers"]
        erc_total = erc_total / n
        combo_total = combo_total / n
        stats["router_erc_loss"] = float(erc_total.detach().item())
        stats["selected_combo_kd_loss"] = float(combo_total.detach().item())
        stats["router_erc_target_entropy"] /= n
        stats["expert_recon_tokens"] /= n
        stats["recon_marginal_target_entropy"] /= n
        stats["recon_marginal_target_top1_prob"] /= n
        stats["recon_marginal_score_cv"] /= n
        stats["recon_marginal_selected_target_mass"] /= n
        stats["recon_marginal_selected_top1_hit"] /= n
        stats["selected_combo_rel_l2"] /= n
    return erc_total, combo_total, stats

def router_regularization(model, args) -> Tuple[torch.Tensor, Dict[str, float]]:
                                                                              
    device = next(unwrap_model(model).parameters()).device
    total = torch.zeros((), device=device)
    stats = {
        "router_balance_loss": 0.0,
        "router_z_loss": 0.0,
        "router_entropy": 0.0,
        "router_topk_entropy": 0.0,
        "router_topk_entropy_ratio": 0.0,
        "router_topk_max_freq": 0.0,
        "router_near_zero_experts": 0.0,
        "router_layers": 0.0,
        "router_token_cosine_alpha": 0.0,
        "router_context_alpha": 0.0,
        "router_context_stage_embed_norm": 0.0,
        "router_context_cond_proj_norm": 0.0,
    }
    for block in unwrap_model(model).blocks:
        ffn = block.ffn
        gate = getattr(ffn, "gate", None)
        if gate is None or gate.last_logits is None or gate.last_probs is None:
            continue
        probs = gate.last_probs.float()
        mean_p = probs.mean(dim=0)
        max_entropy = math.log(probs.shape[-1])
        entropy = -(mean_p.clamp_min(1e-9) * mean_p.clamp_min(1e-9).log()).sum()
        balance = 1.0 - entropy / max_entropy
        z_loss = torch.logsumexp(gate.last_logits.float(), dim=-1).pow(2).mean()

        topk_entropy = probs.new_zeros(())
        topk_entropy_ratio = probs.new_zeros(())
        topk_max_freq = probs.new_zeros(())
        near_zero_experts = probs.new_zeros(())
        if gate.last_indices is not None:
            indices = gate.last_indices.to(device=probs.device)
            hard_mask = F.one_hot(indices, num_classes=probs.shape[-1]).float().sum(dim=1)
            hard_mask = hard_mask / max(1, indices.shape[-1])
            hard_freq = hard_mask.mean(dim=0)
            topk_entropy = -(hard_freq.clamp_min(1e-9) * hard_freq.clamp_min(1e-9).log()).sum()
            topk_entropy_ratio = topk_entropy / max_entropy
            topk_max_freq = hard_freq.max()
            near_zero_experts = (hard_freq < args.router_near_zero_threshold).float().sum()

        total = total + args.router_balance_weight * balance + args.router_z_weight * z_loss
        stats["router_balance_loss"] += float(balance.detach().item())
        stats["router_z_loss"] += float(z_loss.detach().item())
        stats["router_entropy"] += float(entropy.detach().item())
        stats["router_topk_entropy"] += float(topk_entropy.detach().item())
        stats["router_topk_entropy_ratio"] += float(topk_entropy_ratio.detach().item())
        stats["router_topk_max_freq"] += float(topk_max_freq.detach().item())
        stats["router_near_zero_experts"] += float(near_zero_experts.detach().item())
        if hasattr(gate, "token_cosine_alpha") and gate.token_cosine_alpha is not None:
            stats["router_token_cosine_alpha"] += float(gate.token_cosine_alpha.detach().item())
        if hasattr(gate, "context_alpha") and gate.context_alpha is not None:
            stats["router_context_alpha"] += float(gate.context_alpha.detach().item())
        if hasattr(gate, "stage_embed") and gate.stage_embed is not None:
            stats["router_context_stage_embed_norm"] += float(gate.stage_embed.weight.detach().float().norm().item())
        if hasattr(gate, "cond_proj") and gate.cond_proj is not None:
            stats["router_context_cond_proj_norm"] += float(gate.cond_proj.weight.detach().float().norm().item())
        stats["router_layers"] += 1.0

    if stats["router_layers"] > 0:
        n = stats["router_layers"]
        for key in list(stats):
            if key != "router_layers":
                stats[key] /= n
        total = total / n
    return total, stats

@torch.no_grad()
def collect_router_stats(model) -> dict:
    model = unwrap_model(model)
    out = {}
    for layer_idx, block in enumerate(model.blocks):
        ffn = block.ffn
        gate = getattr(ffn, "gate", None)
        if gate is None:
            continue
        layer = {
            "n_experts": int(ffn.n_experts),
            "topk": int(ffn.topk),
            "hard_mode": bool(ffn.hard_mode),
        }
        if gate.last_indices is not None:
            counts = torch.bincount(gate.last_indices.reshape(-1).cpu(), minlength=ffn.n_experts).float()
            total = counts.sum().clamp_min(1)
            layer["topk_counts"] = counts.tolist()
            layer["topk_freq"] = (counts / total).tolist()
            layer["topk_entropy"] = float((-(counts / total).clamp_min(1e-9) * (counts / total).clamp_min(1e-9).log()).sum().item())
        if gate.last_probs is not None:
            mean_p = gate.last_probs.float().mean(dim=0).detach().cpu()
            layer["prob_mean"] = mean_p.tolist()
            layer["prob_entropy"] = float((-(mean_p.clamp_min(1e-9)) * mean_p.clamp_min(1e-9).log()).sum().item())
        layer["router_alpha"] = float(gate.alpha.detach().cpu().item()) if hasattr(gate, "alpha") else 0.0
        layer["context_alpha"] = (
            float(gate.context_alpha.detach().cpu().item())
            if hasattr(gate, "context_alpha") and gate.context_alpha is not None
            else 0.0
        )
        layer["stage_embed_norm"] = (
            float(gate.stage_embed.weight.detach().float().cpu().norm().item())
            if hasattr(gate, "stage_embed") and gate.stage_embed is not None
            else 0.0
        )
        layer["cond_proj_norm"] = (
            float(gate.cond_proj.weight.detach().float().cpu().norm().item())
            if hasattr(gate, "cond_proj") and gate.cond_proj is not None
            else 0.0
        )
        out[str(layer_idx)] = layer
    return out


@torch.no_grad()
def init_router_stats_accum(model) -> dict:
    model = unwrap_model(model)
    device = next(model.parameters()).device
    accum = {}
    for layer_idx, block in enumerate(model.blocks):
        ffn = block.ffn
        gate = getattr(ffn, "gate", None)
        if gate is None:
            continue
        n_experts = int(ffn.n_experts)
        accum[str(layer_idx)] = {
            "counts": torch.zeros(n_experts, device=device),
            "slots": torch.zeros((), device=device),
            "prob_sum": torch.zeros(n_experts, device=device),
            "prob_tokens": torch.zeros((), device=device),
            "sampled_batches": torch.zeros((), device=device),
        }
    return accum


@torch.no_grad()
def accumulate_router_stats(model, accum: dict) -> None:
    model = unwrap_model(model)
    for layer_idx, block in enumerate(model.blocks):
        ffn = block.ffn
        gate = getattr(ffn, "gate", None)
        layer = accum.get(str(layer_idx))
        if gate is None or layer is None:
            continue
        touched = False
        if gate.last_indices is not None:
            counts = torch.bincount(gate.last_indices.reshape(-1), minlength=ffn.n_experts).float()
            layer["counts"].add_(counts.to(layer["counts"].device))
            layer["slots"].add_(counts.sum())
            touched = True
        if gate.last_probs is not None:
            probs = gate.last_probs.float()
            layer["prob_sum"].add_(probs.sum(dim=0).to(layer["prob_sum"].device))
            layer["prob_tokens"].add_(probs.shape[0])
            touched = True
        if touched:
            layer["sampled_batches"].add_(1)


@torch.no_grad()
def finalize_router_stats(model, accum: dict) -> dict:
    model = unwrap_model(model)
    out = {}
    for layer_idx, block in enumerate(model.blocks):
        ffn = block.ffn
        gate = getattr(ffn, "gate", None)
        layer_accum = accum.get(str(layer_idx))
        if gate is None or layer_accum is None:
            continue
        for key in ("counts", "slots", "prob_sum", "prob_tokens", "sampled_batches"):
            if is_dist():
                tdist.all_reduce(layer_accum[key], op=tdist.ReduceOp.SUM)

        counts = layer_accum["counts"].detach().cpu()
        slots = float(layer_accum["slots"].detach().cpu().item())
        freq = counts / max(slots, 1.0)
        layer = {
            "n_experts": int(ffn.n_experts),
            "topk": int(ffn.topk),
            "hard_mode": bool(ffn.hard_mode),
            "sampled_batches": int(layer_accum["sampled_batches"].detach().cpu().item()),
            "topk_counts": counts.tolist(),
            "topk_freq": freq.tolist(),
            "topk_entropy": float((-(freq.clamp_min(1e-9)) * freq.clamp_min(1e-9).log()).sum().item()),
            "unused_experts": int((counts == 0).sum().item()),
            "max_freq": float(freq.max().item()) if len(freq) else 0.0,
            "min_freq": float(freq.min().item()) if len(freq) else 0.0,
        }
        prob_tokens = float(layer_accum["prob_tokens"].detach().cpu().item())
        if prob_tokens > 0:
            mean_p = (layer_accum["prob_sum"] / prob_tokens).detach().cpu()
            layer["prob_mean"] = mean_p.tolist()
            layer["prob_entropy"] = float((-(mean_p.clamp_min(1e-9)) * mean_p.clamp_min(1e-9).log()).sum().item())
        layer["router_alpha"] = float(gate.alpha.detach().cpu().item()) if hasattr(gate, "alpha") else 0.0
        layer["context_alpha"] = (
            float(gate.context_alpha.detach().cpu().item())
            if hasattr(gate, "context_alpha") and gate.context_alpha is not None
            else 0.0
        )
        layer["stage_embed_norm"] = (
            float(gate.stage_embed.weight.detach().float().cpu().norm().item())
            if hasattr(gate, "stage_embed") and gate.stage_embed is not None
            else 0.0
        )
        layer["cond_proj_norm"] = (
            float(gate.cond_proj.weight.detach().float().cpu().norm().item())
            if hasattr(gate, "cond_proj") and gate.cond_proj is not None
            else 0.0
        )
        out[str(layer_idx)] = layer
    return out


@torch.no_grad()
def update_router_dynamic_bias(model, args, global_step: int) -> Dict[str, float]:
    if args.router_dynamic_bias_weight <= 0:
        return {
            "router_dynamic_bias_factor": 0.0,
            "router_dynamic_bias_mean_abs": 0.0,
            "router_dynamic_stage_bias_mean_abs": 0.0,
            "router_dynamic_bias_updates": 0.0,
        }
    factor = ramp_factor(
        global_step,
        args.router_dynamic_bias_warmup_steps,
        args.router_dynamic_bias_ramp_steps,
    )
    if factor <= 0.0:
        return {
            "router_dynamic_bias_factor": 0.0,
            "router_dynamic_bias_mean_abs": 0.0,
            "router_dynamic_stage_bias_mean_abs": 0.0,
            "router_dynamic_bias_updates": 0.0,
        }
    model = unwrap_model(model)
    mean_abs = []
    stage_mean_abs = []
    updates = 0
    for block in model.blocks:
        ffn = getattr(block, "ffn", None)
        gate = getattr(ffn, "gate", None)
        if gate is None or gate.last_indices is None:
            continue
        indices = gate.last_indices.reshape(-1).to(device=gate.dynamic_bias.device)
        counts = torch.bincount(indices, minlength=ffn.n_experts).float()
        if is_dist():
            tdist.all_reduce(counts, op=tdist.ReduceOp.SUM)
        if counts.sum() > 0:
            freq = counts / counts.sum().clamp_min(1.0)
            target = torch.full_like(freq, 1.0 / max(1, ffn.n_experts))
            delta = (target - freq).clamp(
                min=-float(args.router_dynamic_bias_max_delta),
                max=float(args.router_dynamic_bias_max_delta),
            )
            gate.dynamic_bias.add_(float(args.router_dynamic_bias_weight) * float(factor) * delta)
            gate.dynamic_bias.clamp_(
                min=-float(args.router_dynamic_bias_clip),
                max=float(args.router_dynamic_bias_clip),
            )
            mean_abs.append(gate.dynamic_bias.float().abs().mean())
            updates += 1

        if args.router_dynamic_stage_bias_weight <= 0 or gate.dynamic_stage_bias.numel() == 0:
            continue
        stage_ids = getattr(ffn, "last_stage_ids_flat", None)
        if stage_ids is None:
            continue
        stage_ids = stage_ids.reshape(-1).to(device=gate.dynamic_stage_bias.device, dtype=torch.long)
        flat_idx = gate.last_indices.to(device=gate.dynamic_stage_bias.device)
        if stage_ids.shape[0] != flat_idx.shape[0]:
            continue
        num_stages = gate.dynamic_stage_bias.shape[0]
        stage_ids = stage_ids.clamp(0, num_stages - 1)
        group_ids = stage_ids.unsqueeze(1) * ffn.n_experts + flat_idx
        stage_counts = torch.bincount(
            group_ids.reshape(-1),
            minlength=num_stages * ffn.n_experts,
        ).float().reshape(num_stages, ffn.n_experts)
        if is_dist():
            tdist.all_reduce(stage_counts, op=tdist.ReduceOp.SUM)
        stage_slots = stage_counts.sum(dim=-1, keepdim=True)
        valid = stage_slots.squeeze(-1) >= float(args.router_dynamic_stage_bias_min_slots)
        if bool(valid.any()):
            stage_freq = stage_counts[valid] / stage_slots[valid].clamp_min(1.0)
            stage_target = torch.full_like(stage_freq, 1.0 / max(1, ffn.n_experts))
            stage_delta = (stage_target - stage_freq).clamp(
                min=-float(args.router_dynamic_bias_max_delta),
                max=float(args.router_dynamic_bias_max_delta),
            )
            gate.dynamic_stage_bias[valid].add_(
                float(args.router_dynamic_stage_bias_weight) * float(factor) * stage_delta
            )
            gate.dynamic_stage_bias.clamp_(
                min=-float(args.router_dynamic_bias_clip),
                max=float(args.router_dynamic_bias_clip),
            )
            stage_mean_abs.append(gate.dynamic_stage_bias.float().abs().mean())
    return {
        "router_dynamic_bias_factor": float(factor),
        "router_dynamic_bias_mean_abs": float(torch.stack(mean_abs).mean().item()) if mean_abs else 0.0,
        "router_dynamic_stage_bias_mean_abs": (
            float(torch.stack(stage_mean_abs).mean().item()) if stage_mean_abs else 0.0
        ),
        "router_dynamic_bias_updates": float(updates),
    }


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def append_jsonl(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, sort_keys=True) + "\n")


def save_checkpoint(path: Path, model, optimizer, scaler, epoch: int, step: int, args, meta: dict) -> None:
    if not dist.is_master():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "step": step,
            "var_wo_ddp": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "args": vars(args),
            "config": {
                **meta["moe_config"],
                "active_ratio": meta["active_ratio"],
                "method": "prism_moe_trainable_router_finetune",
                "train_scope": args.train_scope,
                "train_with_cfg_pair": args.train_with_cfg_pair,
                "router_balance_weight": args.router_balance_weight,
                "router_topk_schedule": args.router_topk_schedule,
                "router_dynamic_bias_weight": args.router_dynamic_bias_weight,
                "router_dynamic_stage_bias_weight": args.router_dynamic_stage_bias_weight,
                "router_dynamic_bias_clip": args.router_dynamic_bias_clip,
                "router_dynamic_bias_max_delta": args.router_dynamic_bias_max_delta,
                "router_dynamic_bias_warmup_steps": args.router_dynamic_bias_warmup_steps,
                "router_dynamic_bias_ramp_steps": args.router_dynamic_bias_ramp_steps,
                "router_dynamic_stage_bias_min_slots": args.router_dynamic_stage_bias_min_slots,
                "router_token_cosine": args.router_token_cosine,
                "router_token_cosine_init_alpha": args.router_token_cosine_init_alpha,
                "router_logit_mode": args.router_logit_mode,
                "router_cosine_tau": args.router_cosine_tau,
                "router_capture_input_sample_tokens": args.router_capture_input_sample_tokens,
                "expert_recon_sample_tokens": args.expert_recon_sample_tokens,
                "selected_combo_kd_weight": args.selected_combo_kd_weight,
                "router_erc_weight": args.router_erc_weight,
                "router_erc_target_temperature": args.router_erc_target_temperature,
                "logit_kd_weight": args.logit_kd_weight,
                "logit_kd_temperature": args.logit_kd_temperature,
                "logit_kd_stage_weight_power": args.logit_kd_stage_weight_power,
                "logit_kd_stage_weight_min": args.logit_kd_stage_weight_min,
                "router_z_weight": args.router_z_weight,
            },
        },
        path,
    )


def is_allowed_new_router_key(key: str) -> bool:
    return (
        ".ffn.gate.delta." in key
        or key.endswith(".ffn.gate.alpha")
        or ".ffn.gate.cond_proj." in key
        or ".ffn.gate.stage_embed." in key
        or ".ffn.gate.branch_embed." in key
        or ".ffn.gate.cond_token_proj." in key
        or ".ffn.gate.cond_context_proj." in key
        or ".ffn.gate.cond_interaction_out." in key
        or key.endswith(".ffn.gate.cond_cosine_proto")
        or key.endswith(".ffn.gate.token_cosine_proto")
        or key.endswith(".ffn.gate.dynamic_bias")
        or key.endswith(".ffn.gate.dynamic_stage_bias")
        or key.endswith(".ffn.gate.context_alpha")
        or key.endswith(".ffn.gate.cond_interaction_alpha")
        or key.endswith(".ffn.gate.cond_cosine_alpha")
        or key.endswith(".ffn.gate.token_cosine_alpha")
    )


def load_resume_checkpoint(
    path: str,
    model,
    optimizer,
    scaler,
    load_optimizer: bool = True,
    allow_missing_router: bool = False,
) -> Tuple[int, int, dict]:
    ckpt = torch.load(path, map_location="cpu")
    if not isinstance(ckpt, dict) or "var_wo_ddp" not in ckpt:
        raise RuntimeError(f"Resume checkpoint has unsupported format: {path}")
    ret = unwrap_model(model).load_state_dict(ckpt["var_wo_ddp"], strict=not allow_missing_router)
    missing_bad = [k for k in ret.missing_keys if not is_allowed_new_router_key(k)]
    if missing_bad or ret.unexpected_keys:
        raise RuntimeError(f"Resume checkpoint load mismatch: missing={missing_bad[:20]}, unexpected={ret.unexpected_keys[:20]}")
    if allow_missing_router and ret.missing_keys and dist.is_master():
        print(f"Initialized missing new router tensors while resuming: {len(ret.missing_keys)}", flush=True)
    if load_optimizer:
        if "optimizer" not in ckpt:
            raise RuntimeError(f"Resume checkpoint has no optimizer state: {path}")
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
    return int(ckpt.get("epoch", 0)), int(ckpt.get("step", 0)), ckpt


def infer_resume_schedule_steps_per_epoch(ckpt: dict, fallback: int) -> int:
    if not isinstance(ckpt, dict):
        return fallback
    ckpt_args = ckpt.get("args") if isinstance(ckpt.get("args"), dict) else {}
    saved_schedule = int(ckpt_args.get("lr_schedule_steps_per_epoch", 0) or 0)
    if saved_schedule > 0:
        return saved_schedule
    epoch = int(ckpt.get("epoch", 0) or 0)
    step = int(ckpt.get("step", 0) or 0)
    if epoch > 0 and step % epoch == 0:
        return max(1, step // epoch)
    if epoch > 0 and step > 0:
        return max(1, round(step / epoch))
    return fallback


def prune_resume_logs(out_dir: Path, resume_epoch: int, resume_step: int) -> None:
    def filter_jsonl(path: Path, keep_fn) -> None:
        if not path.exists():
            return
        kept = []
        changed = False
        with open(path, "r") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                keep = keep_fn(obj)
                changed = changed or not keep
                if keep:
                    kept.append(line)
        if changed:
            tmp = path.with_suffix(path.suffix + ".tmp")
            with open(tmp, "w") as f:
                f.writelines(kept)
            os.replace(tmp, path)

    logs_dir = out_dir / "logs"
    stale_done = logs_dir / "done.json"
    if stale_done.exists():
        stale_done.unlink()
    filter_jsonl(logs_dir / "metrics.jsonl", lambda obj: int(obj.get("step", -1)) < resume_step)
    filter_jsonl(logs_dir / "epoch_metrics.jsonl", lambda obj: int(obj.get("epoch", 0)) <= resume_epoch)
    filter_jsonl(logs_dir / "eval_metrics.jsonl", lambda obj: int(obj.get("epoch", 0)) <= resume_epoch)
    for path in logs_dir.glob("router_stats_epoch*.json"):
        stem = path.stem.replace("router_stats_epoch", "")
        if stem.isdigit() and int(stem) > resume_epoch:
            path.unlink()


def get_lr_step(epoch: int, it: int, epoch_len: int, global_step: int, args) -> int:
    if args.lr_schedule_steps_per_epoch <= 0:
        return global_step
    progress = it / max(1, epoch_len)
    return int(round(epoch * args.lr_schedule_steps_per_epoch + progress * args.lr_schedule_steps_per_epoch))


@torch.no_grad()
def vae_encode_idxBl(vae, images: torch.Tensor, chunk_size: int):
    if chunk_size <= 0 or images.shape[0] <= chunk_size:
        return vae.img_to_idxBl(images)
    chunks = []
    for start in range(0, images.shape[0], chunk_size):
        end = min(start + chunk_size, images.shape[0])
        chunks.append(vae.img_to_idxBl(images[start:end]))
    num_stages = len(chunks[0])
    return [torch.cat([chunk[stage] for chunk in chunks], dim=0) for stage in range(num_stages)]


@torch.no_grad()
def autoregressive_sample_idxBl(var, labels: torch.Tensor, args, global_step: int) -> List[torch.Tensor]:
                                                                                    
    from models.helpers import sample_with_top_k_top_p_

    model = unwrap_model(var)
    device = labels.device
    B = int(labels.shape[0])
    seed = int(args.on_policy_seed_base + global_step * 1009 + dist.get_rank() * 9173)
    model.rng.manual_seed(seed)
    rng = model.rng

    label_B = labels.to(device=device, dtype=torch.long)
    sos = cond_BD = model.class_emb(
        torch.cat((label_B, torch.full_like(label_B, fill_value=model.num_classes)), dim=0)
    )
    lvl_pos = model.lvl_embed(model.lvl_1L) + model.pos_1LC
    next_token_map = (
        sos.unsqueeze(1).expand(2 * B, model.first_l, -1)
        + model.pos_start.expand(2 * B, model.first_l, -1)
        + lvl_pos[:, : model.first_l]
    )
    cur_L = 0
    f_hat = sos.new_zeros(B, model.Cvae, model.patch_nums[-1], model.patch_nums[-1])
    sampled_idx_Bl: List[torch.Tensor] = []

    was_training = model.training
    model.eval()
    for block in model.blocks:
        block.attn.kv_caching(True)
    try:
        for si, pn in enumerate(model.patch_nums):
            ratio = si / model.num_stages_minus_1
            cur_L += pn * pn
            cond_BD_or_gss = model.shared_ada_lin(cond_BD)
            x = next_token_map
            stage_ids = torch.full((2 * B, pn * pn), si, device=device, dtype=torch.long)
            branch_ids = torch.cat(
                (
                    torch.zeros(B, device=device, dtype=torch.long),
                    torch.ones(B, device=device, dtype=torch.long),
                ),
                dim=0,
            )
            model.set_moe_router_context(cond_BD=cond_BD, stage_ids=stage_ids, branch_ids=branch_ids)
            for block in model.blocks:
                x = block(x=x, cond_BD=cond_BD_or_gss, attn_bias=None)
            model.set_moe_router_context()

            logits_BlV = model.get_logits(x, cond_BD)
            t = float(args.on_policy_cfg) * ratio
            logits_BlV = (1 + t) * logits_BlV[:B] - t * logits_BlV[B:]
            idx_Bl = sample_with_top_k_top_p_(
                logits_BlV,
                rng=rng,
                top_k=int(args.on_policy_sample_top_k),
                top_p=float(args.on_policy_sample_top_p),
                num_samples=1,
            )[:, :, 0]
            sampled_idx_Bl.append(idx_Bl.detach())

            h_BChw = model.vae_quant_proxy[0].embedding(idx_Bl).transpose_(1, 2).reshape(B, model.Cvae, pn, pn)
            f_hat, next_token_map = model.vae_quant_proxy[0].get_next_autoregressive_input(
                si,
                len(model.patch_nums),
                f_hat,
                h_BChw,
            )
            if si != model.num_stages_minus_1:
                next_token_map = next_token_map.view(B, model.Cvae, -1).transpose(1, 2)
                next_token_map = model.word_embed(next_token_map) + lvl_pos[
                    :, cur_L : cur_L + model.patch_nums[si + 1] ** 2
                ]
                next_token_map = next_token_map.repeat(2, 1, 1)
    finally:
        model.set_moe_router_context()
        for block in model.blocks:
            block.attn.kv_caching(False)
        if was_training:
            model.train()

    return sampled_idx_Bl


def should_run_on_policy(args, epoch: int, global_step: int, micro_idx: int) -> bool:
    if (
        args.on_policy_erc_weight <= 0
        and args.on_policy_logit_kd_weight <= 0
        and args.on_policy_teacher_ce_weight <= 0
    ):
        return False
    if args.on_policy_interval_steps <= 0:
        return False
    if global_step < args.on_policy_start_step:
        return False
    if epoch < args.on_policy_start_epoch:
        return False
    if micro_idx != 0:
        return False
    return (global_step - args.on_policy_start_step) % args.on_policy_interval_steps == 0


def on_policy_weight_factor(args, global_step: int) -> float:
    ramp_steps = int(getattr(args, "on_policy_weight_ramp_steps", 0) or 0)
    if ramp_steps <= 0:
        return 1.0
    start_step = int(getattr(args, "on_policy_start_step", 0) or 0)
    return min(1.0, max(0.0, float(global_step - start_step + 1) / float(ramp_steps)))


def on_policy_recon_erc_loss(
    vae,
    teacher,
    student,
    labels: torch.Tensor,
    args,
    global_step: int,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    device = labels.device
    zero = torch.zeros((), device=device)
    stats = {
        "on_policy_erc": 0.0,
        "on_policy_layers": 0.0,
        "on_policy_recon_h": 0.0,
        "on_policy_recon_top1": 0.0,
        "on_policy_sel_mass": 0.0,
        "on_policy_sel_hit": 0.0,
        "on_policy_combo_rel": 0.0,
    }
    if labels.numel() <= 0:
        return zero, stats

    batch_size = min(int(args.on_policy_batch_size), int(labels.shape[0]))
    if batch_size <= 0:
        return zero, stats
    op_labels = labels[:batch_size].detach()

    sampled_idx_Bl = autoregressive_sample_idxBl(student, op_labels, args, global_step)
    with torch.no_grad():
        op_x_BLCv = vae.quantize.idxBl_to_var_input(sampled_idx_Bl).to(device=device).detach()
        pair_null = torch.full_like(op_labels, unwrap_model(student).num_classes)
        op_train_labels = torch.cat([op_labels, op_labels], dim=0)
        op_label_dropped = torch.cat([op_labels, pair_null], dim=0)
        op_train_x = torch.cat([op_x_BLCv, op_x_BLCv], dim=0)

    with torch.autocast("cuda", enabled=args.fp16 in {1, 2}, dtype=torch.bfloat16 if args.fp16 == 2 else torch.float16):
        student(
            op_train_labels,
            op_train_x,
            return_layers=[],
            return_hidden_states=False,
            label_B_dropped=op_label_dropped,
            return_pre_logits=True,
        )
        erc_args = argparse.Namespace(**vars(args))
        erc_args.selected_combo_kd_weight = 0.0
        erc_args.router_erc_weight = 1.0
        erc_args.router_erc_target_temperature = float(args.on_policy_erc_target_temperature)
        erc_args.expert_recon_layers = args.on_policy_erc_layers
        op_erc, _, op_stats = expert_recon_alignment_loss(student, teacher, erc_args)

    if op_stats["expert_recon_layers"] > 0:
        stats["on_policy_erc"] = float(op_erc.detach().item())
        stats["on_policy_layers"] = float(op_stats["expert_recon_layers"])
        stats["on_policy_recon_h"] = float(op_stats["recon_marginal_target_entropy"])
        stats["on_policy_recon_top1"] = float(op_stats["recon_marginal_target_top1_prob"])
        stats["on_policy_sel_mass"] = float(op_stats["recon_marginal_selected_target_mass"])
        stats["on_policy_sel_hit"] = float(op_stats["recon_marginal_selected_top1_hit"])
        stats["on_policy_combo_rel"] = float(op_stats["selected_combo_rel_l2"])
    return op_erc, stats


def chunked_on_policy_output_loss_from_pre_logits(
    student,
    teacher,
    student_hidden: torch.Tensor,
    student_cond: torch.Tensor,
    teacher_hidden: torch.Tensor,
    teacher_cond: torch.Tensor,
    args,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, L, _ = student_hidden.shape
    total_tokens = B * L
    chunk_tokens = int(args.logit_kd_chunk_tokens or 8192)
    chunk_tokens = max(1, chunk_tokens)
    ce_sum = torch.zeros((), device=student_hidden.device, dtype=torch.float32)
    kd_sum = torch.zeros((), device=student_hidden.device, dtype=torch.float32)
    kd_weight_sum = torch.zeros((), device=student_hidden.device, dtype=torch.float32)
    teacher_h_sum = torch.zeros((), device=student_hidden.device, dtype=torch.float32)
    teacher_top1_sum = torch.zeros((), device=student_hidden.device, dtype=torch.float32)
    token_weights = build_logit_kd_stage_weights(student, torch.Size((B, L)), args, student_hidden.device)
    w_flat = None
    if token_weights is not None:
        w_flat = token_weights.to(device=student_hidden.device, dtype=torch.float32).reshape(-1)
        w_flat = w_flat / w_flat.mean().clamp_min(1e-6)
    temp = max(float(args.logit_kd_temperature), 1e-6)

    for start in range(0, total_tokens, chunk_tokens):
        end = min(start + chunk_tokens, total_tokens)
        s_logits = _head_logits_flat_chunk(student, student_hidden, student_cond, start, end)
        with torch.no_grad():
            t_logits = _head_logits_flat_chunk(teacher, teacher_hidden, teacher_cond, start, end).float()
            hard_targets = t_logits.argmax(dim=-1)
            teacher_prob = F.softmax(t_logits, dim=-1)
            teacher_h_sum = teacher_h_sum + (-(teacher_prob * teacher_prob.clamp_min(1e-12).log()).sum(dim=-1)).sum()
            teacher_top1_sum = teacher_top1_sum + teacher_prob.max(dim=-1).values.sum()

        ce_sum = ce_sum + F.cross_entropy(s_logits.float(), hard_targets, reduction="sum")
        s = s_logits.float() / temp
        t = t_logits / temp
        teacher_logp = F.log_softmax(t, dim=-1)
        per_token = (teacher_logp.exp() * (teacher_logp - F.log_softmax(s, dim=-1))).sum(dim=-1)
        if w_flat is not None:
            weights = w_flat[start:end]
            kd_sum = kd_sum + (per_token * weights).sum()
            kd_weight_sum = kd_weight_sum + weights.sum()
        else:
            kd_sum = kd_sum + per_token.sum()
            kd_weight_sum = kd_weight_sum + per_token.new_tensor(float(per_token.numel()))

    denom = max(1, total_tokens)
    teacher_ce = ce_sum / denom
    logit_kd = (kd_sum / kd_weight_sum.clamp_min(1e-6)) * (temp * temp)
    teacher_h = teacher_h_sum / denom
    teacher_top1 = teacher_top1_sum / denom
    return teacher_ce, logit_kd, teacher_h.detach(), teacher_top1.detach()


def on_policy_output_loss(
    vae,
    teacher,
    student,
    labels: torch.Tensor,
    args,
    global_step: int,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, float]]:
    device = labels.device
    zero = torch.zeros((), device=device)
    stats = {
        "on_policy_teacher_ce": 0.0,
        "on_policy_logit_kd": 0.0,
        "on_policy_output_tokens": 0.0,
        "on_policy_teacher_h": 0.0,
        "on_policy_teacher_top1": 0.0,
    }
    if labels.numel() <= 0:
        return zero, zero, stats

    batch_size = min(int(args.on_policy_batch_size), int(labels.shape[0]))
    if batch_size <= 0:
        return zero, zero, stats
    op_labels = labels[:batch_size].detach()

    sampled_idx_Bl = autoregressive_sample_idxBl(student, op_labels, args, global_step)
    with torch.no_grad():
        op_x_BLCv = vae.quantize.idxBl_to_var_input(sampled_idx_Bl).to(device=device).detach()
        if args.train_with_cfg_pair:
            pair_null = torch.full_like(op_labels, unwrap_model(student).num_classes)
            op_train_labels = torch.cat([op_labels, op_labels], dim=0)
            op_label_dropped = torch.cat([op_labels, pair_null], dim=0)
            op_train_x = torch.cat([op_x_BLCv, op_x_BLCv], dim=0)
        else:
            op_train_labels = op_labels
            op_label_dropped = op_labels
            op_train_x = op_x_BLCv

    with torch.no_grad(), torch.autocast("cuda", enabled=args.fp16 in {1, 2}, dtype=torch.bfloat16 if args.fp16 == 2 else torch.float16):
        t_pre_logits, t_cond = teacher(
            op_train_labels,
            op_train_x,
            label_B_dropped=op_label_dropped,
            return_pre_logits=True,
        )

    with torch.autocast("cuda", enabled=args.fp16 in {1, 2}, dtype=torch.bfloat16 if args.fp16 == 2 else torch.float16):
        s_pre_logits, s_cond = student(
            op_train_labels,
            op_train_x,
            label_B_dropped=op_label_dropped,
            return_pre_logits=True,
        )
        teacher_ce, logit_kd, teacher_h, teacher_top1 = chunked_on_policy_output_loss_from_pre_logits(
            student,
            teacher,
            s_pre_logits,
            s_cond,
            t_pre_logits,
            t_cond,
            args,
        )

    stats["on_policy_teacher_ce"] = float(teacher_ce.detach().item())
    stats["on_policy_logit_kd"] = float(logit_kd.detach().item())
    stats["on_policy_output_tokens"] = float(s_pre_logits.shape[0] * s_pre_logits.shape[1])
    stats["on_policy_teacher_h"] = float(teacher_h.item())
    stats["on_policy_teacher_top1"] = float(teacher_top1.item())
    return teacher_ce, logit_kd, stats


def logit_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
    chunk_tokens: int = 0,
    token_weights: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    temp = max(float(temperature), 1e-6)
    vocab = student_logits.shape[-1]
    s_flat = student_logits.float().reshape(-1, vocab)
    t_flat = teacher_logits.float().reshape(-1, vocab)
    w_flat = None
    if token_weights is not None:
        w_flat = token_weights.to(device=student_logits.device, dtype=torch.float32).reshape(-1)
        w_flat = w_flat / w_flat.mean().clamp_min(1e-6)
    if chunk_tokens <= 0 or s_flat.shape[0] <= chunk_tokens:
        s = s_flat / temp
        t = t_flat / temp
        teacher_logp = F.log_softmax(t, dim=-1)
        per_token = (teacher_logp.exp() * (teacher_logp - F.log_softmax(s, dim=-1))).sum(dim=-1)
        if w_flat is not None:
            loss = (per_token * w_flat).sum() / w_flat.sum().clamp_min(1e-6)
        else:
            loss = per_token.mean()
        per_token = per_token.reshape(student_logits.shape[:-1])
        return loss * (temp * temp), per_token.detach()

    loss_sum = s_flat.new_zeros(())
    weight_sum = s_flat.new_zeros(())
    per_token_chunks = []
    for start in range(0, s_flat.shape[0], chunk_tokens):
        end = min(start + chunk_tokens, s_flat.shape[0])
        s = s_flat[start:end] / temp
        t = t_flat[start:end] / temp
        teacher_logp = F.log_softmax(t, dim=-1)
        per_token = (teacher_logp.exp() * (teacher_logp - F.log_softmax(s, dim=-1))).sum(dim=-1)
        if w_flat is not None:
            weights = w_flat[start:end]
            loss_sum = loss_sum + (per_token * weights).sum()
            weight_sum = weight_sum + weights.sum()
        else:
            loss_sum = loss_sum + per_token.sum()
            weight_sum = weight_sum + per_token.new_tensor(float(per_token.numel()))
        per_token_chunks.append(per_token.detach())
    per_token_all = torch.cat(per_token_chunks, dim=0).reshape(student_logits.shape[:-1])
    return (loss_sum / weight_sum.clamp_min(1e-6)) * (temp * temp), per_token_all


def build_logit_kd_stage_weights(model, logits_shape: torch.Size, args, device: torch.device) -> Optional[torch.Tensor]:
    if args.logit_kd_stage_weight_power <= 0:
        return None
    B, L = int(logits_shape[0]), int(logits_shape[1])
    var = unwrap_model(model)
    stage_ids = var.lvl_1L[:, :L].to(device=device, dtype=torch.float32)
    denom = max(float(len(PATCH_NUMS) - 1), 1.0)
    stage_pos = (stage_ids / denom).clamp(0, 1)
    min_w = max(0.0, float(args.logit_kd_stage_weight_min))
    weights = min_w + (1.0 - min_w) * stage_pos.pow(float(args.logit_kd_stage_weight_power))
    weights = weights / weights.mean().clamp_min(1e-6)
    return weights.expand(B, -1)


def _head_logits_flat_chunk(model, hidden_BLC: torch.Tensor, cond_BD: torch.Tensor, start: int, end: int) -> torch.Tensor:
    var = unwrap_model(model)
    B, L, C = hidden_BLC.shape
    flat_h = hidden_BLC.reshape(B * L, C)
    h = flat_h[start:end]
    token_ids = torch.arange(start, end, device=hidden_BLC.device, dtype=torch.long)
    batch_ids = torch.div(token_ids, L, rounding_mode="floor")

    scale, shift = var.head_nm.ada_lin(cond_BD).view(B, 1, 2, C).unbind(2)
    scale = scale.squeeze(1).index_select(0, batch_ids)
    shift = shift.squeeze(1).index_select(0, batch_ids)
    h = F.layer_norm(h, (C,), None, None, var.head_nm.ln_wo_grad.eps)
    h = h.mul(scale.add(1)).add_(shift)
    return var.head(h)


def chunked_ce_and_logit_kd_from_pre_logits(
    student,
    teacher,
    student_hidden: torch.Tensor,
    student_cond: torch.Tensor,
    targets: torch.Tensor,
    args,
    teacher_hidden: Optional[torch.Tensor] = None,
    teacher_cond: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, L, _ = student_hidden.shape
    total_tokens = B * L
    chunk_tokens = int(args.logit_kd_chunk_tokens or 8192)
    chunk_tokens = max(1, chunk_tokens)
    flat_targets = targets.reshape(-1)
    ce_sum = torch.zeros((), device=student_hidden.device, dtype=torch.float32)
    kd_sum = torch.zeros((), device=student_hidden.device, dtype=torch.float32)
    kd_weight_sum = torch.zeros((), device=student_hidden.device, dtype=torch.float32)
    need_logit_kd = (
        args.logit_kd_weight > 0
        and teacher_hidden is not None
        and teacher_cond is not None
    )
    token_weights = build_logit_kd_stage_weights(student, torch.Size((B, L)), args, student_hidden.device)
    w_flat = None
    if token_weights is not None:
        w_flat = token_weights.to(device=student_hidden.device, dtype=torch.float32).reshape(-1)
        w_flat = w_flat / w_flat.mean().clamp_min(1e-6)
    temp = max(float(args.logit_kd_temperature), 1e-6)

    for start in range(0, total_tokens, chunk_tokens):
        end = min(start + chunk_tokens, total_tokens)
        s_logits = _head_logits_flat_chunk(student, student_hidden, student_cond, start, end)
        ce_sum = ce_sum + F.cross_entropy(s_logits.float(), flat_targets[start:end], reduction="sum")

        if need_logit_kd:
            with torch.no_grad():
                t_logits = _head_logits_flat_chunk(teacher, teacher_hidden, teacher_cond, start, end)
            s = s_logits.float() / temp
            t = t_logits.float() / temp
            teacher_logp = F.log_softmax(t, dim=-1)
            per_token = (teacher_logp.exp() * (teacher_logp - F.log_softmax(s, dim=-1))).sum(dim=-1)
            if w_flat is not None:
                weights = w_flat[start:end]
                kd_sum = kd_sum + (per_token * weights).sum()
                kd_weight_sum = kd_weight_sum + weights.sum()
            else:
                kd_sum = kd_sum + per_token.sum()
                kd_weight_sum = kd_weight_sum + per_token.new_tensor(float(per_token.numel()))
    ce = ce_sum / max(1, total_tokens)
    if need_logit_kd:
        logit_kd = (kd_sum / kd_weight_sum.clamp_min(1e-6)) * (temp * temp)
    else:
        logit_kd = ce.new_zeros(())
    return ce, logit_kd


def hidden_state_kd_loss(student_hidden: Sequence[torch.Tensor], teacher_hidden: Sequence[torch.Tensor]) -> torch.Tensor:
    terms = []
    for sh, th in zip(student_hidden, teacher_hidden):
        diff = sh - th.to(device=sh.device, dtype=sh.dtype)
        terms.append(diff.mul(diff).mean().float())
    if not terms:
        raise ValueError("hidden_state_kd_loss requires at least one hidden-state pair.")
    return torch.stack(terms).mean()


def train_one_epoch(epoch, vae, teacher, student, loader, optimizer, scaler, args, meta, total_steps, global_step, stop_step, log_path):
    student.train()
    teacher.eval()
    device = dist.get_device()
    schedule = getattr(args, "_parsed_topk_schedule", None)
    epoch_topk = int(schedule[epoch] if schedule is not None and epoch < len(schedule) else args.train_topk)
    set_model_runtime_topk(student, epoch_topk)
    ce_losses, kd_losses, logit_kd_losses = [], [], []
    erc_losses, combo_kd_losses, total_losses = [], [], []
    router_accum = init_router_stats_accum(student)
    router_stats_interval = args.router_stats_interval if args.router_stats_interval > 0 else args.log_interval
    accum_steps = max(1, int(args.grad_accum_steps))
    epoch_len = len(loader)
    optimizer.zero_grad(set_to_none=True)
    for it, (images, labels) in enumerate(loader):
        if stop_step is not None and global_step >= stop_step:
            break
        micro_idx = it % accum_steps
        accum_start = it - micro_idx
        accum_end = min(accum_start + accum_steps, epoch_len)
        current_accum_steps = max(1, accum_end - accum_start)
        stepping = (it + 1) == accum_end
        sync_context = (
            student.no_sync()
            if isinstance(student, DDP) and not stepping
            else nullcontext()
        )
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.no_grad():
            gt_idx_Bl = vae_encode_idxBl(vae, images, args.vae_encode_batch_size)
            gt_BL = torch.cat(gt_idx_Bl, dim=1)
            x_BLCv = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
            if args.train_with_cfg_pair:
                pair_null = torch.full_like(labels, unwrap_model(student).num_classes)
                train_labels = torch.cat([labels, labels], dim=0)
                train_x_BLCv = torch.cat([x_BLCv, x_BLCv], dim=0)
                train_gt_BL = torch.cat([gt_BL, gt_BL], dim=0)
                label_dropped = torch.cat([labels, pair_null], dim=0)
            else:
                train_labels = labels
                train_x_BLCv = x_BLCv
                train_gt_BL = gt_BL
                label_dropped = torch.where(
                    torch.rand(labels.shape[0], device=device) < unwrap_model(student).cond_drop_rate,
                    unwrap_model(student).num_classes,
                    labels,
                )
            need_teacher_logits = args.logit_kd_weight > 0
            if args.kd_weight > 0 or need_teacher_logits:
                with torch.autocast("cuda", enabled=args.fp16 in {1, 2}, dtype=torch.bfloat16 if args.fp16 == 2 else torch.float16):
                    t_pre_logits, t_hidden, t_cond = teacher(
                        train_labels,
                        train_x_BLCv,
                        return_layers=args.kd_layers,
                        return_hidden_states=True,
                        label_B_dropped=label_dropped,
                        return_pre_logits=True,
                    )
            else:
                t_pre_logits = None
                t_cond = None
                t_hidden = []

        lr_step = get_lr_step(epoch, it, len(loader), global_step, args)
        factor = lr_factor(lr_step, total_steps, args.warmup_steps, args.min_lr_factor)
        for group in optimizer.param_groups:
            group["lr"] = args.lr * factor

        with sync_context, torch.autocast("cuda", enabled=args.fp16 in {1, 2}, dtype=torch.bfloat16 if args.fp16 == 2 else torch.float16):
            s_pre_logits, s_hidden, s_cond = student(
                train_labels,
                train_x_BLCv,
                return_layers=args.kd_layers,
                return_hidden_states=True,
                label_B_dropped=label_dropped,
                return_pre_logits=True,
            )
            ce, logit_kd = chunked_ce_and_logit_kd_from_pre_logits(
                student,
                teacher,
                s_pre_logits,
                s_cond,
                train_gt_BL,
                args,
                teacher_hidden=t_pre_logits,
                teacher_cond=t_cond,
            )
            if args.kd_weight > 0 and s_hidden:
                kd = hidden_state_kd_loss(s_hidden, t_hidden)
            else:
                kd = ce.new_zeros(())
            router_loss, router_loss_stats = router_regularization(student, args)
            router_erc, selected_combo_kd, expert_kd_stats = expert_recon_alignment_loss(student, teacher, args)
            if (
                (args.router_erc_weight > 0 or args.selected_combo_kd_weight > 0)
                and expert_kd_stats["expert_recon_layers"] <= 0
            ):
                raise RuntimeError(
                    "Expert reconstruction alignment is enabled but no MoE layer produced router probabilities "
                    "and cached input samples. Check --hard_mode and --router_capture_input_sample_tokens."
                )
            on_policy_erc = ce.new_zeros(())
            on_policy_stats = {
                "on_policy_weight_factor": 0.0,
                "on_policy_erc": 0.0,
                "on_policy_layers": 0.0,
                "on_policy_recon_h": 0.0,
                "on_policy_recon_top1": 0.0,
                "on_policy_sel_mass": 0.0,
                "on_policy_sel_hit": 0.0,
                "on_policy_combo_rel": 0.0,
                "on_policy_teacher_ce": 0.0,
                "on_policy_logit_kd": 0.0,
                "on_policy_output_tokens": 0.0,
                "on_policy_teacher_h": 0.0,
                "on_policy_teacher_top1": 0.0,
            }
            run_on_policy = should_run_on_policy(args, epoch, global_step, micro_idx)
                                                                               
                                                               
            if run_on_policy and isinstance(student, DDP) and stepping:
                run_on_policy = False
            if run_on_policy:
                on_policy_stats["on_policy_weight_factor"] = on_policy_weight_factor(args, global_step)
                if args.on_policy_erc_weight > 0:
                    on_policy_erc, recon_on_policy_stats = on_policy_recon_erc_loss(
                        vae,
                        teacher,
                        student,
                        labels,
                        args,
                        global_step,
                    )
                    on_policy_stats.update(recon_on_policy_stats)
                if args.on_policy_logit_kd_weight > 0 or args.on_policy_teacher_ce_weight > 0:
                    on_policy_teacher_ce, on_policy_logit_kd, output_on_policy_stats = on_policy_output_loss(
                        vae,
                        teacher,
                        student,
                        labels,
                        args,
                        global_step,
                    )
                    on_policy_stats.update(output_on_policy_stats)
                else:
                    on_policy_teacher_ce = ce.new_zeros(())
                    on_policy_logit_kd = ce.new_zeros(())
                if args.on_policy_erc_weight > 0 and on_policy_stats["on_policy_layers"] <= 0:
                    raise RuntimeError(
                        "On-policy ERC was triggered but no MoE layer produced router probabilities "
                        "and cached input samples. Check --router_capture_input_sample_tokens and "
                        "--expert_recon_sample_tokens."
                    )
            else:
                on_policy_teacher_ce = ce.new_zeros(())
                on_policy_logit_kd = ce.new_zeros(())
            on_policy_loss = on_policy_erc * float(current_accum_steps)
            on_policy_teacher_ce_loss = on_policy_teacher_ce * float(current_accum_steps)
            on_policy_logit_kd_loss = on_policy_logit_kd * float(current_accum_steps)
            op_weight_factor = float(on_policy_stats["on_policy_weight_factor"])
            loss = (
                args.ce_weight * ce
                + args.kd_weight * kd
                + args.logit_kd_weight * logit_kd
                + args.router_erc_weight * router_erc
                + args.selected_combo_kd_weight * selected_combo_kd
                + op_weight_factor * args.on_policy_erc_weight * on_policy_loss
                + op_weight_factor * args.on_policy_teacher_ce_weight * on_policy_teacher_ce_loss
                + op_weight_factor * args.on_policy_logit_kd_weight * on_policy_logit_kd_loss
                + router_loss
            )
            loss_for_backward = loss / float(current_accum_steps)

        if scaler is not None:
            scaler.scale(loss_for_backward).backward()
            if stepping:
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        else:
            loss_for_backward.backward()
            if stepping:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(student.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        if stepping:
            dynamic_bias_stats = update_router_dynamic_bias(student, args, global_step)
        else:
            dynamic_bias_stats = {
                "router_dynamic_bias_factor": 0.0,
                "router_dynamic_bias_mean_abs": 0.0,
                "router_dynamic_stage_bias_mean_abs": 0.0,
                "router_dynamic_bias_updates": 0.0,
            }

        ce_losses.append(float(ce.detach().item()))
        kd_losses.append(float(kd.detach().item()))
        logit_kd_losses.append(float(logit_kd.detach().item()))
        erc_losses.append(float(router_erc.detach().item()))
        combo_kd_losses.append(float(selected_combo_kd.detach().item()))
        total_losses.append(float(loss.detach().item()))
        should_sample_router = (
            router_stats_interval > 0
            and (it % router_stats_interval == 0 or it == len(loader) - 1)
        )
        if should_sample_router:
            accumulate_router_stats(student, router_accum)
        if it % args.log_interval == 0 or it == len(loader) - 1:
            stats = {
                "epoch": epoch,
                "iter": it,
                "step": global_step,
                "micro_step": micro_idx,
                "grad_accum_steps": accum_steps,
                "current_accum_steps": current_accum_steps,
                "optimizer_stepping": stepping,
                "lr_step": lr_step,
                "lr": args.lr * factor,
                "runtime_topk": epoch_topk,
                "ce": float(ce.detach().item()),
                "kd": float(kd.detach().item()),
                "logit_kd": float(logit_kd.detach().item()),
                "router_erc": float(router_erc.detach().item()),
                "selected_combo_kd": float(selected_combo_kd.detach().item()),
                "on_policy_erc": float(on_policy_erc.detach().item()),
                "on_policy_teacher_ce": float(on_policy_teacher_ce.detach().item()),
                "on_policy_logit_kd": float(on_policy_logit_kd.detach().item()),
                "on_policy_triggered": bool(
                    on_policy_stats["on_policy_layers"] > 0
                    or on_policy_stats["on_policy_output_tokens"] > 0
                ),
                "loss": float(loss.detach().item()),
                **router_loss_stats,
                **expert_kd_stats,
                **on_policy_stats,
                **dynamic_bias_stats,
            }
            reduced = torch.tensor(
                [
                    stats["ce"],
                    stats["kd"],
                    stats["logit_kd"],
                    stats["router_erc"],
                    stats["selected_combo_kd"],
                    stats["loss"],
                    stats["router_erc_target_entropy"],
                    stats["expert_recon_layers"],
                    stats["expert_recon_tokens"],
                    stats["recon_marginal_target_entropy"],
                    stats["recon_marginal_target_top1_prob"],
                    stats["recon_marginal_score_cv"],
                    stats["recon_marginal_selected_target_mass"],
                    stats["recon_marginal_selected_top1_hit"],
                    stats["selected_combo_rel_l2"],
                    stats["on_policy_erc"],
                    stats["on_policy_layers"],
                    stats["on_policy_recon_h"],
                    stats["on_policy_recon_top1"],
                    stats["on_policy_sel_mass"],
                    stats["on_policy_sel_hit"],
                    stats["on_policy_combo_rel"],
                    stats["on_policy_teacher_ce"],
                    stats["on_policy_logit_kd"],
                    stats["on_policy_output_tokens"],
                    stats["on_policy_teacher_h"],
                    stats["on_policy_teacher_top1"],
                ],
                device=device,
            )
            all_reduce_mean(reduced)
            stats["ce_mean_all_ranks"] = float(reduced[0].item())
            stats["kd_mean_all_ranks"] = float(reduced[1].item())
            stats["logit_kd_mean_all_ranks"] = float(reduced[2].item())
            stats["router_erc_mean_all_ranks"] = float(reduced[3].item())
            stats["selected_combo_kd_mean_all_ranks"] = float(reduced[4].item())
            stats["loss_mean_all_ranks"] = float(reduced[5].item())
            stats["router_erc_target_entropy_mean_all_ranks"] = float(reduced[6].item())
            stats["expert_recon_layers_mean_all_ranks"] = float(reduced[7].item())
            stats["expert_recon_tokens_mean_all_ranks"] = float(reduced[8].item())
            stats["recon_marginal_target_entropy_mean_all_ranks"] = float(reduced[9].item())
            stats["recon_marginal_target_top1_prob_mean_all_ranks"] = float(reduced[10].item())
            stats["recon_marginal_score_cv_mean_all_ranks"] = float(reduced[11].item())
            stats["recon_marginal_selected_target_mass_mean_all_ranks"] = float(reduced[12].item())
            stats["recon_marginal_selected_top1_hit_mean_all_ranks"] = float(reduced[13].item())
            stats["selected_combo_rel_l2_mean_all_ranks"] = float(reduced[14].item())
            stats["on_policy_erc_mean_all_ranks"] = float(reduced[15].item())
            stats["on_policy_layers_mean_all_ranks"] = float(reduced[16].item())
            stats["on_policy_recon_h_mean_all_ranks"] = float(reduced[17].item())
            stats["on_policy_recon_top1_mean_all_ranks"] = float(reduced[18].item())
            stats["on_policy_sel_mass_mean_all_ranks"] = float(reduced[19].item())
            stats["on_policy_sel_hit_mean_all_ranks"] = float(reduced[20].item())
            stats["on_policy_combo_rel_mean_all_ranks"] = float(reduced[21].item())
            stats["on_policy_teacher_ce_mean_all_ranks"] = float(reduced[22].item())
            stats["on_policy_logit_kd_mean_all_ranks"] = float(reduced[23].item())
            stats["on_policy_output_tokens_mean_all_ranks"] = float(reduced[24].item())
            stats["on_policy_teacher_h_mean_all_ranks"] = float(reduced[25].item())
            stats["on_policy_teacher_top1_mean_all_ranks"] = float(reduced[26].item())
            if dist.is_master():
                append_jsonl(log_path, stats)
                print(
                    f"[ep {epoch} it {it}/{len(loader)} step {global_step}] "
                    f"lr={stats['lr']:.3e} ce={stats['ce_mean_all_ranks']:.4f} "
                    f"kd={stats['kd_mean_all_ranks']:.4f} "
                    f"lkd={stats['logit_kd_mean_all_ranks']:.4f} "
                    f"erc={stats['router_erc_mean_all_ranks']:.4f} "
                    f"combo={stats['selected_combo_kd_mean_all_ranks']:.4f} "
                    f"comboRel={stats['selected_combo_rel_l2_mean_all_ranks']:.3f} "
                    f"ercH={stats['router_erc_target_entropy_mean_all_ranks']:.3f} "
                    f"reconH={stats['recon_marginal_target_entropy_mean_all_ranks']:.3f} "
                    f"reconTop1={stats['recon_marginal_target_top1_prob_mean_all_ranks']:.3f} "
                    f"selMass={stats['recon_marginal_selected_target_mass_mean_all_ranks']:.3f} "
                    f"selHit={stats['recon_marginal_selected_top1_hit_mean_all_ranks']:.3f} "
                    f"opW={stats['on_policy_weight_factor']:.2f} "
                    f"opErc={stats['on_policy_erc_mean_all_ranks']:.4f} "
                    f"opLkd={stats['on_policy_logit_kd_mean_all_ranks']:.4f} "
                    f"opTCe={stats['on_policy_teacher_ce_mean_all_ranks']:.4f} "
                    f"opLayers={stats['on_policy_layers_mean_all_ranks']:.1f} "
                    f"opMass={stats['on_policy_sel_mass_mean_all_ranks']:.3f} "
                    f"opHit={stats['on_policy_sel_hit_mean_all_ranks']:.3f} "
                    f"loss={stats['loss_mean_all_ranks']:.4f} "
                    f"dynB={stats['router_dynamic_bias_mean_abs']:.3f}/"
                    f"{stats['router_dynamic_stage_bias_mean_abs']:.3f} "
                    f"k={stats['runtime_topk']} "
                    f"topkH={stats['router_topk_entropy_ratio']:.3f} "
                    f"topkMax={stats['router_topk_max_freq']:.3f} "
                    f"ctxA={stats['router_context_alpha']:.3f} "
                    f"ctxS={stats['router_context_stage_embed_norm']:.3f} "
                    f"ctxC={stats['router_context_cond_proj_norm']:.3f}",
                    flush=True,
                )
        if stepping:
            global_step += 1
    epoch_stats = {
        "epoch": epoch,
        "ce": float(np.mean(ce_losses)) if ce_losses else None,
        "kd": float(np.mean(kd_losses)) if kd_losses else None,
        "logit_kd": float(np.mean(logit_kd_losses)) if logit_kd_losses else None,
        "router_erc": float(np.mean(erc_losses)) if erc_losses else None,
        "selected_combo_kd": float(np.mean(combo_kd_losses)) if combo_kd_losses else None,
        "loss": float(np.mean(total_losses)) if total_losses else None,
        "steps": len(total_losses),
    }
    router_stats = finalize_router_stats(student, router_accum)
    if not router_stats:
        router_stats = collect_router_stats(student)
    return global_step, epoch_stats, router_stats


@torch.no_grad()
def evaluate(vae, model, loader, args) -> dict:
    model.eval()
    device = dist.get_device()
    losses, accs = [], []
    for it, (images, labels) in enumerate(loader):
        if args.eval_batches > 0 and it >= args.eval_batches:
            break
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        gt_idx_Bl = vae_encode_idxBl(vae, images, args.vae_encode_batch_size)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        x_BLCv = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
        with torch.autocast("cuda", enabled=args.fp16 in {1, 2}, dtype=torch.bfloat16 if args.fp16 == 2 else torch.float16):
            pre_logits, cond = model(labels, x_BLCv, label_B_dropped=labels, return_pre_logits=True)
            B, L, _ = pre_logits.shape
            total_tokens = B * L
            flat_targets = gt_BL.reshape(-1)
            chunk_tokens = max(1, int(args.logit_kd_chunk_tokens or 8192))
            loss_sum = torch.zeros((), device=device, dtype=torch.float32)
            correct_sum = torch.zeros((), device=device, dtype=torch.float32)
            for start in range(0, total_tokens, chunk_tokens):
                end = min(start + chunk_tokens, total_tokens)
                logits = _head_logits_flat_chunk(model, pre_logits, cond, start, end)
                targets = flat_targets[start:end]
                loss_sum = loss_sum + F.cross_entropy(logits.float(), targets, reduction="sum")
                correct_sum = correct_sum + (logits.argmax(dim=-1) == targets).float().sum()
        loss = loss_sum / max(1, total_tokens)
        acc = correct_sum / max(1, total_tokens) * 100.0
        losses.append(float(loss.item()))
        accs.append(float(acc.item()))
    t = torch.tensor([
        np.mean(losses) if losses else 0.0,
        np.mean(accs) if accs else 0.0,
    ], device=device)
    all_reduce_mean(t)
    return {"val_loss": float(t[0].item()), "val_acc": float(t[1].item()), "batches": len(losses)}


def main():
    parser = argparse.ArgumentParser("Finetune VAR Prism-MoE initialized from improved dense-to-MoE")
    parser.add_argument("--data_path", type=str, default="/liying06/liying/datasets/imagenet")
    parser.add_argument("--vae_ckpt", type=str, default="/liying06/liying/pretrained/var_model_zoo/vae_ch160v4096z32.pth")
    parser.add_argument("--dense_ckpt", type=str, required=True)
    parser.add_argument("--moe_init_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--var_root", type=str, default=None)
    parser.add_argument("--depth", type=int, default=16, choices=[16, 20, 24, 30])
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--shared_aln", action="store_true")
    parser.add_argument("--fuse", action="store_true")
    parser.add_argument("--train_topk", type=int, default=2)
    parser.add_argument(
        "--router_topk_schedule",
        type=str,
        default="",
        help="Epoch-level runtime top-k schedule, e.g. '8x2,4x2,2x4'. Final checkpoint is restored to --train_topk.",
    )
    parser.add_argument("--hard_mode", action="store_true")
    parser.add_argument("--norm_topk_prob", action="store_true", default=True)
    parser.add_argument("--router_temp", type=float, default=1.0)
    parser.add_argument("--router_init_alpha", type=float, default=0.1)
    parser.add_argument("--router_delta_hidden_mult", type=float, default=0.25)
    parser.add_argument(
        "--router_context_mode",
        type=str,
        default="none",
        choices=["none", "cond", "cond_stage", "cond_stage_branch"],
        help="Optional router logit delta from class condition, stage id, and CFG branch.",
    )
    parser.add_argument("--router_context_init_alpha", type=float, default=0.1)
    parser.add_argument("--router_context_interaction_rank", type=int, default=0)
    parser.add_argument("--router_context_interaction_init_alpha", type=float, default=0.1)
    parser.add_argument("--router_context_cosine", action="store_true")
    parser.add_argument("--router_context_cosine_init_alpha", type=float, default=0.0)
    parser.add_argument(
        "--router_token_cosine",
        action="store_true",
        help="Add token-to-expert-prototype cosine logits to the router.",
    )
    parser.add_argument("--router_token_cosine_init_alpha", type=float, default=0.0)
    parser.add_argument(
        "--router_logit_mode",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="Router logit parameterization. 'cosine' uses normalized token/router vectors with bounded delta.",
    )
    parser.add_argument("--router_cosine_tau", type=float, default=10.0)
    parser.add_argument(
        "--router_capture_input_sample_tokens",
        type=int,
        default=0,
        help="Per-layer token samples cached for reconstruction-alignment router losses.",
    )
    parser.add_argument(
        "--preserve_router_delta_from_init",
        action="store_true",
        help="Keep trainable router delta/alpha tensors from moe_init_ckpt instead of resetting them.",
    )
    parser.add_argument("--train_scope", type=str, default="router_experts", choices=["router", "router_experts", "all_moe", "full"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32, help="Per-GPU batch size.")
    parser.add_argument("--target_global_batch", type=int, default=0, help="Minimum effective image batch across all GPUs; derives grad_accum_steps when set.")
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Optimizer gradient accumulation steps.")
    parser.add_argument("--vae_encode_batch_size", type=int, default=0, help="Optional per-GPU micro-batch size for VAE tokenization.")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--min_lr_factor", type=float, default=0.05)
    parser.add_argument("--ce_weight", type=float, default=1.0)
    parser.add_argument("--kd_weight", type=float, default=0.25)
    parser.add_argument("--kd_layers", type=int, nargs="*", default=None)
    parser.add_argument("--logit_kd_weight", type=float, default=0.0)
    parser.add_argument("--logit_kd_temperature", type=float, default=2.0)
    parser.add_argument("--logit_kd_chunk_tokens", type=int, default=0)
    parser.add_argument("--logit_kd_stage_weight_power", type=float, default=0.0)
    parser.add_argument("--logit_kd_stage_weight_min", type=float, default=1.0)
    parser.add_argument(
        "--selected_combo_kd_weight",
        type=float,
        default=0.0,
        help=(
            "KD weight for the actually selected top-k normal-expert mixture to reconstruct "
            "dense FFN minus the current student shared path."
        ),
    )
    parser.add_argument(
        "--expert_recon_sample_tokens",
        type=int,
        default=0,
        help="Per-layer token samples used by selected-combo KD and reconstruction-marginal ERC.",
    )
    parser.add_argument("--router_erc_weight", type=float, default=0.0)
    parser.add_argument(
        "--router_erc_target_temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature applied to ERC target logits before softmax. Values >1 soften the target; "
            "1.0 preserves the v11a/v11b target exactly."
        ),
    )
    parser.add_argument(
        "--on_policy_erc_weight",
        type=float,
        default=0.0,
        help=(
            "Extra reconstruction-marginal ERC weight on student-generated token prefixes. "
            "Default 0 disables the on-policy path."
        ),
    )
    parser.add_argument(
        "--on_policy_logit_kd_weight",
        type=float,
        default=0.0,
        help="Extra dense-teacher logit KD weight on student-generated token prefixes.",
    )
    parser.add_argument(
        "--on_policy_teacher_ce_weight",
        type=float,
        default=0.0,
        help="Optional hard-label CE weight using dense-teacher argmax targets on generated token prefixes.",
    )
    parser.add_argument("--on_policy_batch_size", type=int, default=2)
    parser.add_argument("--on_policy_interval_steps", type=int, default=0)
    parser.add_argument("--on_policy_start_step", type=int, default=0)
    parser.add_argument("--on_policy_start_epoch", type=int, default=0)
    parser.add_argument(
        "--on_policy_weight_ramp_steps",
        type=int,
        default=0,
        help="Linearly ramp all on-policy loss weights over this many optimizer steps after on_policy_start_step.",
    )
    parser.add_argument("--on_policy_cfg", type=float, default=4.0)
    parser.add_argument("--on_policy_sample_top_k", type=int, default=900)
    parser.add_argument("--on_policy_sample_top_p", type=float, default=0.96)
    parser.add_argument("--on_policy_seed_base", type=int, default=12345)
    parser.add_argument(
        "--on_policy_erc_target_temperature",
        type=float,
        default=1.0,
        help="Temperature for reconstruction-marginal ERC targets on generated-prefix batches.",
    )
    parser.add_argument(
        "--on_policy_erc_layers",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional 0-indexed block list for on-policy reconstruction-marginal ERC. "
            "Teacher-forced combo/recon ERC remains full-depth."
        ),
    )
    parser.add_argument(
        "--train_with_cfg_pair",
        action="store_true",
        help="Run each teacher-forced batch as paired conditional/unconditional CFG branches.",
    )
    parser.add_argument("--router_balance_weight", type=float, default=0.01)
    parser.add_argument("--router_dynamic_bias_weight", type=float, default=0.0)
    parser.add_argument("--router_dynamic_stage_bias_weight", type=float, default=0.0)
    parser.add_argument("--router_dynamic_bias_clip", type=float, default=2.0)
    parser.add_argument("--router_dynamic_bias_max_delta", type=float, default=0.10)
    parser.add_argument("--router_dynamic_bias_warmup_steps", type=int, default=0)
    parser.add_argument("--router_dynamic_bias_ramp_steps", type=int, default=0)
    parser.add_argument("--router_dynamic_stage_bias_min_slots", type=int, default=64)
    parser.add_argument("--router_near_zero_threshold", type=float, default=1e-4)
    parser.add_argument("--router_z_weight", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=2.0)
    parser.add_argument("--fp16", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--fused_adamw", action="store_true")
    parser.add_argument("--ddp_find_unused_parameters", action="store_true", help="Enable DDP unused-parameter detection for sparse-routing objectives that do not touch every expert each step.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--router_stats_interval", type=int, default=100)
    parser.add_argument("--save_every_epoch", type=int, default=1)
    parser.add_argument("--eval_every_epoch", type=int, default=1)
    parser.add_argument("--eval_batches", type=int, default=100)
    parser.add_argument("--max_train_steps", type=int, default=0, help="Smoke-test limit; 0 means no limit.")
    parser.add_argument("--resume_ckpt", type=str, default=None, help="Resume a finetune checkpoint saved by this script.")
    parser.add_argument("--resume_no_optimizer", action="store_true", help="Resume weights only, with a freshly initialized optimizer.")
    parser.add_argument("--resume_allow_missing_router", action="store_true", help="Allow newly added router tensors to be initialized when resuming an older checkpoint.")
    parser.add_argument("--prune_resume_logs", action="store_true", help="Drop partial logs beyond the resumed checkpoint epoch/step.")
    parser.add_argument("--max_extra_steps", type=int, default=0, help="Stop after this many additional optimizer steps after resume/start; 0 means no limit.")
    parser.add_argument("--lr_schedule_steps_per_epoch", type=int, default=0, help="Use this many logical steps per epoch for LR scheduling; inferred on resume if omitted.")
    parser.add_argument("--lr_schedule_total_steps", type=int, default=0, help="Override total logical steps used by the LR scheduler without changing training epochs.")
    args = parser.parse_args()
    if args.hard_mode and (
        args.router_erc_weight > 0
        or args.selected_combo_kd_weight > 0
        or args.on_policy_erc_weight > 0
    ):
        raise ValueError("--hard_mode disables router probabilities, so it cannot be used with combo KD or ERC.")
    if args.grad_accum_steps <= 0:
        raise ValueError("--grad_accum_steps must be positive.")
    if (
        args.router_erc_weight > 0
        or args.selected_combo_kd_weight > 0
        or args.on_policy_erc_weight > 0
    ) and args.router_capture_input_sample_tokens <= 0:
        args.router_capture_input_sample_tokens = int(args.expert_recon_sample_tokens or 128)
    if args.expert_recon_sample_tokens <= 0:
        args.expert_recon_sample_tokens = int(args.router_capture_input_sample_tokens)
    args._parsed_topk_schedule = parse_topk_schedule(args.router_topk_schedule, args.epochs, args.train_topk)
    if args._parsed_topk_schedule[-1] != args.train_topk:
        raise ValueError(
            f"router_topk_schedule must end at train_topk={args.train_topk}; got {args._parsed_topk_schedule[-1]}"
        )
    if args.on_policy_erc_layers is not None:
        layers = sorted({int(v) for v in args.on_policy_erc_layers})
        if any(v < 0 or v >= args.depth for v in layers):
            raise ValueError(f"--on_policy_erc_layers must be in [0, {args.depth}); got {args.on_policy_erc_layers}")
        args.on_policy_erc_layers = layers

    launched = "RANK" in os.environ
    if launched:
        dist.initialize(timeout=60)
    else:
        dist.initialize(gpu_id_if_not_distibuted=0)
    device = dist.get_device()
    if args.target_global_batch > 0:
        micro_global_batch = int(args.batch_size) * int(dist.get_world_size())
        args.grad_accum_steps = max(1, math.ceil(int(args.target_global_batch) / max(1, micro_global_batch)))
    args.effective_global_batch = int(args.batch_size) * int(dist.get_world_size()) * int(args.grad_accum_steps)
    if (
        launched
        and (
            args.on_policy_erc_weight > 0
            or args.on_policy_logit_kd_weight > 0
            or args.on_policy_teacher_ce_weight > 0
        )
        and args.grad_accum_steps <= 1
    ):
        raise ValueError(
            "On-policy losses perform a second DDP forward before the backward pass. "
            "Use gradient accumulation >1 so the on-policy forward runs under no_sync(); "
            "increase --target_global_batch or --grad_accum_steps."
        )
    seed_everything(args.seed + dist.get_rank())
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    out_dir = Path(args.output_dir)
    if dist.is_master():
        out_dir.mkdir(parents=True, exist_ok=True)
        write_json(out_dir / "config.json", vars(args))

    vae, teacher, student, meta = build_models(args, device)
    set_trainable(student, args.train_scope)
    apply_trainable_config_overrides(student, args)
    if args.kd_layers is None:
        args.kd_layers = [i for i in range(3, args.depth, 4)]
    if dist.is_master():
        write_json(out_dir / "moe_meta.json", meta)
        trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
        total = sum(p.numel() for p in student.parameters())
        print(f"Trainable params: {trainable:,}/{total:,} ({trainable / total * 100:.2f}%)")
        print(f"KD layers: {args.kd_layers}")
        print(f"Runtime top-k schedule: {args._parsed_topk_schedule}")

    if launched:
        student = DDP(
            student,
            device_ids=[dist.get_local_rank()],
            find_unused_parameters=bool(args.ddp_find_unused_parameters),
            broadcast_buffers=False,
        )

    _, train_set, val_set = build_dataset(args.data_path, final_reso=256, hflip=False, mid_reso=1.125)
    train_sampler = DistributedSampler(train_set, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=True, seed=args.seed) if launched else None
    val_sampler = DistributedSampler(val_set, num_replicas=dist.get_world_size(), rank=dist.get_rank(), shuffle=False, seed=args.seed) if launched else None
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=max(1, min(args.workers, 4)),
        pin_memory=True,
        drop_last=False,
    )

    optimizer = make_optimizer(student, args)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16 == 1)

    start_epoch = 0
    global_step = 0
    resume_info = {}
    if args.resume_ckpt:
        start_epoch, global_step, resume_ckpt = load_resume_checkpoint(
            args.resume_ckpt,
            student,
            optimizer,
            scaler,
            load_optimizer=not args.resume_no_optimizer,
            allow_missing_router=args.resume_allow_missing_router,
        )
        args.lr_schedule_steps_per_epoch = args.lr_schedule_steps_per_epoch or infer_resume_schedule_steps_per_epoch(resume_ckpt, len(train_loader))
        resume_info = {
            "resume_ckpt": args.resume_ckpt,
            "resume_epoch": start_epoch,
            "resume_step": global_step,
            "resume_loaded_optimizer": not args.resume_no_optimizer,
            "resume_allow_missing_router": args.resume_allow_missing_router,
            "lr_schedule_steps_per_epoch": args.lr_schedule_steps_per_epoch,
        }
        if args.prune_resume_logs and dist.is_master():
            prune_resume_logs(out_dir, start_epoch, global_step)
        if dist.is_master():
            print(
                f"Resumed checkpoint {args.resume_ckpt}: epoch={start_epoch}, "
                f"step={global_step}, lr_schedule_steps_per_epoch={args.lr_schedule_steps_per_epoch}",
                flush=True,
            )
    elif args.lr_schedule_steps_per_epoch <= 0:
        args.lr_schedule_steps_per_epoch = math.ceil(len(train_loader) / max(1, int(args.grad_accum_steps)))

    total_steps = args.lr_schedule_total_steps if args.lr_schedule_total_steps > 0 else args.epochs * args.lr_schedule_steps_per_epoch
    stop_candidates = []
    if args.max_train_steps > 0:
        stop_candidates.append(args.max_train_steps)
    if args.max_extra_steps > 0:
        stop_candidates.append(global_step + args.max_extra_steps)
    stop_step = min(stop_candidates) if stop_candidates else None
    metrics_path = out_dir / "logs" / "metrics.jsonl"
    start = time.time()

    completed_epochs = start_epoch
    if dist.is_master():
        write_json(out_dir / "config.json", vars(args))
        if resume_info:
            write_json(out_dir / "logs" / "resume.json", resume_info)

    for epoch in range(start_epoch, args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        global_step, epoch_stats, router_stats = train_one_epoch(
            epoch,
            vae,
            teacher,
            student,
            train_loader,
            optimizer,
            scaler,
            args,
            meta,
            total_steps,
            global_step,
            stop_step,
            metrics_path,
        )
        completed_epochs = epoch + 1
        if dist.is_master():
            write_json(out_dir / "logs" / f"router_stats_epoch{epoch + 1}.json", router_stats)
            append_jsonl(out_dir / "logs" / "epoch_metrics.jsonl", {"epoch": epoch + 1, **epoch_stats})

        if args.eval_every_epoch > 0 and (epoch + 1) % args.eval_every_epoch == 0:
            val_stats = evaluate(vae, unwrap_model(student), val_loader, args)
            if dist.is_master():
                append_jsonl(out_dir / "logs" / "eval_metrics.jsonl", {"epoch": epoch + 1, **val_stats})
                print(f"[eval ep {epoch + 1}] {val_stats}", flush=True)

        if args.save_every_epoch > 0 and (epoch + 1) % args.save_every_epoch == 0:
            save_checkpoint(out_dir / "checkpoints" / f"ckpt_ep{epoch + 1}.pth", student, optimizer, scaler, epoch + 1, global_step, args, meta)
            save_checkpoint(out_dir / "checkpoints" / "ckpt_latest.pth", student, optimizer, scaler, epoch + 1, global_step, args, meta)

        if stop_step is not None and global_step >= stop_step:
            break

    set_model_runtime_topk(student, args.train_topk)
    if dist.is_master():
        elapsed = time.time() - start
        write_json(out_dir / "logs" / "done.json", {"elapsed_seconds": elapsed, "global_step": global_step, "completed_epochs": completed_epochs})
    dist.barrier()
    if dist.initialized():
        dist.finalize()


if __name__ == "__main__":
    main()
