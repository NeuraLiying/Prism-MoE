                      
import os
import sys
import argparse
import math
from pathlib import Path
from tqdm import tqdm
import torch
from typing import List, Tuple, Any, Optional

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).resolve().parents[0]
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.path_utils import add_var_root
from var_d2m_utils import (
    build_var_loss_fn,
    compute_hidden_neuron_importance_taylor,
    split_hidden_by_importance,
    split_hidden_by_importance_two_stage,
    compute_hidden_activation_stability,
    compute_hidden_contribution_energy,
    compute_autoregressive_hidden_contribution_profiles,
    compute_autoregressive_hidden_contribution_profile_stats,
    build_hybrid_assignment_features,
    build_var_d2m_from_ffn,
    init_router_from_expert_weights,
    fit_router_from_activation_energy,
    fit_routers_from_autoregressive_activation_energy,
    calibrate_router_bias_from_teacher_forcing_counts,
    calibrate_router_bias_from_autoregressive_counts,
)

DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _layer_relative_position(layer_idx: int, start_layer: int, end_layer: int) -> float:
    if end_layer <= start_layer:
        return 0.0
    return float(layer_idx - start_layer) / float(end_layer - start_layer)


def _resolve_trajectory_fc1_weight(args, layer_idx: int, start_layer: int, end_layer: int, profile: Optional[torch.Tensor]) -> float:
\
\
\
\
\
\
       
    base = float(args.trajectory_profile_fc1_weight)
    schedule = args.trajectory_profile_fc1_weight_schedule
    if schedule == "fixed":
        return base

    min_w = float(args.trajectory_profile_fc1_weight_min)
    max_w = float(args.trajectory_profile_fc1_weight_max)
    if min_w < 0 or max_w < 0:
        raise ValueError("trajectory_profile_fc1_weight_min/max must be non-negative")
    if min_w > max_w:
        raise ValueError("trajectory_profile_fc1_weight_min cannot exceed max")

    if schedule in {"linear_depth", "cosine_depth"}:
        start_w = (
            float(args.trajectory_profile_fc1_weight_start)
            if args.trajectory_profile_fc1_weight_start is not None
            else min_w
        )
        end_w = (
            float(args.trajectory_profile_fc1_weight_end)
            if args.trajectory_profile_fc1_weight_end is not None
            else max_w
        )
        if start_w < 0 or end_w < 0:
            raise ValueError("trajectory_profile_fc1_weight_start/end must be non-negative")
        t = _layer_relative_position(layer_idx, start_layer, end_layer)
        if schedule == "cosine_depth":
            t = 0.5 - 0.5 * math.cos(math.pi * t)
        return start_w + (end_w - start_w) * t

    if schedule == "profile_concentration":
        if profile is None:
            return base
        P = torch.nan_to_num(profile.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if P.ndim != 2 or P.shape[1] <= 1:
            return base
        P = torch.nn.functional.normalize(P, p=2, dim=1)
        stage_count = float(P.shape[1])
        uniform_max = 1.0 / math.sqrt(stage_count)
        concentration = float(P.abs().max(dim=1).values.mean().item())
        denom = max(1e-12, 1.0 - uniform_max)
        signal = max(0.0, min(1.0, (concentration - uniform_max) / denom))
        return max_w - (max_w - min_w) * signal

    raise ValueError(f"Unknown trajectory_profile_fc1_weight_schedule: {schedule}")


def _resolve_trajectory_fc2_weight(args, layer_idx: int, start_layer: int, end_layer: int, profile: Optional[torch.Tensor]) -> float:
\
\
\
\
\
\
\
       
    base = float(args.trajectory_profile_fc2_weight)
    schedule = args.trajectory_profile_fc2_weight_schedule
    if schedule == "fixed":
        return base

    min_w = float(args.trajectory_profile_fc2_weight_min)
    max_w = float(args.trajectory_profile_fc2_weight_max)
    if min_w < 0 or max_w < 0:
        raise ValueError("trajectory_profile_fc2_weight_min/max must be non-negative")
    if min_w > max_w:
        raise ValueError("trajectory_profile_fc2_weight_min cannot exceed max")

    if schedule in {"linear_depth", "cosine_depth"}:
        start_w = (
            float(args.trajectory_profile_fc2_weight_start)
            if args.trajectory_profile_fc2_weight_start is not None
            else min_w
        )
        end_w = (
            float(args.trajectory_profile_fc2_weight_end)
            if args.trajectory_profile_fc2_weight_end is not None
            else max_w
        )
        if start_w < 0 or end_w < 0:
            raise ValueError("trajectory_profile_fc2_weight_start/end must be non-negative")
        t = _layer_relative_position(layer_idx, start_layer, end_layer)
        if schedule == "cosine_depth":
            t = 0.5 - 0.5 * math.cos(math.pi * t)
        return start_w + (end_w - start_w) * t

    if schedule == "profile_concentration":
        if profile is None:
            return base
        P = torch.nan_to_num(profile.detach().float().cpu(), nan=0.0, posinf=0.0, neginf=0.0)
        if P.ndim != 2 or P.shape[1] <= 1:
            return base
        P = torch.nn.functional.normalize(P, p=2, dim=1)
        stage_count = float(P.shape[1])
        uniform_max = 1.0 / math.sqrt(stage_count)
        concentration = float(P.abs().max(dim=1).values.mean().item())
        denom = max(1e-12, 1.0 - uniform_max)
        signal = max(0.0, min(1.0, (concentration - uniform_max) / denom))
        return min_w + (max_w - min_w) * signal

    raise ValueError(f"Unknown trajectory_profile_fc2_weight_schedule: {schedule}")


@torch.no_grad()
def _prepare_calibration_pairs(
    vae,
    device: torch.device,
    num_classes: int,
    nsamples: int,
    batch_size: int,
    calib_seed: int,
    use_images: bool,
    imagenet_dir: Optional[str],
) -> List[Tuple[torch.Tensor, Any]]:
\
\
\
\
\
\
\
       
    g = torch.Generator(device="cpu")
    g.manual_seed(calib_seed)

    if not use_images:
        pairs: List[Tuple[torch.Tensor, Any]] = []
        n_batches = (nsamples + batch_size - 1) // batch_size
        for bi in range(n_batches):
            cur_bs = min(batch_size, nsamples - bi * batch_size)
            label_B = torch.randint(0, num_classes, (cur_bs,), generator=g).to(device=device)
            seed = calib_seed + bi
            pairs.append((label_B, seed))
        return pairs

    if not imagenet_dir:
        raise ValueError("--imagenet_dir is required when --use_images is set")

    from utils.data import build_dataset

    num_classes_ds, _train_set, val_set = build_dataset(
        data_path=imagenet_dir,
        final_reso=256,
        hflip=False,
        mid_reso=1.125,
    )
    if num_classes_ds != num_classes:
        print(f"Warning: dataset classes={num_classes_ds}, model classes={num_classes}")

    if nsamples <= len(val_set):
        indices = torch.randperm(len(val_set), generator=g)[:nsamples]
    else:
        indices = torch.randperm(len(val_set), generator=g)
        repeat = (nsamples + len(indices) - 1) // len(indices)
        indices = indices.repeat(repeat)[:nsamples]

    sampled_labels = [val_set[int(idx)][1] for idx in indices]
    print(
        f"Using real-image calibration: {len(indices)} samples from {imagenet_dir}/val, "
        f"covering {len(set(sampled_labels))}/{num_classes_ds} classes"
    )

    pairs = []
    n_batches = (len(indices) + batch_size - 1) // batch_size
    for bi in tqdm(range(n_batches), desc="Encoding calibration images"):
        batch_indices = indices[bi * batch_size:(bi + 1) * batch_size]
        images = []
        labels = []
        for idx in batch_indices:
            img, label = val_set[int(idx)]
            images.append(img)
            labels.append(label)

        image_B3HW = torch.stack(images).to(device, non_blocking=True)
        label_B = torch.tensor(labels, device=device, dtype=torch.long)
        gt_idx_Bl = vae.img_to_idxBl(image_B3HW)
        gt_BL = torch.cat(gt_idx_Bl, dim=1)
        tokens_BLCv = vae.quantize.idxBl_to_var_input(gt_idx_Bl)
        seed = calib_seed + bi
        pairs.append((label_B, (tokens_BLCv.detach(), gt_BL.detach(), seed)))

    return pairs


def main():
    parser = argparse.ArgumentParser("VAR Dense2MoE Conversion (FFN->MoE)")

                 
    parser.add_argument("--vae_ckpt", type=str,
                        default="/home/liying/pretrained/model_zoo/vae_ch160v4096z32.pth")
    parser.add_argument("--var_ckpt", type=str,
                        default="/home/liying/pretrained/model_zoo/var_d16.pth")
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--var_root", type=str, default=None,
                        help="Path to the VAR runtime root. Defaults to $VAR_ROOT or this standalone project.")

                  
    parser.add_argument("--model_depth", type=int, default=16, choices=[16, 20, 24, 30])
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--shared_aln", action="store_true")

                                         
    parser.add_argument("--nexperts", type=int, default=12, help="number of normal experts (n)")
    parser.add_argument("--topk", type=int, default=2, help="top-k normal experts per token (k)")
    parser.add_argument("--shared_ratio", type=float, default=0.25,
                        help="shared hidden ratio in [0,1], shared_hidden = round(shared_ratio * H)")
    parser.add_argument("--hard_mode", action="store_true",
                        help="hard routing (TopK on logits, no weights). For debugging; paper uses weights g(x,t,i).")

                              
    parser.add_argument("--nsamples", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--calib_seed", type=int, default=42)
    parser.add_argument("--use_images", action="store_true",
                        help="Use real ImageNet val images for Taylor/stability calibration.")
    parser.add_argument("--imagenet_dir", type=str, default=None,
                        help="ImageNet root with train/val folders. Required with --use_images.")
    

                    
    parser.add_argument("--loss_mode", type=str, default="trainer",
                        choices=["trainer", "model", "custom_stub"],
                        help="how to compute differentiable loss for VAR on calib samples")
    parser.add_argument("--cfg", type=float, default=4.0,
                        help="kept for compatibility; used only if your loss_fn uses it.")

                                  
    parser.add_argument("--use_two_stage", action="store_true",
                        help="Use two-stage sorting: Taylor importance -> activation stability")
    parser.add_argument("--candidate_multiplier", type=float, default=2.0,
                        help="Candidate pool multiplier for two-stage sorting (M = multiplier × shared_hidden)")
    parser.add_argument("--shared_second_score", type=str, default="stability",
                        choices=["stability", "contribution_energy", "trajectory_contribution_energy"],
                        help="Second-stage score used to choose shared neurons from the Taylor candidate pool.")
    parser.add_argument("--shared_selection_mode", type=str, default="second_score",
                        choices=["second_score", "rank_fusion"],
                        help="How to rank neurons inside the Taylor candidate pool for shared expert selection. "
                             "second_score preserves historical behavior; rank_fusion combines Taylor rank "
                             "and the selected second-stage score.")
    parser.add_argument("--shared_importance_weight", type=float, default=0.0,
                        help="Taylor-rank weight for --shared_selection_mode rank_fusion. "
                             "0.0 preserves pure second-score selection; 1.0 uses pure Taylor rank.")
    parser.add_argument("--contribution_max_tokens", type=int, default=8192,
                        help="Max tokens per calibration forward hook for --shared_second_score contribution_energy.")
    parser.add_argument("--contribution_transform", type=str, default="log",
                        choices=["log", "sqrt", "none"],
                        help="Transform applied to contribution-energy scores before shared selection.")
    parser.add_argument("--trajectory_shared_score_mode", type=str, default="sum",
                        choices=["sum", "max", "mean", "stable_contribution"],
                        help="How to aggregate AR stage-wise contribution profiles for shared selection.")
    parser.add_argument("--expert_assignment", type=str, default="contiguous",
                        choices=["contiguous", "round_robin", "balanced_kmeans", "trajectory_profile_kmeans"],
                        help="How to assign non-shared hidden neurons to experts after sorting.")
    parser.add_argument("--kmeans_iters", type=int, default=8,
                        help="Iterations for k-means based expert assignment.")
    parser.add_argument("--kmeans_restarts", type=int, default=1,
                        help="Deterministic restarts for k-means based expert assignment. "
                             "Default 1 preserves the historical initialization.")
    parser.add_argument("--trajectory_profile_nsamples", type=int, default=None,
                        help="Dense AR samples for --expert_assignment trajectory_profile_kmeans. Defaults to --nsamples.")
    parser.add_argument("--trajectory_profile_batch_size", type=int, default=None,
                        help="Batch size for --expert_assignment trajectory_profile_kmeans. Defaults to --batch_size.")
    parser.add_argument("--trajectory_profile_top_k", type=int, default=900,
                        help="Sampling top-k for trajectory profile collection.")
    parser.add_argument("--trajectory_profile_top_p", type=float, default=0.96,
                        help="Sampling top-p for trajectory profile collection.")
    parser.add_argument("--trajectory_profile_max_tokens", type=int, default=8192,
                        help="Max tokens per AR stage hook for trajectory profile collection.")
    parser.add_argument("--trajectory_profile_transform", type=str, default="log",
                        choices=["log", "sqrt", "none"],
                        help="Transform applied to stage-wise contribution profiles.")
    parser.add_argument("--trajectory_profile_position_bins", type=int, default=1,
                        help="Number of position bins per AR stage for trajectory contribution profiles. "
                             "Default 1 preserves the validated stage-only profile behavior.")
    parser.add_argument("--trajectory_profile_feature_weight", type=float, default=1.0,
                        help="Weight for trajectory profile features in trajectory_profile_kmeans.")
    parser.add_argument("--trajectory_profile_fc1_weight", type=float, default=0.5,
                        help="Weight for fc1 direction features in trajectory_profile_kmeans.")
    parser.add_argument("--trajectory_profile_fc2_weight", type=float, default=0.0,
                        help="Weight for fc2 output-direction features in trajectory_profile_kmeans.")
    parser.add_argument("--trajectory_profile_fc2_weight_schedule", type=str, default="fixed",
                        choices=["fixed", "linear_depth", "cosine_depth", "profile_concentration"],
                        help="Optional per-layer fc2 output-direction feature weight schedule for "
                             "trajectory_profile_kmeans. fixed preserves the scalar "
                             "--trajectory_profile_fc2_weight behavior.")
    parser.add_argument("--trajectory_profile_fc2_weight_min", type=float, default=0.0,
                        help="Minimum fc2 feature weight for adaptive trajectory-profile schedules.")
    parser.add_argument("--trajectory_profile_fc2_weight_max", type=float, default=0.2,
                        help="Maximum fc2 feature weight for adaptive trajectory-profile schedules.")
    parser.add_argument("--trajectory_profile_fc2_weight_start", type=float, default=None,
                        help="Start-layer fc2 feature weight for linear_depth/cosine_depth schedules. "
                             "Defaults to --trajectory_profile_fc2_weight_min.")
    parser.add_argument("--trajectory_profile_fc2_weight_end", type=float, default=None,
                        help="End-layer fc2 feature weight for linear_depth/cosine_depth schedules. "
                             "Defaults to --trajectory_profile_fc2_weight_max.")
    parser.add_argument("--trajectory_profile_stage_onehot_weight", type=float, default=0.0,
                        help="Weight for dominant AR-stage one-hot features in trajectory_profile_kmeans.")
    parser.add_argument("--trajectory_profile_fc1_weight_schedule", type=str, default="fixed",
                        choices=["fixed", "linear_depth", "cosine_depth", "profile_concentration"],
                        help="Optional per-layer fc1 feature weight schedule for trajectory_profile_kmeans. "
                             "fixed preserves the scalar --trajectory_profile_fc1_weight behavior.")
    parser.add_argument("--trajectory_profile_fc1_weight_min", type=float, default=0.32,
                        help="Minimum fc1 feature weight for adaptive trajectory-profile schedules.")
    parser.add_argument("--trajectory_profile_fc1_weight_max", type=float, default=0.48,
                        help="Maximum fc1 feature weight for adaptive trajectory-profile schedules.")
    parser.add_argument("--trajectory_profile_fc1_weight_start", type=float, default=None,
                        help="Start-layer fc1 feature weight for linear_depth/cosine_depth schedules. "
                             "Defaults to --trajectory_profile_fc1_weight_min.")
    parser.add_argument("--trajectory_profile_fc1_weight_end", type=float, default=None,
                        help="End-layer fc1 feature weight for linear_depth/cosine_depth schedules. "
                             "Defaults to --trajectory_profile_fc1_weight_max.")
    parser.add_argument("--router_init", type=str, default="centroid",
                        choices=["centroid", "importance_centroid", "calibrated_energy", "trajectory_energy"],
                        help="Router initialization method for normal experts.")
    parser.add_argument("--router_bias", action="store_true",
                        help="Use a router projection bias and fit it as a per-expert calibrated prior.")
    parser.add_argument("--router_force_bias", action="store_true",
                        help="Create router bias parameters even when the ridge router fit itself is bias-free. "
                             "Useful for closed-form post-calibration priors.")
    parser.add_argument("--router_calib_max_tokens", type=int, default=8192,
                        help="Max tokens per calibration forward hook for calibrated-energy router init.")
    parser.add_argument("--router_ridge_lambda", type=float, default=1e-2,
                        help="Relative ridge coefficient for calibrated-energy router init.")
    parser.add_argument("--router_target_transform", type=str, default="log",
                        choices=["log", "sqrt", "none"],
                        help="Target transform for calibrated-energy router init.")
    parser.add_argument("--router_target_metric", type=str, default="activation_norm",
                        choices=["activation_norm", "output_norm", "activation_topk", "output_topk"],
                        help="Expert contribution metric for calibrated/trajectory router targets. "
                             "activation_norm keeps the historical fast proxy; output_norm fits "
                             "targets from each dense expert slice's actual fc2 output norm; "
                             "activation_topk/output_topk convert those scores into hard-TopK "
                             "membership targets.")
    parser.add_argument("--trajectory_router_nsamples", type=int, default=None,
                        help="Number of dense autoregressive samples for --router_init trajectory_energy. Defaults to --nsamples.")
    parser.add_argument("--trajectory_router_batch_size", type=int, default=None,
                        help="Batch size for --router_init trajectory_energy. Defaults to --batch_size.")
    parser.add_argument("--trajectory_router_top_k", type=int, default=900,
                        help="Sampling top-k for dense autoregressive trajectory router calibration.")
    parser.add_argument("--trajectory_router_top_p", type=float, default=0.96,
                        help="Sampling top-p for dense autoregressive trajectory router calibration.")
    parser.add_argument("--trajectory_router_stage_weight", type=str, default="uniform",
                        choices=["token", "uniform", "sqrt"],
                        help="How to weight generation stages in trajectory router ridge stats.")
    parser.add_argument("--router_balance_calib", type=str, default="none",
                        choices=["none", "teacher_forcing", "trajectory"],
                        help="Training-free post-calibration for hard-TopK route balance. "
                             "Adds a calibrated expert-prior bias from actual route counts.")
    parser.add_argument("--router_balance_strength", type=float, default=0.25,
                        help="Strength for route-balance log-prior correction.")
    parser.add_argument("--router_balance_max_abs_bias", type=float, default=2.0,
                        help="Maximum absolute calibrated router bias after route-balance correction.")
    parser.add_argument("--router_balance_delta_linf_cap", type=float, default=0.0,
                        help="Optional per-layer cap for the route-balance correction delta L-infinity norm. "
                             "0 disables the adaptive cap and preserves historical behavior.")
    parser.add_argument("--router_balance_target_metric", type=str, default="uniform",
                        choices=["uniform", "activation_norm", "output_norm", "activation_topk", "output_topk"],
                        help="Target route prior for trajectory route-balance calibration. "
                             "uniform preserves the validated behavior. The contribution-aware targets "
                             "estimate each expert's dense-slice contribution on dense AR hidden states "
                             "and bias hard routing toward that target frequency.")
    parser.add_argument("--router_balance_target_transform", type=str, default="log",
                        choices=["log", "sqrt", "none"],
                        help="Transform applied before aggregating contribution-aware route-balance targets.")
    parser.add_argument("--router_balance_target_mix_uniform", type=float, default=0.0,
                        help="Mix contribution-aware route target with uniform target. "
                             "0 uses the contribution target; 1 recovers uniform.")
    parser.add_argument("--router_balance_nsamples", type=int, default=None,
                        help="Dense AR samples for --router_balance_calib trajectory. Defaults to trajectory router nsamples or nsamples.")
    parser.add_argument("--router_balance_batch_size", type=int, default=None,
                        help="Dense AR batch size for --router_balance_calib trajectory. Defaults to trajectory router batch size or batch_size.")
    parser.add_argument("--router_balance_top_k", type=int, default=None,
                        help="Sampling top-k for trajectory route-balance calibration. Defaults to --trajectory_router_top_k.")
    parser.add_argument("--router_balance_top_p", type=float, default=None,
                        help="Sampling top-p for trajectory route-balance calibration. Defaults to --trajectory_router_top_p.")
    parser.add_argument("--router_balance_stage_weight", type=str, default="token",
                        choices=["token", "uniform", "sqrt"],
                        help="How to weight generation stages when collecting trajectory route-balance counts. "
                             "token preserves the historical token-count behavior; uniform gives each AR stage "
                             "equal total count weight; sqrt is between the two.")
    parser.add_argument("--router_balance_label_sampling", type=str, default="random",
                        choices=["random", "cycle", "stratified"],
                        help="Class-label schedule for trajectory route-balance calibration. "
                             "random preserves historical behavior; cycle/stratified reduce class-coverage "
                             "variance without using labels or samples from the FID reference stats.")

                 
    parser.add_argument("--start_layer", type=int, default=0)
    parser.add_argument("--end_layer", type=int, default=None)

    args = parser.parse_args()

    var_root = add_var_root(args.var_root)
    print(f"Using VAR root: {var_root}")

    from common import dist

    dist.initialize()
    device = dist.get_device() if hasattr(dist, "get_device") else DEV

                 
    from models import build_vae_var
    patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
    vae, var = build_vae_var(
        V=4096, Cvae=32, ch=160, share_quant_resi=4,
        device=device, patch_nums=patch_nums,
        num_classes=args.num_classes, depth=args.model_depth, shared_aln=args.shared_aln,
    )

                  
    print(f"Loading VAE weights: {args.vae_ckpt}")
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device), strict=True)

    print(f"Loading VAR weights: {args.var_ckpt}")
    ckpt = torch.load(args.var_ckpt, map_location="cpu")
    if isinstance(ckpt, dict) and "trainer" in ckpt and "var_wo_ddp" in ckpt["trainer"]:
        var.load_state_dict(ckpt["trainer"]["var_wo_ddp"], strict=True)
        print("Loaded VAR weights from trainer[var_wo_ddp]")
    elif isinstance(ckpt, dict) and "var_wo_ddp" in ckpt:
        var.load_state_dict(ckpt["var_wo_ddp"], strict=True)
        print("Loaded VAR weights from var_wo_ddp")
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        var.load_state_dict(ckpt["state_dict"], strict=True)
        print("Loaded VAR weights from state_dict")
    else:
        var.load_state_dict(ckpt, strict=True)
        print("Loaded VAR weights directly")

    var = var.to(device).eval()
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad_(False)

    calib_pairs = _prepare_calibration_pairs(
        vae=vae,
        device=device,
        num_classes=var.num_classes,
        nsamples=args.nsamples,
        batch_size=args.batch_size,
        calib_seed=args.calib_seed,
        use_images=args.use_images,
        imagenet_dir=args.imagenet_dir,
    )

                 
    n_layers = len(var.blocks)
    start_layer = args.start_layer
    end_layer = args.end_layer if args.end_layer is not None else (n_layers - 1)
    print(f"Converting layers {start_layer}..{end_layer} (total blocks={n_layers})")

                                                                  
                                                     
                                                                  
                                                                          
                                                              
                                                                     
                                                                  
    loss_fn = build_var_loss_fn(var=var, vae=vae, mode=args.loss_mode, args=args)

                                                                  
                                                               
                                                                  
                                                                        
                                                                        
    if args.use_two_stage:
        print(
            "\nStage 1: Computing Taylor importance and second-stage shared scores "
            f"({args.shared_second_score}) for all layers (on dense model)..."
        )
    else:
        print("\nStage 1: Computing Taylor importance for all layers (on dense model)...")
    
    layer_importances = {}
    layer_importance_scores = {}
    layer_router_weights = {}
    layer_configs = {}
    trajectory_assignment_profiles = {}
    trajectory_shared_scores = {}
    needs_trajectory_profiles = (
        args.expert_assignment == "trajectory_profile_kmeans"
        or (args.use_two_stage and args.shared_second_score == "trajectory_contribution_energy")
    )
    if needs_trajectory_profiles:
        profile_nsamples = args.trajectory_profile_nsamples or args.nsamples
        profile_batch_size = args.trajectory_profile_batch_size or args.batch_size
        profile_layer_indices = []
        for layer_idx in range(start_layer, end_layer + 1):
            block = var.blocks[layer_idx]
            if hasattr(block, "ffn") and hasattr(block.ffn, "fc1") and hasattr(block.ffn, "fc2"):
                profile_layer_indices.append(layer_idx)
        print(
            "\nCollecting trajectory contribution profiles "
            f"(layers={len(profile_layer_indices)}, nsamples={profile_nsamples}, "
            f"batch_size={profile_batch_size}, max_tokens={args.trajectory_profile_max_tokens}, "
            f"transform={args.trajectory_profile_transform}, "
            f"position_bins={args.trajectory_profile_position_bins}, "
            f"profile_weight={args.trajectory_profile_feature_weight}, "
            f"fc1_weight={args.trajectory_profile_fc1_weight}, "
            f"fc2_weight={args.trajectory_profile_fc2_weight}, "
            f"fc2_weight_schedule={args.trajectory_profile_fc2_weight_schedule}, "
            f"stage_onehot_weight={args.trajectory_profile_stage_onehot_weight}, "
            f"fc1_weight_schedule={args.trajectory_profile_fc1_weight_schedule}, "
            f"shared_score_mode={args.trajectory_shared_score_mode})..."
        )
        trajectory_assignment_profiles, trajectory_shared_scores = (
            compute_autoregressive_hidden_contribution_profile_stats(
                model=var,
                layer_indices=profile_layer_indices,
                device=device,
                nsamples=profile_nsamples,
                batch_size=profile_batch_size,
                calib_seed=args.calib_seed,
                num_classes=args.num_classes,
                cfg=args.cfg,
                top_k=args.trajectory_profile_top_k,
                top_p=args.trajectory_profile_top_p,
                max_tokens_per_call=args.trajectory_profile_max_tokens,
                transform=args.trajectory_profile_transform,
                shared_score_mode=args.trajectory_shared_score_mode,
                position_bins=args.trajectory_profile_position_bins,
            )
        )
    
    for layer_idx in tqdm(range(start_layer, end_layer + 1), desc="Computing importance"):
        block = var.blocks[layer_idx]
        if not (hasattr(block, "ffn") and hasattr(block.ffn, "fc1") and hasattr(block.ffn, "fc2")):
            print(f"[Layer {layer_idx}] skip: not standard FFN(fc1/fc2)")
            continue

        ffn = block.ffn
        H = ffn.fc1.out_features
        C = ffn.fc1.in_features

        shared_hidden = int(round(args.shared_ratio * H))
        shared_hidden = max(1, min(shared_hidden, H - args.nexperts))

        remain = H - shared_hidden
        if remain % args.nexperts != 0:
            new_remain = (remain // args.nexperts) * args.nexperts
            shared_hidden = H - new_remain
            remain = new_remain

        expert_hidden = remain // args.nexperts
        assert expert_hidden > 0, "expert_hidden must be > 0; reduce shared_ratio or n_experts"

        print(f"\n[Layer {layer_idx}] C={C}, H={H}, shared_hidden={shared_hidden}, "
              f"n_experts={args.nexperts}, expert_hidden={expert_hidden}, topk={args.topk}")

        assignment_features = None
        trajectory_layer_fc1_weight = None
        trajectory_layer_fc2_weight = None
        if args.expert_assignment == "trajectory_profile_kmeans":
            trajectory_layer_fc1_weight = _resolve_trajectory_fc1_weight(
                args=args,
                layer_idx=layer_idx,
                start_layer=start_layer,
                end_layer=end_layer,
                profile=trajectory_assignment_profiles[layer_idx],
            )
            trajectory_layer_fc2_weight = _resolve_trajectory_fc2_weight(
                args=args,
                layer_idx=layer_idx,
                start_layer=start_layer,
                end_layer=end_layer,
                profile=trajectory_assignment_profiles[layer_idx],
            )
            print(
                f"  trajectory_profile_kmeans feature weights: "
                f"fc1={trajectory_layer_fc1_weight:.6f}, "
                f"fc2={trajectory_layer_fc2_weight:.6f}, "
                f"profile={args.trajectory_profile_feature_weight:.6f}, "
                f"stage_onehot={args.trajectory_profile_stage_onehot_weight:.6f}, "
                f"fc1_schedule={args.trajectory_profile_fc1_weight_schedule}, "
                f"fc2_schedule={args.trajectory_profile_fc2_weight_schedule}"
            )
            assignment_features = build_hybrid_assignment_features(
                ffn_fc1_weight=ffn.fc1.weight.data,
                ffn_fc2_weight=ffn.fc2.weight.data,
                trajectory_profile=trajectory_assignment_profiles[layer_idx],
                fc1_weight=trajectory_layer_fc1_weight,
                fc2_weight=trajectory_layer_fc2_weight,
                profile_weight=args.trajectory_profile_feature_weight,
                stage_onehot_weight=args.trajectory_profile_stage_onehot_weight,
            )

                                                                             
        importance = compute_hidden_neuron_importance_taylor(
            model=var,
            layer_idx=layer_idx,
            calib_pairs=calib_pairs,
            loss_fn=loss_fn,
            device=device,
        )
        
                                                 
        if args.use_two_stage:
            if args.shared_second_score == "stability":
                print(f"  Computing activation stability for layer {layer_idx}...")
                second_score = compute_hidden_activation_stability(
                    model=var,
                    layer_idx=layer_idx,
                    calib_pairs=calib_pairs,
                    vae=vae,
                    device=device,
                )
            elif args.shared_second_score == "contribution_energy":
                print(
                    f"  Computing contribution energy for layer {layer_idx} "
                    f"(max_tokens={args.contribution_max_tokens}, "
                    f"transform={args.contribution_transform})..."
                )
                second_score = compute_hidden_contribution_energy(
                    model=var,
                    layer_idx=layer_idx,
                    calib_pairs=calib_pairs,
                    vae=vae,
                    device=device,
                    max_tokens_per_call=args.contribution_max_tokens,
                    transform=args.contribution_transform,
                )
            elif args.shared_second_score == "trajectory_contribution_energy":
                print(
                    f"  Using trajectory contribution energy for layer {layer_idx} "
                    f"(mode={args.trajectory_shared_score_mode})..."
                )
                second_score = trajectory_shared_scores[layer_idx]
            else:
                raise ValueError(f"Unknown shared_second_score: {args.shared_second_score}")
            
                     
            shared_idx, expert_idx_list = split_hidden_by_importance_two_stage(
                importance=importance,
                stability=second_score,
                shared_hidden=shared_hidden,
                n_experts=args.nexperts,
                candidate_multiplier=args.candidate_multiplier,
                shared_selection_mode=args.shared_selection_mode,
                shared_importance_weight=args.shared_importance_weight,
                expert_assignment=args.expert_assignment,
                ffn_fc1_weight=ffn.fc1.weight.data,
                assignment_features=assignment_features,
                kmeans_iters=args.kmeans_iters,
                kmeans_restarts=args.kmeans_restarts,
            )
            M = int(round(args.candidate_multiplier * shared_hidden))
            print(f"  Two-stage sorting: candidate_pool={M}, "
                  f"selected {shared_hidden} shared neurons from pool by {args.shared_second_score}, "
                  f"shared_selection_mode={args.shared_selection_mode}, "
                  f"shared_importance_weight={args.shared_importance_weight}, "
                  f"expert_assignment={args.expert_assignment}")
        else:
                     
            shared_idx, expert_idx_list = split_hidden_by_importance(
                importance=importance,
                shared_hidden=shared_hidden,
                n_experts=args.nexperts,
                expert_assignment=args.expert_assignment,
                ffn_fc1_weight=ffn.fc1.weight.data,
                assignment_features=assignment_features,
                kmeans_iters=args.kmeans_iters,
                kmeans_restarts=args.kmeans_restarts,
            )

        if args.router_init == "calibrated_energy":
            print(
                f"  Fitting calibrated-energy router for layer {layer_idx} "
                f"(max_tokens={args.router_calib_max_tokens}, ridge={args.router_ridge_lambda})..."
            )
            layer_router_weights[layer_idx] = fit_router_from_activation_energy(
                model=var,
                layer_idx=layer_idx,
                calib_pairs=calib_pairs,
                vae=vae,
                expert_indices_list=expert_idx_list,
                device=device,
                max_tokens=args.router_calib_max_tokens,
                ridge_lambda=args.router_ridge_lambda,
                target_transform=args.router_target_transform,
                target_metric=args.router_target_metric,
                topk=args.topk,
                fit_bias=args.router_bias,
            )
            print(f"  Calibrated router fitted for layer {layer_idx}.")
        
                                   
        layer_importances[layer_idx] = (shared_idx, expert_idx_list)
        layer_importance_scores[layer_idx] = importance
        layer_configs[layer_idx] = {
            'C': C,
            'H': H,
            'shared_hidden': shared_hidden,
            'expert_hidden': expert_hidden,
            'trajectory_profile_fc1_weight_used': trajectory_layer_fc1_weight,
            'trajectory_profile_fc2_weight_used': trajectory_layer_fc2_weight,
        }

    if args.router_init == "trajectory_energy":
        trajectory_nsamples = args.trajectory_router_nsamples or args.nsamples
        trajectory_batch_size = args.trajectory_router_batch_size or args.batch_size
        print(
            "\nFitting trajectory-energy routers on dense autoregressive hidden states "
            f"(nsamples={trajectory_nsamples}, batch_size={trajectory_batch_size}, "
            f"max_tokens={args.router_calib_max_tokens}, ridge={args.router_ridge_lambda}, "
            f"stage_weight={args.trajectory_router_stage_weight})..."
        )
        layer_router_weights = fit_routers_from_autoregressive_activation_energy(
            model=var,
            layer_expert_indices={
                layer_idx: expert_idx_list
                for layer_idx, (_shared_idx, expert_idx_list) in layer_importances.items()
            },
            device=device,
            nsamples=trajectory_nsamples,
            batch_size=trajectory_batch_size,
            calib_seed=args.calib_seed,
            num_classes=args.num_classes,
            cfg=args.cfg,
            top_k=args.trajectory_router_top_k,
            top_p=args.trajectory_router_top_p,
            max_tokens_per_call=args.router_calib_max_tokens,
            ridge_lambda=args.router_ridge_lambda,
            target_transform=args.router_target_transform,
            stage_weight=args.trajectory_router_stage_weight,
            target_metric=args.router_target_metric,
            topk=args.topk,
            fit_bias=args.router_bias,
        )
        print("Trajectory-energy routers fitted.")

    router_balance_stats = {}
    if args.router_balance_calib != "none":
        if args.router_init not in {"calibrated_energy", "trajectory_energy"}:
            raise ValueError("--router_balance_calib requires calibrated_energy or trajectory_energy router init")
        print(
            f"\nApplying training-free router route-balance calibration "
            f"({args.router_balance_calib}, strength={args.router_balance_strength}, "
            f"max_abs_bias={args.router_balance_max_abs_bias})..."
        )
        if args.router_balance_calib == "teacher_forcing":
            layer_router_weights, router_balance_stats = calibrate_router_bias_from_teacher_forcing_counts(
                model=var,
                layer_router_weights=layer_router_weights,
                calib_pairs=calib_pairs,
                device=device,
                vae=vae,
                topk=args.topk,
                max_tokens_per_call=args.router_calib_max_tokens,
                strength=args.router_balance_strength,
                max_abs_bias=args.router_balance_max_abs_bias,
                delta_linf_cap=args.router_balance_delta_linf_cap,
            )
        elif args.router_balance_calib == "trajectory":
            balance_nsamples = (
                args.router_balance_nsamples
                or args.trajectory_router_nsamples
                or args.nsamples
            )
            balance_batch_size = (
                args.router_balance_batch_size
                or args.trajectory_router_batch_size
                or args.batch_size
            )
            layer_router_weights, router_balance_stats = calibrate_router_bias_from_autoregressive_counts(
                model=var,
                layer_router_weights=layer_router_weights,
                device=device,
                nsamples=balance_nsamples,
                batch_size=balance_batch_size,
                calib_seed=args.calib_seed,
                num_classes=args.num_classes,
                topk=args.topk,
                cfg=args.cfg,
                top_k=args.router_balance_top_k or args.trajectory_router_top_k,
                top_p=args.router_balance_top_p if args.router_balance_top_p is not None else args.trajectory_router_top_p,
                max_tokens_per_call=args.router_calib_max_tokens,
                strength=args.router_balance_strength,
                max_abs_bias=args.router_balance_max_abs_bias,
                stage_weight=args.router_balance_stage_weight,
                delta_linf_cap=args.router_balance_delta_linf_cap,
                target_metric=args.router_balance_target_metric,
                target_transform=args.router_balance_target_transform,
                target_mix_uniform=args.router_balance_target_mix_uniform,
                layer_expert_indices={
                    layer_idx: expert_idx_list
                    for layer_idx, (_shared_idx, expert_idx_list) in layer_importances.items()
                },
                label_sampling=args.router_balance_label_sampling,
            )
        else:
            raise AssertionError(args.router_balance_calib)
        print("Router route-balance calibration applied.")

                                                                  
                                                         
                                                                  
    print("\nStage 2: Converting layers to MoE (using cached importance)...")
    for layer_idx in tqdm(range(start_layer, end_layer + 1), desc="Converting to MoE"):
        if layer_idx not in layer_importances:
            continue
        
        block = var.blocks[layer_idx]
        ffn = block.ffn
        shared_idx, expert_idx_list = layer_importances[layer_idx]
        config = layer_configs[layer_idx]
        
                                         
        moe_ffn = build_var_d2m_from_ffn(
            ffn=ffn,
            shared_indices=shared_idx,
            expert_indices_list=expert_idx_list,
            n_experts=args.nexperts,
            topk=args.topk,
            hard_mode=args.hard_mode,
            device=device,
            router_bias=args.router_bias or args.router_force_bias or args.router_balance_calib != "none",
        )

                                                
        if args.router_init in {"calibrated_energy", "trajectory_energy"}:
            with torch.no_grad():
                router_entry = layer_router_weights[layer_idx]
                if isinstance(router_entry, tuple):
                    router_w, router_b = router_entry
                else:
                    router_w, router_b = router_entry, None
                router_w = router_w.to(
                    device=device,
                    dtype=moe_ffn.gate.proj.weight.dtype,
                )
                moe_ffn.gate.proj.weight.data.copy_(router_w)
                if moe_ffn.gate.proj.bias is not None:
                    if router_b is None:
                        moe_ffn.gate.proj.bias.zero_()
                    else:
                        moe_ffn.gate.proj.bias.data.copy_(
                            router_b.to(
                                device=device,
                                dtype=moe_ffn.gate.proj.bias.dtype,
                            )
                        )
        else:
            init_router_from_expert_weights(
                router=moe_ffn.gate,
                ffn_fc1_weight=ffn.fc1.weight.data,
                expert_indices_list=expert_idx_list,
                device=device,
                importance=(
                    layer_importance_scores[layer_idx]
                    if args.router_init == "importance_centroid"
                    else None
                ),
            )

        block.ffn = moe_ffn
        print(f"[Layer {layer_idx}] converted.")

          
    print(f"\nSaving converted model to: {args.output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    torch.save(
        {
            "var_wo_ddp": var.state_dict(),
            "config": {
                "method": "Dense2MoE_FFN_to_MoE",
                "nexperts": args.nexperts,
                "topk": args.topk,
                "shared_ratio": args.shared_ratio,
                "hard_mode": args.hard_mode,
                "loss_mode": args.loss_mode,
                "use_images": args.use_images,
                "imagenet_dir": args.imagenet_dir if args.use_images else None,
                "nsamples": args.nsamples,
                "calib_seed": args.calib_seed,
                "use_two_stage": args.use_two_stage,
                "candidate_multiplier": args.candidate_multiplier if args.use_two_stage else None,
                "shared_second_score": args.shared_second_score if args.use_two_stage else None,
                "shared_selection_mode": args.shared_selection_mode if args.use_two_stage else None,
                "shared_importance_weight": args.shared_importance_weight if args.use_two_stage else None,
                "contribution_max_tokens": (
                    args.contribution_max_tokens
                    if args.use_two_stage and args.shared_second_score == "contribution_energy"
                    else None
                ),
                "contribution_transform": (
                    args.contribution_transform
                    if args.use_two_stage and args.shared_second_score == "contribution_energy"
                    else None
                ),
                "trajectory_shared_score_mode": (
                    args.trajectory_shared_score_mode
                    if args.use_two_stage and args.shared_second_score == "trajectory_contribution_energy"
                    else None
                ),
                "expert_assignment": args.expert_assignment,
                "kmeans_iters": (
                    args.kmeans_iters
                    if args.expert_assignment in {"balanced_kmeans", "trajectory_profile_kmeans"}
                    else None
                ),
                "kmeans_restarts": (
                    args.kmeans_restarts
                    if args.expert_assignment in {"balanced_kmeans", "trajectory_profile_kmeans"}
                    else None
                ),
                "trajectory_profile_nsamples": (
                    (args.trajectory_profile_nsamples or args.nsamples)
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_batch_size": (
                    (args.trajectory_profile_batch_size or args.batch_size)
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_top_k": (
                    args.trajectory_profile_top_k
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_top_p": (
                    args.trajectory_profile_top_p
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_max_tokens": (
                    args.trajectory_profile_max_tokens
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_transform": (
                    args.trajectory_profile_transform
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_position_bins": (
                    args.trajectory_profile_position_bins
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_feature_weight": (
                    args.trajectory_profile_feature_weight
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_fc1_weight": (
                    args.trajectory_profile_fc1_weight
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_fc2_weight": (
                    args.trajectory_profile_fc2_weight
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_fc2_weight_schedule": (
                    args.trajectory_profile_fc2_weight_schedule
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_fc2_weight_min": (
                    args.trajectory_profile_fc2_weight_min
                    if (
                        args.expert_assignment == "trajectory_profile_kmeans"
                        and args.trajectory_profile_fc2_weight_schedule != "fixed"
                    ) else None
                ),
                "trajectory_profile_fc2_weight_max": (
                    args.trajectory_profile_fc2_weight_max
                    if (
                        args.expert_assignment == "trajectory_profile_kmeans"
                        and args.trajectory_profile_fc2_weight_schedule != "fixed"
                    ) else None
                ),
                "trajectory_profile_fc2_weight_start": (
                    args.trajectory_profile_fc2_weight_start
                    if (
                        args.expert_assignment == "trajectory_profile_kmeans"
                        and args.trajectory_profile_fc2_weight_schedule in {"linear_depth", "cosine_depth"}
                    ) else None
                ),
                "trajectory_profile_fc2_weight_end": (
                    args.trajectory_profile_fc2_weight_end
                    if (
                        args.expert_assignment == "trajectory_profile_kmeans"
                        and args.trajectory_profile_fc2_weight_schedule in {"linear_depth", "cosine_depth"}
                    ) else None
                ),
                "trajectory_profile_stage_onehot_weight": (
                    args.trajectory_profile_stage_onehot_weight
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_fc1_weight_schedule": (
                    args.trajectory_profile_fc1_weight_schedule
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_fc1_weight_min": (
                    args.trajectory_profile_fc1_weight_min
                    if (
                        args.expert_assignment == "trajectory_profile_kmeans"
                        and args.trajectory_profile_fc1_weight_schedule != "fixed"
                    ) else None
                ),
                "trajectory_profile_fc1_weight_max": (
                    args.trajectory_profile_fc1_weight_max
                    if (
                        args.expert_assignment == "trajectory_profile_kmeans"
                        and args.trajectory_profile_fc1_weight_schedule != "fixed"
                    ) else None
                ),
                "trajectory_profile_fc1_weight_start": (
                    args.trajectory_profile_fc1_weight_start
                    if (
                        args.expert_assignment == "trajectory_profile_kmeans"
                        and args.trajectory_profile_fc1_weight_schedule in {"linear_depth", "cosine_depth"}
                    ) else None
                ),
                "trajectory_profile_fc1_weight_end": (
                    args.trajectory_profile_fc1_weight_end
                    if (
                        args.expert_assignment == "trajectory_profile_kmeans"
                        and args.trajectory_profile_fc1_weight_schedule in {"linear_depth", "cosine_depth"}
                    ) else None
                ),
                "trajectory_profile_fc1_weight_by_layer": (
                    {
                        str(layer_idx): layer_config.get("trajectory_profile_fc1_weight_used")
                        for layer_idx, layer_config in layer_configs.items()
                    }
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "trajectory_profile_fc2_weight_by_layer": (
                    {
                        str(layer_idx): layer_config.get("trajectory_profile_fc2_weight_used")
                        for layer_idx, layer_config in layer_configs.items()
                    }
                    if args.expert_assignment == "trajectory_profile_kmeans" else None
                ),
                "router_init": args.router_init,
                "router_bias": args.router_bias or args.router_force_bias or args.router_balance_calib != "none",
                "router_fit_bias": args.router_bias,
                "router_force_bias": args.router_force_bias,
                "router_calib_max_tokens": (
                    args.router_calib_max_tokens
                    if args.router_init in {"calibrated_energy", "trajectory_energy"}
                    else None
                ),
                "router_ridge_lambda": (
                    args.router_ridge_lambda
                    if args.router_init in {"calibrated_energy", "trajectory_energy"}
                    else None
                ),
                "router_target_transform": (
                    args.router_target_transform
                    if args.router_init in {"calibrated_energy", "trajectory_energy"}
                    else None
                ),
                "router_target_metric": (
                    args.router_target_metric
                    if args.router_init in {"calibrated_energy", "trajectory_energy"}
                    else None
                ),
                "trajectory_router_nsamples": (
                    (args.trajectory_router_nsamples or args.nsamples)
                    if args.router_init == "trajectory_energy" else None
                ),
                "trajectory_router_batch_size": (
                    (args.trajectory_router_batch_size or args.batch_size)
                    if args.router_init == "trajectory_energy" else None
                ),
                "trajectory_router_top_k": (
                    args.trajectory_router_top_k if args.router_init == "trajectory_energy" else None
                ),
                "trajectory_router_top_p": (
                    args.trajectory_router_top_p if args.router_init == "trajectory_energy" else None
                ),
                "trajectory_router_stage_weight": (
                    args.trajectory_router_stage_weight if args.router_init == "trajectory_energy" else None
                ),
                "router_balance_calib": args.router_balance_calib,
                "router_balance_strength": (
                    args.router_balance_strength
                    if args.router_balance_calib != "none" else None
                ),
                "router_balance_max_abs_bias": (
                    args.router_balance_max_abs_bias
                    if args.router_balance_calib != "none" else None
                ),
                "router_balance_delta_linf_cap": (
                    args.router_balance_delta_linf_cap
                    if args.router_balance_calib != "none" else None
                ),
                "router_balance_nsamples": (
                    (
                        args.router_balance_nsamples
                        or args.trajectory_router_nsamples
                        or args.nsamples
                    )
                    if args.router_balance_calib == "trajectory" else None
                ),
                "router_balance_batch_size": (
                    (
                        args.router_balance_batch_size
                        or args.trajectory_router_batch_size
                        or args.batch_size
                    )
                    if args.router_balance_calib == "trajectory" else None
                ),
                "router_balance_top_k": (
                    (args.router_balance_top_k or args.trajectory_router_top_k)
                    if args.router_balance_calib == "trajectory" else None
                ),
                "router_balance_top_p": (
                    (
                        args.router_balance_top_p
                        if args.router_balance_top_p is not None
                        else args.trajectory_router_top_p
                    )
                    if args.router_balance_calib == "trajectory" else None
                ),
                "router_balance_stage_weight": (
                    args.router_balance_stage_weight
                    if args.router_balance_calib == "trajectory" else None
                ),
                "router_balance_label_sampling": (
                    args.router_balance_label_sampling
                    if args.router_balance_calib == "trajectory" else None
                ),
                "router_balance_target_metric": (
                    args.router_balance_target_metric
                    if args.router_balance_calib == "trajectory" else None
                ),
                "router_balance_target_transform": (
                    args.router_balance_target_transform
                    if (
                        args.router_balance_calib == "trajectory"
                        and args.router_balance_target_metric != "uniform"
                    ) else None
                ),
                "router_balance_target_mix_uniform": (
                    args.router_balance_target_mix_uniform
                    if (
                        args.router_balance_calib == "trajectory"
                        and args.router_balance_target_metric != "uniform"
                    ) else None
                ),
                "router_balance_stats": router_balance_stats if args.router_balance_calib != "none" else None,
            },
        },
        args.output_path,
    )
    print("Done.")


if __name__ == "__main__":
    main()
