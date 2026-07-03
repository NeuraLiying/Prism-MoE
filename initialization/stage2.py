                
import os
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F                      
from tqdm import tqdm 

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = Path(script_dir).resolve().parents[0]
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from common.path_utils import add_var_root

from var_sub_utils import (
    collect_dense_ffn_io_stats_for_shared,
    collect_dense_ffn_io_stats_for_experts,          
    collect_expert_sequential_cache,                
    solve_ridge_from_stats,
    solve_delta_ridge,
    solve_residual_delta_ridge,
    load_var_weights_into_model,
    ensure_all_layers_are_moe,
    verify_out_bias_consistency,
)
from var_sub_model import VARD2MFFN


def _subtract_residual_delta_in_chunks(
    r_cur: torch.Tensor,
    token_idx: torch.Tensor,
    h_e: torch.Tensor,
    delta_w: torch.Tensor,
    chunk_tokens: int,
) -> None:

       
    if chunk_tokens is None or chunk_tokens <= 0:
        chunk_tokens = h_e.shape[0]
    for start in range(0, h_e.shape[0], int(chunk_tokens)):
        end = min(start + int(chunk_tokens), h_e.shape[0])
        idx = token_idx[start:end]
        delta_y = F.linear(h_e[start:end], delta_w)
        r_cur.index_copy_(0, idx, r_cur.index_select(0, idx) - delta_y)


def _build_models(args, device):
\
\
\
\
       
    add_var_root(getattr(args, "var_root", None))
    from models import build_vae_var

    patch_nums = tuple(int(x) for x in args.patch_nums.split(","))

    vae, var = build_vae_var(
        V=args.codebook_size,
        Cvae=args.cvae_dim,
        ch=args.vae_ch,
        share_quant_resi=args.share_quant_resi,
        device=device,
        patch_nums=patch_nums,
        num_classes=args.num_classes,
        depth=args.depth,
        shared_aln=getattr(args, "shared_aln", False),
    )
    return vae, var


def load_var_weights_strict(var_model: torch.nn.Module, ckpt_path: str) -> None:
       
    ckpt_data = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt_data, dict) and "trainer" in ckpt_data and "var_wo_ddp" in ckpt_data["trainer"]:
        ckpt_sd = ckpt_data["trainer"]["var_wo_ddp"]
    elif isinstance(ckpt_data, dict) and "var_wo_ddp" in ckpt_data:
        ckpt_sd = ckpt_data["var_wo_ddp"]
    elif isinstance(ckpt_data, dict) and "state_dict" in ckpt_data:
        ckpt_sd = ckpt_data["state_dict"]
    else:
        ckpt_sd = ckpt_data
    
                      
    missing_keys, unexpected_keys = var_model.load_state_dict(ckpt_sd, strict=True)
    
    if missing_keys:
        raise RuntimeError(
            f"Failed to load checkpoint: {len(missing_keys)} keys are missing in model.\n"
            f"First 20 missing keys:\n" + "\n".join(f"  - {k}" for k in missing_keys[:20]) +
            (f"\n  ... and {len(missing_keys) - 20} more" if len(missing_keys) > 20 else "")
        )
    
    if unexpected_keys:
        print(f"  Warning: {len(unexpected_keys)} unexpected keys in checkpoint (ignored)")
        if len(unexpected_keys) <= 10:
            for k in unexpected_keys:
                print(f"    - {k}")
        else:
            for k in unexpected_keys[:10]:
                print(f"    - {k}")
            print(f"    ... and {len(unexpected_keys) - 10} more")


def _prepare_calibration_data_with_tokens(
    vae,
    device: torch.device,
    num_classes: int,
    num_calib: int,
    calib_bs: int,
    seed: int,
    use_images: bool = False,
    image_dir: Optional[str] = None,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
\
\
\
\
\
\
\
       
                                                                 
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    
    pairs: List[Tuple[torch.Tensor, torch.Tensor]] = []
    
    if use_images and image_dir is not None:
                          
        print("Loading and encoding real ImageNet images...")
        
                                                 
        var_root_dir = str(add_var_root(getattr(_prepare_calibration_data_with_tokens, "_var_root", None)))
        
        try:
            from utils.data import build_dataset
        except ImportError:
                            
            possible_paths = [
                os.path.join(var_root_dir, 'utils'),
                os.path.join(os.path.dirname(var_root_dir), 'utils'),
            ]
            for path in possible_paths:
                if os.path.exists(os.path.join(path, 'data.py')):
                    sys.path.insert(0, os.path.dirname(path))
                    from utils.data import build_dataset
                    break
            else:
                raise ImportError(
                    f"Cannot find utils.data module. "
                    f"var_root_dir: {var_root_dir}, "
                    f"Please ensure the selected VAR runtime root contains utils/data.py."
                )
        
        num_classes_ds, train_set, val_set = build_dataset(
            data_path=image_dir,
            final_reso=256,
            hflip=False,
            mid_reso=1.125
        )
        
        dataset = val_set
        
                         
        print(f"  Random sampling: sampling {num_calib} samples from {len(dataset)} images (seed={seed})")
        
        if num_calib <= len(dataset):
                                      
                                                  
            indices = torch.randperm(len(dataset), generator=g)[:num_calib]
        else:
                                       
            indices = torch.randperm(len(dataset), generator=g)
                                
            n_repeats = (num_calib + len(indices) - 1) // len(indices)
            indices = indices.repeat(n_repeats)[:num_calib]
        
                         
        sampled_labels = [dataset[int(idx)][1] for idx in indices]
        unique_classes = len(set(sampled_labels))
        print(f"  Selected {len(indices)} samples covering {unique_classes}/{num_classes_ds} classes")
        
        n_batches = (len(indices) + calib_bs - 1) // calib_bs
        for i in range(n_batches):
            batch_indices = indices[i * calib_bs:(i + 1) * calib_bs]
            images = []
            labels = []
            
            for idx in batch_indices:
                img, label = dataset[int(idx)]
                images.append(img)
                labels.append(label)
            
            images = torch.stack(images).to(device)                    
            labels = torch.tensor(labels).to(device)
            
                                           
            gt_idx_Bl = vae.img_to_idxBl(images)                      
            tokens_BLCv = vae.quantize.idxBl_to_var_input(gt_idx_Bl)                
            
            pairs.append((labels, tokens_BLCv))
            
            if (i + 1) * calib_bs % 64 == 0:
                print(f"  Processed {(i + 1) * calib_bs}/{len(indices)} images")
        
        print(f"✓ Prepared {len(pairs)} batches with real tokens")
    else:
                                        
        print("⚠️  Using randomly generated tokens (not recommended, use --use_images for better results)")
        n_batches = (num_calib + calib_bs - 1) // calib_bs
        for i in range(n_batches):
            B = min(calib_bs, num_calib - i * calib_bs)
                                                                           
                                                                          
                            
            label_B = torch.randint(0, num_classes, (B,), generator=g).to(device=device)
                                
            tokens_BLCv = torch.zeros(B, 679, vae.Cvae, device=device)       
            pairs.append((label_B, tokens_BLCv))
    
    return pairs


def _infer_moe_shapes_from_checkpoint(
    moe_ckpt_path: str,
    num_layers: int,
) -> Dict[int, Dict[str, int]]:

       
    ckpt_data = torch.load(moe_ckpt_path, map_location="cpu")
    if isinstance(ckpt_data, dict) and "var_wo_ddp" in ckpt_data:
        state_dict = ckpt_data["var_wo_ddp"]
    else:
        state_dict = ckpt_data
    
    layer_shapes = {}
    
    for li in range(num_layers):
                                    
        shared_fc1_key = f"blocks.{li}.ffn.shared.fc1.weight"
        expert0_fc1_key = f"blocks.{li}.ffn.experts.0.fc1.weight"
        gate_proj_key = f"blocks.{li}.ffn.gate.proj.weight"
        gate_bias_key = f"blocks.{li}.ffn.gate.proj.bias"
        
        if shared_fc1_key not in state_dict:
            continue
        
        shared_fc1_shape = state_dict[shared_fc1_key].shape
        shared_hidden = shared_fc1_shape[0]                                
        
        if expert0_fc1_key in state_dict:
            expert0_fc1_shape = state_dict[expert0_fc1_key].shape
            expert_hidden = expert0_fc1_shape[0]                                
        else:
            expert_hidden = None
        
        if gate_proj_key in state_dict:
            gate_proj_shape = state_dict[gate_proj_key].shape
            n_experts = gate_proj_shape[0]                            
        else:
            n_experts = None
        
        layer_shapes[li] = {
            'shared_hidden': shared_hidden,
            'expert_hidden': expert_hidden,
            'n_experts': n_experts,
            'router_bias': gate_bias_key in state_dict,
        }
    
    return layer_shapes


def _replace_ffn_with_moe_from_shapes(
    var_model,
    layer_shapes: Dict[int, Dict[str, int]],
    moe_config: dict,
    device: torch.device,
) -> None:
\
\
\
\
       
                                          
    if hasattr(moe_config, 'get'):
        topk = moe_config.get('topk', 8)
        router_temp = moe_config.get('router_temp', 1.0)
        norm_topk_prob = moe_config.get('norm_topk_prob', False)
        hard_mode = moe_config.get('hard_mode', True)
    else:
        topk = 8
        router_temp = 1.0
        norm_topk_prob = False
        hard_mode = True
    
    for layer_idx, block in enumerate(var_model.blocks):
        if not (hasattr(block, 'ffn') and hasattr(block.ffn, 'fc1') and hasattr(block.ffn, 'fc2')):
            continue
        
        if layer_idx not in layer_shapes:
            raise RuntimeError(
                f"Layer {layer_idx}: Cannot infer MoE shapes from checkpoint. "
                f"Please ensure checkpoint contains MoE weights."
            )
        
        shapes = layer_shapes[layer_idx]
        shared_hidden = shapes['shared_hidden']
        expert_hidden = shapes['expert_hidden']
        n_experts = shapes['n_experts']
        router_bias = bool(shapes.get('router_bias', False))
        
        if expert_hidden is None or n_experts is None:
            raise RuntimeError(
                f"Layer {layer_idx}: Cannot infer expert_hidden or n_experts from checkpoint. "
                f"Found shapes: {shapes}"
            )
        
        ffn = block.ffn
        C = ffn.fc1.in_features         
        
                         
        drop_rate = 0.0
        if hasattr(ffn, 'drop') and isinstance(ffn.drop, torch.nn.Dropout):
            drop_rate = float(ffn.drop.p)
        
                                            
        moe_ffn = VARD2MFFN(
            in_features=C,
            shared_hidden=shared_hidden,
            expert_hidden=expert_hidden,
            n_experts=n_experts,
            topk=topk,
            drop=drop_rate,
            hard_mode=hard_mode,
            norm_topk_prob=norm_topk_prob,
            router_temp=router_temp,
            router_bias=router_bias,
        ).to(device)
        
                
        block.ffn = moe_ffn


@torch.no_grad()
def _force_align_out_bias(teacher: torch.nn.Module, student: torch.nn.Module) -> None:
\
\
\
\
\
       
    n_layers = len(teacher.blocks)
    aligned_count = 0
    
    for li in range(n_layers):
        ffn_t = teacher.blocks[li].ffn
        ffn_s = student.blocks[li].ffn
        
        if not isinstance(ffn_s, VARD2MFFN):
            continue
        
                                
        teacher_bias = None
        if hasattr(ffn_t, 'fc2') and hasattr(ffn_t.fc2, 'bias') and ffn_t.fc2.bias is not None:
            teacher_bias = ffn_t.fc2.bias.detach().clone()
        
        if teacher_bias is not None:
                                                      
            ffn_s.out_bias.data.copy_(teacher_bias.to(ffn_s.out_bias.device, dtype=ffn_s.out_bias.dtype))
            aligned_count += 1
        else:
                                                      
            ffn_s.out_bias.data.zero_()
    
    print(f"  ✓ Force-aligned out_bias for {aligned_count}/{n_layers} layers (from teacher fc2.bias)")


@torch.no_grad()
def stage2_refine_fc2_only(args):
\
\
\
\
\
\
\
\
\
       
    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vae_t, teacher = _build_models(args, device)
    vae_s, student = _build_models(args, device)

    if args.vae_ckpt:
        print(f"\n[Loading] Loading VAE weights: {args.vae_ckpt}")
        vae_sd = torch.load(args.vae_ckpt, map_location="cpu")
        vae_t.load_state_dict(vae_sd, strict=True)
        vae_s.load_state_dict(vae_sd, strict=True)

                        
    print("\n[Loading] Loading teacher (dense) weights...")
    load_var_weights_into_model(teacher, args.dense_ckpt)
    
                                                
    print("\n[Loading] Loading MoE checkpoint config...")
    moe_ckpt_data = torch.load(args.moe_ckpt, map_location="cpu")
    moe_config = {}
    if isinstance(moe_ckpt_data, dict):
        if 'config' in moe_ckpt_data:
            moe_config = moe_ckpt_data['config']
                               
        config_path = args.moe_ckpt.replace('.pth', '_config.json')
        if os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                moe_config.update(json.load(f))
    
                                           
    print("\n[Inference] Inferring MoE shapes from checkpoint...")
    n_layers = len(student.blocks)
    layer_shapes = _infer_moe_shapes_from_checkpoint(args.moe_ckpt, n_layers)
    
    if len(layer_shapes) == 0:
        raise RuntimeError(
            "Cannot infer MoE shapes from checkpoint. "
            "Please ensure checkpoint contains MoE weights (e.g., blocks.0.ffn.shared.fc1.weight)."
        )
    
    print(f"  ✓ Inferred shapes for {len(layer_shapes)}/{n_layers} layers")
                  
    if 0 in layer_shapes:
        shapes = layer_shapes[0]
        print(f"  Example (Layer 0): shared_hidden={shapes['shared_hidden']}, "
              f"expert_hidden={shapes['expert_hidden']}, n_experts={shapes['n_experts']}, "
              f"router_bias={shapes.get('router_bias', False)}")
    
                                           
    print("\n[Setup] Creating MoE FFN modules with inferred shapes...")
    _replace_ffn_with_moe_from_shapes(student, layer_shapes, moe_config, device)
    
                                     
    print("\n[Loading] Loading MoE weights from checkpoint (STRICT MODE)...")
    print(f"  Checkpoint: {args.moe_ckpt}")
    try:
        load_var_weights_strict(student, args.moe_ckpt)
        print("  ✓ All weights loaded successfully (strict=True)")
    except RuntimeError as e:
        print(f"\n❌ Failed to load weights in strict mode:")
        print(f"   {e}")
        print(f"\n   This usually means:")
        print(f"   1. Model architecture doesn't match checkpoint")
        print(f"   2. Some MoE parameters (e.g., norm_topk_prob, router_temp) differ")
        print(f"   3. Checkpoint format is incorrect")
        raise

    teacher.eval().to(device)
    student.eval().to(device)
    vae_t.eval().to(device)
    vae_s.eval().to(device)

                                   
    ensure_all_layers_are_moe(student, moe_cls=VARD2MFFN)
    
                             
    print("\n[Verification] Verifying loaded MoE weights...")
    all_valid = True
    for li in layer_shapes.keys():
        ffn_s = student.blocks[li].ffn
        if not isinstance(ffn_s, VARD2MFFN):
            continue
        
                                 
        w_shared_fc2 = ffn_s.shared.fc2.weight.data
        w_max = w_shared_fc2.abs().max().item()
        w_mean = w_shared_fc2.abs().mean().item()
        
        if w_max < 1e-6:
            print(f"  ⚠️  Layer {li}: shared.fc2.weight is all zeros (max={w_max:.2e})")
            all_valid = False
        else:
            print(f"  ✓ Layer {li}: shared.fc2.weight loaded (max={w_max:.4f}, mean={w_mean:.4f})")
        
                    
        w_gate = ffn_s.gate.proj.weight.data
        if w_gate.abs().max().item() < 1e-6:
            print(f"  ⚠️  Layer {li}: gate.proj.weight is all zeros")
            all_valid = False
        
                          
        if len(ffn_s.experts) > 0:
            w_exp_fc2 = ffn_s.experts[0].fc2.weight.data
            if w_exp_fc2.abs().max().item() < 1e-6:
                print(f"  ⚠️  Layer {li}: experts[0].fc2.weight is all zeros")
                all_valid = False
    
    if not all_valid:
        import warnings
        warnings.warn(
            "\n⚠️  Some MoE weights appear to be invalid (zeros). "
            "This may indicate loading failure despite strict=True.\n"
        )
    
                                               
    print("\n[Alignment] Force-aligning out_bias from teacher...")
    _force_align_out_bias(teacher, student)
    
                                            
    print("\n[Verification] Verifying out_bias consistency...")
    all_consistent = True
    for li in range(len(student.blocks)):
        if not verify_out_bias_consistency(teacher, student, li, tolerance=1e-5, verbose=True):
            all_consistent = False
    
    if not all_consistent:
        import warnings
        warnings.warn(
            "\n⚠️  Some layers still have out_bias inconsistency after force alignment. "
            "This should not happen. Please check the code.\n"
        )
    else:
        print("  ✓ All layers: out_bias consistency verified.\n")

                              
    print("\n[Calibration] Preparing calibration data...")
    if hasattr(args, 'use_images') and args.use_images:
        calib_pairs = _prepare_calibration_data_with_tokens(
            vae=vae_s,
            device=device,
            num_classes=args.num_classes,
            num_calib=args.num_calib,
            calib_bs=args.calib_bs,
            seed=args.calib_seed,
            use_images=True,
            image_dir=getattr(args, 'imagenet_dir', None),
        )
    else:
                              
        print("⚠️  Warning: Using simplified calibration (random tokens). "
              "For better results, use --use_images with --imagenet_dir")
        calib_pairs = _prepare_calibration_data_with_tokens(
            vae=vae_s,
            device=device,
            num_classes=args.num_classes,
            num_calib=args.num_calib,
            calib_bs=args.calib_bs,
            seed=args.calib_seed,
            use_images=False,
        )

    n_layers = len(teacher.blocks)
    
             
    print(f"\nStarting Stage II refinement for {n_layers} layers")
    print(f"Calibration batches: {len(calib_pairs)}")
    stage2_calib_mode = getattr(args, "stage2_calib_mode", "forward")
    shared_calib_mode = getattr(args, "stage2_shared_calib_mode", "inherit")
    expert_calib_mode = getattr(args, "stage2_expert_calib_mode", "inherit")
    if shared_calib_mode == "inherit":
        shared_calib_mode = stage2_calib_mode
    if expert_calib_mode == "inherit":
        expert_calib_mode = stage2_calib_mode
    use_trajectory_shared = shared_calib_mode == "trajectory"
    use_trajectory_expert = expert_calib_mode == "trajectory"

    if use_trajectory_shared or use_trajectory_expert:
        print(
            f"  [Mode] Autoregressive student trajectory rollout "
            f"(top_k={args.trajectory_top_k}, top_p={args.trajectory_top_p})"
        )
    else:
        print(f"  [Mode] Forward path (teacher forcing / provided tokens)")
    print(f"    - Shared calibration: {shared_calib_mode}")
    print(f"    - Expert calibration: {expert_calib_mode}")
    print(f"  [Strategy] Two-Stage Compensation:")
    print(f"    Stage 1: Shared fc2 delta ridge")
    refine_experts = getattr(args, 'refine_experts', False)
    if refine_experts:
        print(f"    Stage 2: Experts fc2 sequential residual refinement (enabled)")
        print(f"      - Ridge lambda (experts): {getattr(args, 'ridge_lambda_expert', 20.0)}")
        print(f"      - Max delta norm (experts): {getattr(args, 'max_delta_norm_expert', 0.02):.2%}")
        print(f"      - Min tokens per expert: {getattr(args, 'min_tokens_per_expert', 4096)}")
    else:
        print(f"    Stage 2: Experts fc2 (disabled, use --refine_experts to enable)")
    print(f"    - Max delta norm (shared): {args.max_delta_norm:.2%}")
    print(f"    - Ridge lambda (shared): {args.ridge_lambda_shared} (relative to mean(diag(A)))")
    
                   
    layer_progress = tqdm(range(n_layers), desc="Overall Progress", position=0, leave=True)
    
    for li in layer_progress:
        ffn_s = student.blocks[li].ffn
        if not isinstance(ffn_s, VARD2MFFN):
            raise RuntimeError(
                f"[Stage2] student layer {li} ffn is not VARD2MFFN. "
                f"Found: {type(ffn_s)}. Ensure your moe_ckpt is a full-layer MoE checkpoint."
            )

        layer_progress.set_description(f"Layer {li}/{n_layers-1}")
        print(f"\n[Stage2 Refine] Layer {li}/{n_layers-1}")

                                                      
        print(f"  [Stage 1] Shared fc2 delta ridge")
        stats_shared = collect_dense_ffn_io_stats_for_shared(
            teacher=teacher,
            student=student,
            layer_idx=li,
            calib_pairs=calib_pairs,
            cfg=args.cfg,
            dtype=torch.float32,
            max_tokens_per_call=args.max_tokens_per_call,
            use_student_trajectory=use_trajectory_shared,
            trajectory_seed=args.trajectory_seed,
            trajectory_top_k=args.trajectory_top_k,
            trajectory_top_p=args.trajectory_top_p,
        )
        
        W_old_shared = ffn_s.shared.fc2.weight.data.clone()
        
        diag_mean = stats_shared["A"].diag().mean().item()
        adaptive_lambda = args.ridge_lambda_shared * diag_mean
        print(f"    Adaptive lambda: {adaptive_lambda:.2e} "
              f"(base_coeff={args.ridge_lambda_shared}, diag_mean={diag_mean:.2e})")
        
        W_shared = solve_delta_ridge(
            A=stats_shared["A"],
            B=stats_shared["B"],
            W_old=W_old_shared,
            ridge_lambda=args.ridge_lambda_shared,
            max_delta_norm=args.max_delta_norm,
            layer_name=f"L{li}_shared",
            use_adaptive_lambda=True,
        )
        
        delta_shared = W_shared - W_old_shared
        delta_norm_shared = delta_shared.norm().item()
        old_norm_shared = W_old_shared.norm().item()
        delta_ratio_shared = delta_norm_shared / old_norm_shared if old_norm_shared > 1e-6 else 0.0
        
        print(f"    ✓ Shared fc2 updated: shape {W_shared.shape}")
        print(f"      Delta norm: {delta_norm_shared:.4f}, Old norm: {old_norm_shared:.4f}, Ratio: {delta_ratio_shared:.4%}")
        
                                                    
        ffn_s.shared.fc2.weight.data.copy_(W_shared.to(ffn_s.shared.fc2.weight.dtype))

                                                           
        if refine_experts:
            print(f"  [Stage 2] Experts fc2 sequential residual refinement (K={ffn_s.topk})")
            
            min_tokens = getattr(args, 'min_tokens_per_expert', 4096)
            ridge_lambda_expert = getattr(args, 'ridge_lambda_expert', 20.0)
            max_delta_norm_expert = getattr(args, 'max_delta_norm_expert', 0.02)
            relative_expert_trust = getattr(args, 'relative_expert_trust', True)
            
                                 
            cache = collect_expert_sequential_cache(
                teacher=teacher,
                student=student,
                layer_idx=li,
                calib_pairs=calib_pairs,
                cfg=args.cfg,
                dtype=torch.float32,
                max_tokens_per_call=args.max_tokens_per_call,
                use_student_trajectory=use_trajectory_expert,
                trajectory_seed=args.trajectory_seed,
                trajectory_top_k=args.trajectory_top_k,
                trajectory_top_p=args.trajectory_top_p,
            )
            
            indices_sorted = cache["indices_sorted"]                
            h_cache_all = cache["h_cache"]                                     
            x2_all = cache["x2_all"]                
            r_cur = cache["r0_all"].clone()                     
            
            K = ffn_s.topk
            E = ffn_s.n_experts
            H = ffn_s.expert_hidden
            C = ffn_s.in_features
            
            updated_experts_total = 0
            skipped_experts_total = 0
            
                                                        
            for s in range(K):
                print(f"    [Rank {s}/{K-1}] Updating experts with current residual...")
                
                                                        
                decay = 1.2 ** s        
                rank_lambda = ridge_lambda_expert * decay
                rank_max_delta = max_delta_norm_expert / decay
                
                                                 
                e_id_per_token = indices_sorted[:, s]             
                
                                            
                experts_in_rank = torch.unique(e_id_per_token).tolist()
                
                updated_in_rank = 0
                skipped_in_rank = 0
                
                                             
                for e in experts_in_rank:
                                                  
                    mask_s = (e_id_per_token == e)             
                    if not mask_s.any():
                        continue
                    
                    token_idx_s = mask_s.nonzero(as_tuple=True)[0]          
                    token_count = int(token_idx_s.numel())
                    
                                                  
                                                         
                    min_tokens_for_rank = max(min_tokens, int(H * (0.5 + 0.5 * s / max(K-1, 1))))
                    
                    if token_count < min_tokens_for_rank:
                        skipped_in_rank += 1
                        if skipped_in_rank <= 2:
                            print(f"      [Skip] Expert {e}: token_count={token_count} < {min_tokens_for_rank} "
                                  f"(rank {s}, min_tokens={min_tokens}, H={H})")
                        continue
                    
                                                
                    if e not in h_cache_all:
                        skipped_in_rank += 1
                        continue
                    
                    token_idx_cached, h_cached = h_cache_all[e]
                    
                                           
                                             
                    token_idx_cached_cpu = token_idx_cached.cpu().numpy() if hasattr(token_idx_cached.cpu(), 'numpy') else token_idx_cached.cpu().tolist()
                    token_idx_s_cpu = token_idx_s.cpu()
                    
                                         
                    if isinstance(token_idx_cached_cpu, torch.Tensor):
                        cached_dict = {int(idx.item()): i for i, idx in enumerate(token_idx_cached_cpu)}
                    else:
                        cached_dict = {int(idx): i for i, idx in enumerate(token_idx_cached_cpu)}
                    
                          
                    inter_list = []
                    idx_in_cached_list = []
                    idx_in_s_list = []
                    
                    for i, idx in enumerate(token_idx_s_cpu):
                        idx_val = int(idx.item())
                        if idx_val in cached_dict:
                            inter_list.append(idx_val)
                            idx_in_cached_list.append(cached_dict[idx_val])
                            idx_in_s_list.append(i)
                    
                    if len(inter_list) == 0:
                        skipped_in_rank += 1
                        continue
                    
                    idx_in_cached = torch.tensor(
                        idx_in_cached_list,
                        device=h_cached.device,
                        dtype=torch.long,
                    )
                    idx_in_s = torch.tensor(idx_in_s_list, device=token_idx_s.device, dtype=torch.long)
                    
                                                    
                    token_idx_selected = token_idx_s[idx_in_s]
                    target = r_cur[token_idx_selected].to(dtype=torch.float32)          
                    h_e = h_cached.index_select(0, idx_in_cached).to(
                        device=target.device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )          
                    
                                            
                                                             
                    A_e_s = h_e.t().mm(h_e)          
                    B_e_s = h_e.t().mm(target)                    
                    del target
                    
                                
                    W_old_e = ffn_s.experts[e].fc2.weight.data.clone()          

                                                  
                    diag_mean_e = A_e_s.diag().mean().item()
                    adaptive_lambda_e = rank_lambda * diag_mean_e

                    if relative_expert_trust:
                                                                        
                                                                          
                                                                             
                                                                       
                        Delta_e = solve_residual_delta_ridge(
                            A=A_e_s,
                            B=B_e_s,
                            reference_W=W_old_e,
                            ridge_lambda=rank_lambda,
                            max_delta_norm=rank_max_delta,
                            layer_name=f"L{li}_rank{s}_exp{e}",
                            use_adaptive_lambda=True,
                        )
                    else:
                                                                          
                                                                       
                        zeros_W = torch.zeros_like(W_old_e)
                        Delta_e = solve_delta_ridge(
                            A=A_e_s,
                            B=B_e_s,
                            W_old=zeros_W,
                            ridge_lambda=rank_lambda,
                            max_delta_norm=rank_max_delta,
                            layer_name=f"L{li}_rank{s}_exp{e}",
                            use_adaptive_lambda=True,
                        )
                    
                                    
                    W_new_e = W_old_e + Delta_e
                    
                                        
                    ffn_s.experts[e].fc2.weight.data.copy_(W_new_e.to(ffn_s.experts[e].fc2.weight.dtype))
                    
                                                           
                                                                    
                    _subtract_residual_delta_in_chunks(
                        r_cur=r_cur,
                        token_idx=token_idx_selected,
                        h_e=h_e,
                        delta_w=Delta_e,
                        chunk_tokens=args.residual_update_chunk_tokens,
                    )
                    
                    updated_in_rank += 1
                    updated_experts_total += 1
                    
                    delta_norm_e = Delta_e.norm().item()                   
                    old_norm_e = W_old_e.norm().item()
                    delta_ratio_e = delta_norm_e / old_norm_e if old_norm_e > 1e-6 else 0.0
                    
                    if updated_in_rank <= 3:
                        print(f"      ✓ Expert {e}: token_count={token_count}, "
                              f"delta_ratio={delta_ratio_e:.4%}, lambda={adaptive_lambda_e:.2e} "
                              f"(rank_lambda={rank_lambda:.2f}, decay={decay:.2f})")

                    del A_e_s, B_e_s, h_e, Delta_e, W_old_e, W_new_e
                    if 'zeros_W' in locals():
                        del zeros_W
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                if skipped_in_rank > 2:
                    print(f"      ... (skipped {skipped_in_rank} experts with insufficient tokens)")
                print(f"    ✓ Rank {s}: Updated {updated_in_rank} experts "
                      f"(lambda={rank_lambda:.2f}, max_delta={rank_max_delta:.4f})")
            
            print(f"  ✓ Sequential refinement completed: {updated_experts_total} experts updated across {K} ranks")
        else:
            print(f"  [Skip] Experts fc2: parameters unchanged (use --refine_experts to enable)")

        layer_progress.update(1)

    layer_progress.close()

                         
    os.makedirs(os.path.dirname(args.save_refined_ckpt) if os.path.dirname(args.save_refined_ckpt) else ".", exist_ok=True)
    
                          
    first_ffn = student.blocks[0].ffn
    if isinstance(first_ffn, VARD2MFFN):
        if shared_calib_mode == expert_calib_mode == "forward":
            method_suffix = "Forward"
        elif shared_calib_mode == expert_calib_mode == "trajectory":
            method_suffix = "Trajectory"
        else:
            method_suffix = f"MixedShared{shared_calib_mode.title()}Expert{expert_calib_mode.title()}"
        method_name = (
            f"D2M_Plus_TwoStage_Sequential_DeltaRidge_{method_suffix}"
            if refine_experts
            else f"D2M_Plus_DeltaRidge_{method_suffix}"
        )
        
        save_config = {
            "nexperts": first_ffn.n_experts,
            "topk": first_ffn.topk,
            "shared_ratio": first_ffn.shared_hidden / (first_ffn.shared_hidden + first_ffn.expert_hidden * first_ffn.n_experts),
            "hard_mode": first_ffn.hard_mode,
            "norm_topk_prob": first_ffn.norm_topk_prob,
            "router_temp": getattr(first_ffn.gate, 'router_temp', 1.0),
            "router_bias": first_ffn.gate.proj.bias is not None,
            "method": method_name,
            "use_images": getattr(args, 'use_images', False),
            "ridge_lambda_shared": args.ridge_lambda_shared,
            "max_delta_norm": args.max_delta_norm,
            "stage2_calib_mode": stage2_calib_mode,
            "stage2_shared_calib_mode": shared_calib_mode,
            "stage2_expert_calib_mode": expert_calib_mode,
            "trajectory_seed": getattr(args, "trajectory_seed", None) if (use_trajectory_shared or use_trajectory_expert) else None,
            "trajectory_top_k": getattr(args, "trajectory_top_k", None) if (use_trajectory_shared or use_trajectory_expert) else None,
            "trajectory_top_p": getattr(args, "trajectory_top_p", None) if (use_trajectory_shared or use_trajectory_expert) else None,
        }
        
        if refine_experts:
            save_config.update({
                "ridge_lambda_expert": getattr(args, 'ridge_lambda_expert', 20.0),
                "max_delta_norm_expert": getattr(args, 'max_delta_norm_expert', 0.02),
                "min_tokens_per_expert": getattr(args, 'min_tokens_per_expert', 4096),
                "relative_expert_trust": getattr(args, 'relative_expert_trust', True),
            })
    else:
                                  
        save_config = moe_config.copy() if moe_config else {}
        save_config["method"] = "D2M_Plus_DeltaRidge_Forward"
                                               
        if 'n_experts' in save_config and 'nexperts' not in save_config:
            save_config['nexperts'] = save_config.pop('n_experts')

                                                                          
                                                                            
                                                        
    if moe_config:
        for key in (
            "loss_mode",
            "nsamples",
            "calib_seed",
            "use_two_stage",
            "candidate_multiplier",
            "shared_second_score",
            "shared_selection_mode",
            "shared_importance_weight",
            "contribution_max_tokens",
            "contribution_transform",
            "trajectory_shared_score_mode",
            "expert_assignment",
            "kmeans_iters",
            "kmeans_restarts",
            "trajectory_profile_nsamples",
            "trajectory_profile_batch_size",
            "trajectory_profile_top_k",
            "trajectory_profile_top_p",
            "trajectory_profile_max_tokens",
            "trajectory_profile_transform",
            "trajectory_profile_position_bins",
            "trajectory_profile_feature_weight",
            "trajectory_profile_fc1_weight",
            "trajectory_profile_fc2_weight",
            "trajectory_profile_fc2_weight_schedule",
            "trajectory_profile_fc2_weight_min",
            "trajectory_profile_fc2_weight_max",
            "trajectory_profile_fc2_weight_start",
            "trajectory_profile_fc2_weight_end",
            "trajectory_profile_fc2_weight_by_layer",
            "trajectory_profile_stage_onehot_weight",
            "trajectory_profile_fc1_weight_schedule",
            "trajectory_profile_fc1_weight_min",
            "trajectory_profile_fc1_weight_max",
            "trajectory_profile_fc1_weight_start",
            "trajectory_profile_fc1_weight_end",
            "trajectory_profile_fc1_weight_by_layer",
            "router_init",
            "router_bias",
            "router_fit_bias",
            "router_force_bias",
            "router_calib_max_tokens",
            "router_ridge_lambda",
            "router_target_transform",
            "router_target_metric",
            "trajectory_router_nsamples",
            "trajectory_router_batch_size",
            "trajectory_router_top_k",
            "trajectory_router_top_p",
            "trajectory_router_stage_weight",
            "router_balance_calib",
            "router_balance_strength",
            "router_balance_max_abs_bias",
            "router_balance_delta_linf_cap",
            "router_balance_target_metric",
            "router_balance_target_transform",
            "router_balance_target_mix_uniform",
            "router_balance_nsamples",
            "router_balance_batch_size",
            "router_balance_top_k",
            "router_balance_top_p",
            "router_balance_stage_weight",
            "router_balance_stats",
        ):
            if key in moe_config and key not in save_config:
                save_config[key] = moe_config[key]
    
    torch.save({
        "var_wo_ddp": student.state_dict(),
        "config": save_config
    }, args.save_refined_ckpt)
    
                       
    config_path = args.save_refined_ckpt.replace('.pth', '_config.json')
    import json
    with open(config_path, 'w') as f:
        json.dump(save_config, f, indent=2)
    
    print(f"\n[Stage2 Refine] Saved: {args.save_refined_ckpt}")
    print(f"[Stage2 Refine] Config saved: {config_path}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dense_ckpt", type=str, default="/home/liying/pretrained/model_zoo/var_d16.pth")
    ap.add_argument("--vae_ckpt", type=str, default="/home/liying/pretrained/model_zoo/vae_ch160v4096z32.pth")
    ap.add_argument("--moe_ckpt", type=str, required=True)
    ap.add_argument("--save_refined_ckpt", type=str, default="outputs/var_d2m_refined.pth")
    ap.add_argument("--var_root", type=str, default=None,
                    help="Path to the VAR runtime root. Defaults to $VAR_ROOT or this standalone project.")
    ap.add_argument("--device", type=str, default=None,
                    help="Torch device for refinement. Defaults to cuda if available, otherwise cpu.")

    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--patch_nums", type=str, default="1,2,3,4,5,6,8,10,13,16")
    ap.add_argument("--num_classes", type=int, default=1000)

    ap.add_argument("--codebook_size", type=int, default=4096)
    ap.add_argument("--cvae_dim", type=int, default=32)
    ap.add_argument("--vae_ch", type=int, default=160)
    ap.add_argument("--share_quant_resi", type=int, default=4)
    ap.add_argument("--shared_aln", action="store_true")

    ap.add_argument("--cfg", type=float, default=4.0, help="Not used in forward path, kept for compatibility")

    ap.add_argument("--num_calib", type=int, default=200)
    ap.add_argument("--calib_bs", type=int, default=4)
    ap.add_argument("--calib_seed", type=int, default=42)

                                                   
    ap.add_argument("--ridge_lambda_shared", type=float, default=10.0,
                    help="Ridge lambda coefficient for shared delta compensation (relative to mean(diag(A))). "
                         "Default: 10.0 (strong regularization for controlled delta)")
    ap.add_argument("--ridge_lambda_experts", type=float, default=0.01,
                    help="Ridge lambda coefficient (deprecated, use --ridge_lambda_expert instead)")

                 
    ap.add_argument("--max_delta_norm", type=float, default=0.05,
                    help="Maximum allowed weight change ratio for shared (e.g., 0.05 = 5%%). "
                         "If delta_W norm exceeds this ratio of W_old norm, it will be clipped. "
                         "Default: 0.05 (very conservative)")

    ap.add_argument("--max_tokens_per_call", type=int, default=0)
    ap.add_argument("--residual_update_chunk_tokens", type=int, default=32768,
                    help="Chunk size for Stage II expert residual updates. "
                         "This preserves the ridge solve but avoids large [tokens, C] "
                         "temporary allocations when one expert owns many tokens.")
    ap.add_argument("--stage2_calib_mode", type=str, default="forward",
                    choices=["forward", "trajectory"],
                    help="Calibration inputs for Stage II ridge stats. "
                         "'forward' preserves the legacy teacher-forcing/token path; "
                         "'trajectory' rolls out the current MoE student autoregressively "
                         "and matches dense teacher FFN outputs at those trajectory states.")
    ap.add_argument("--stage2_shared_calib_mode", type=str, default="inherit",
                    choices=["inherit", "forward", "trajectory"],
                    help="Calibration inputs for the shared-fc2 Stage II solve. "
                         "Default 'inherit' follows --stage2_calib_mode.")
    ap.add_argument("--stage2_expert_calib_mode", type=str, default="inherit",
                    choices=["inherit", "forward", "trajectory"],
                    help="Calibration inputs for the expert-fc2 Stage II solve. "
                         "Default 'inherit' follows --stage2_calib_mode.")
    ap.add_argument("--trajectory_seed", type=int, default=42,
                    help="Base seed for --stage2_calib_mode trajectory rollouts.")
    ap.add_argument("--trajectory_top_k", type=int, default=900,
                    help="Sampling top-k for --stage2_calib_mode trajectory.")
    ap.add_argument("--trajectory_top_p", type=float, default=0.96,
                    help="Sampling top-p for --stage2_calib_mode trajectory.")

                   
    ap.add_argument("--use_images", action="store_true",
                    help="Use real ImageNet images for calibration (pre-encode tokens). Recommended for best results.")
    ap.add_argument("--imagenet_dir", type=str, default=None,
                    help="Path to ImageNet root directory (required if --use_images is set)")

                       
    ap.add_argument("--refine_experts", action="store_true",
                    help="Enable Stage 2: sequential residual refinement for experts fc2")
    ap.add_argument("--ridge_lambda_expert", type=float, default=20.0,
                    help="Ridge lambda coefficient for experts delta compensation. "
                         "Default: 20.0 (stronger regularization than shared)")
    ap.add_argument("--max_delta_norm_expert", type=float, default=0.02,
                    help="Maximum allowed weight change ratio for experts (e.g., 0.02 = 2%%). "
                         "Default: 0.02 (more conservative than shared)")
    ap.add_argument("--min_tokens_per_expert", type=int, default=4096,
                    help="Minimum token count per expert to enable update. "
                         "Experts with fewer tokens will be skipped. Default: 4096")
    ap.add_argument("--legacy_absolute_expert_trust", action="store_true",
                    help="Use the legacy expert delta clipping behavior that clips pure residual deltas "
                         "against an absolute norm. Default uses a relative trust region based on each "
                         "expert's original fc2 norm.")

    args = ap.parse_args()
    args.relative_expert_trust = not args.legacy_absolute_expert_trust

    var_root = add_var_root(args.var_root)
    _prepare_calibration_data_with_tokens._var_root = str(var_root)
    print(f"Using VAR root: {var_root}")
    
          
    if args.use_images and args.imagenet_dir is None:
        raise ValueError("--imagenet_dir is required when --use_images is set")
    
    if args.max_delta_norm <= 0 or args.max_delta_norm > 1.0:
        raise ValueError(f"--max_delta_norm must be in (0, 1], got {args.max_delta_norm}")
    
    if getattr(args, 'refine_experts', False):
        if args.max_delta_norm_expert <= 0 or args.max_delta_norm_expert > 1.0:
            raise ValueError(f"--max_delta_norm_expert must be in (0, 1], got {args.max_delta_norm_expert}")
        if args.min_tokens_per_expert <= 0:
            raise ValueError(f"--min_tokens_per_expert must be > 0, got {args.min_tokens_per_expert}")
    
    stage2_refine_fc2_only(args)


if __name__ == "__main__":
    main()

