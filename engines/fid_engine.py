                      
                       

                                                                    

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common.path_utils import add_var_root
from models.var_d2m_model import VARD2MFFN


PATCH_NUMS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)


@dataclass
class GenerationStats:
    total_time: float = 0.0
    total_tokens: int = 0
    num_images: int = 0
    expert_counts: dict = None

    def __post_init__(self):
        if self.expert_counts is None:
            self.expert_counts = defaultdict(lambda: defaultdict(int))

    def add_generation(self, elapsed: float, tokens: int, images: int) -> None:
        self.total_time += elapsed
        self.total_tokens += tokens
        self.num_images += images

    def add_counts(self, layer_counts: Dict[int, torch.Tensor]) -> None:
        for layer_idx, counts in layer_counts.items():
            for expert_id, count in enumerate(counts.cpu().to(torch.long).tolist()):
                self.expert_counts[layer_idx][expert_id] += int(count)

    def as_dict(self) -> dict:
        expert_stats = {}
        for layer_idx, counts in sorted(self.expert_counts.items()):
            total = sum(counts.values())
            if total <= 0:
                continue
            expert_stats[f"layer_{layer_idx}"] = {
                "total_activations": total,
                "expert_counts": dict(counts),
                "expert_frequencies": {str(k): v / total for k, v in counts.items()},
            }
        return {
            "total_time": self.total_time,
            "num_images": self.num_images,
            "avg_time_per_image": self.total_time / self.num_images if self.num_images else 0.0,
            "avg_tokens_per_image": self.total_tokens / self.num_images if self.num_images else 0.0,
            "tokens_per_second": self.total_tokens / self.total_time if self.total_time else 0.0,
            "expert_activation_stats": expert_stats,
        }


def extract_state_and_config(path: str) -> Tuple[dict, dict]:
    ckpt = torch.load(path, map_location="cpu")
    config = {}
    if isinstance(ckpt, dict):
        if "config" in ckpt:
            config.update(ckpt["config"])
        if "args" in ckpt:
            config.setdefault("args", ckpt["args"])
        if "var_wo_ddp" in ckpt:
            return ckpt["var_wo_ddp"], config
        if "trainer" in ckpt and "var_wo_ddp" in ckpt["trainer"]:
            return ckpt["trainer"]["var_wo_ddp"], config
        if "state_dict" in ckpt:
            return ckpt["state_dict"], config
    return ckpt, config


def infer_config(state_dict: dict, config: dict, topk_override, routing_mode: str) -> dict:
    gate_key = "blocks.0.ffn.gate.proj.weight"
    shared_key = "blocks.0.ffn.shared.fc1.weight"
    expert_key = "blocks.0.ffn.experts.0.fc1.weight"
    n_experts = int(state_dict[gate_key].shape[0])
    shared_hidden = int(state_dict[shared_key].shape[0])
    expert_hidden = int(state_dict[expert_key].shape[0])
    total_hidden = shared_hidden + expert_hidden * n_experts
    topk = int(topk_override if topk_override is not None else config.get("topk", 2))
    if routing_mode == "checkpoint":
        hard_mode = bool(config.get("hard_mode", False))
    elif routing_mode == "hard":
        hard_mode = True
    elif routing_mode == "soft":
        hard_mode = False
    else:
        raise ValueError(f"Unknown routing mode: {routing_mode}")
    return {
        "nexperts": n_experts,
        "topk": topk,
        "shared_ratio": shared_hidden / float(total_hidden),
        "hard_mode": hard_mode,
        "router_bias": f"blocks.0.ffn.gate.proj.bias" in state_dict,
        "norm_topk_prob": bool(config.get("norm_topk_prob", True)),
        "router_temp": float(config.get("router_temp", 1.0)),
        "init_alpha": float(config.get("init_alpha", 0.1)),
        "delta_hidden_mult": float(config.get("delta_hidden_mult", 0.25)),
        "router_context_mode": str(config.get("router_context_mode", "none")),
        "router_context_init_alpha": float(config.get("router_context_init_alpha", 0.1)),
        "router_context_interaction_rank": int(config.get("router_context_interaction_rank", 0)),
        "router_context_interaction_init_alpha": float(config.get("router_context_interaction_init_alpha", 0.1)),
        "router_context_cosine": bool(config.get("router_context_cosine", False)),
        "router_context_cosine_init_alpha": float(config.get("router_context_cosine_init_alpha", 0.0)),
        "router_token_cosine": bool(config.get("router_token_cosine", False)),
        "router_token_cosine_init_alpha": float(config.get("router_token_cosine_init_alpha", 0.0)),
        "router_logit_mode": str(config.get("router_logit_mode", "linear")),
        "router_cosine_tau": float(config.get("router_cosine_tau", 10.0)),
        "router_capture_input_sample_tokens": 0,
    }


def load_model(args, device):
    add_var_root(args.var_root)
    from common import dist
    from models import build_vae_var

    state_dict, config = extract_state_and_config(args.moe_ckpt)
    moe_config = infer_config(state_dict, config, args.topk, args.routing_mode)
    vae, var = build_vae_var(
        V=4096,
        Cvae=32,
        ch=160,
        share_quant_resi=4,
        device=device,
        patch_nums=PATCH_NUMS,
        num_classes=args.model_num_classes,
        depth=args.model_depth,
        shared_aln=args.shared_aln,
        use_moe=True,
        moe_config=moe_config,
    )
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location="cpu"), strict=True)
    ret = var.load_state_dict(state_dict, strict=False)
    missing_bad = [
        k
        for k in ret.missing_keys
        if (
            ".ffn.gate.delta." not in k
            and ".ffn.gate.cond_proj." not in k
            and ".ffn.gate.stage_embed." not in k
            and ".ffn.gate.branch_embed." not in k
            and ".ffn.gate.cond_token_proj." not in k
            and ".ffn.gate.cond_context_proj." not in k
            and ".ffn.gate.cond_interaction_out." not in k
            and not k.endswith(".ffn.gate.cond_cosine_proto")
            and not k.endswith(".ffn.gate.token_cosine_proto")
            and not k.endswith(".ffn.gate.dynamic_bias")
            and not k.endswith(".ffn.gate.dynamic_stage_bias")
            and not k.endswith(".ffn.gate.alpha")
            and not k.endswith(".ffn.gate.context_alpha")
            and not k.endswith(".ffn.gate.cond_interaction_alpha")
            and not k.endswith(".ffn.gate.cond_cosine_alpha")
            and not k.endswith(".ffn.gate.token_cosine_alpha")
        )
    ]
    if missing_bad or ret.unexpected_keys:
        raise RuntimeError(f"Checkpoint load mismatch: missing={missing_bad[:20]}, unexpected={ret.unexpected_keys[:20]}")
    vae.eval()
    var.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    for p in var.parameters():
        p.requires_grad_(False)
    if dist.is_master():
        active_ratio = moe_config["shared_ratio"] + (moe_config["topk"] / moe_config["nexperts"]) * (1.0 - moe_config["shared_ratio"])
        print("Loaded Prism-MoE checkpoint:")
        print(f"  checkpoint: {args.moe_ckpt}")
        print(f"  n_experts: {moe_config['nexperts']}")
        print(f"  topk: {moe_config['topk']}")
        print(f"  hard_mode: {moe_config['hard_mode']}")
        print(f"  routing_mode: {args.routing_mode}")
        print(f"  active_ratio: {active_ratio:.2%}")
    return vae, var, {**moe_config, "routing_mode": args.routing_mode}


def collect_counts(var) -> Dict[int, torch.Tensor]:
    out = {}
    for layer_idx, block in enumerate(var.blocks):
        ffn = block.ffn
        if isinstance(ffn, VARD2MFFN) and ffn.last_counts is not None:
            out[layer_idx] = ffn.last_counts.detach().cpu()
    return out


def save_images(images: torch.Tensor, classes, seeds, output_dir: str) -> None:
    for i, (class_id, seed) in enumerate(zip(classes, seeds)):
        arr = images[i].permute(1, 2, 0).mul(255).clamp(0, 255).cpu().numpy().astype(np.uint8)
        Image.fromarray(arr).save(os.path.join(output_dir, f"class_{class_id:04d}_seed_{seed}.png"))


def image_path(output_dir: str, class_id: int, seed: int) -> str:
    return os.path.join(output_dir, f"class_{class_id:04d}_seed_{seed}.png")


def batch_complete(output_dir: str, classes, seeds) -> bool:
    return all(os.path.exists(image_path(output_dir, class_id, seed)) for class_id, seed in zip(classes, seeds))


def main():
    parser = argparse.ArgumentParser("Generate FID samples from Prism-MoE VAR checkpoints")
    parser.add_argument("--dist", action="store_true")
    parser.add_argument("--var_root", type=str, default=None)
    parser.add_argument("--moe_ckpt", type=str, required=True)
    parser.add_argument("--vae_ckpt", type=str, default="/liying06/liying/pretrained/var_model_zoo/vae_ch160v4096z32.pth")
    parser.add_argument("--model_depth", type=int, default=16, choices=[16, 20, 24, 30])
    parser.add_argument("--model_num_classes", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=1000)
    parser.add_argument("--shared_aln", action="store_true")
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--routing_mode", type=str, default="checkpoint", choices=["checkpoint", "hard", "soft"])
    parser.add_argument("--images_per_class", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--stats_file", type=str, default="generation_stats.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--skip_existing_batches",
        action="store_true",
        help="Skip only batches whose full expected PNG set already exists; incomplete batches are regenerated.",
    )
    args = parser.parse_args()

    from common import dist

    use_dist = args.dist or int(os.environ.get("WORLD_SIZE", "1")) > 1
    if use_dist:
        dist.initialize(timeout=60)
    else:
        dist.initialize(gpu_id_if_not_distibuted=0)
    device = dist.get_device()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    vae, var, moe_meta = load_model(args, device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    image_dir = os.path.join(args.output_dir, "fid_outputs")
    os.makedirs(image_dir, exist_ok=True)
    all_classes, all_seeds = [], []
    for class_id in range(args.num_classes):
        for img_idx in range(args.images_per_class):
            all_classes.append(class_id)
            all_seeds.append(args.seed + class_id * args.images_per_class + img_idx)
    if world_size > 1:
        my_classes = all_classes[rank::world_size]
        my_seeds = all_seeds[rank::world_size]
    else:
        my_classes = all_classes
        my_seeds = all_seeds

    if dist.is_master():
        print(f"Generating {len(all_classes)} images with world_size={world_size}, batch_size={args.batch_size}")

    stats = GenerationStats()
    skipped_existing = 0
    tokens_per_image = sum(pn * pn for pn in PATCH_NUMS)
    progress = tqdm(total=(len(my_classes) + args.batch_size - 1) // args.batch_size, desc=f"rank {rank}") if dist.is_master() else None
    for offset in range(0, len(my_classes), args.batch_size):
        batch_classes = my_classes[offset:offset + args.batch_size]
        batch_seeds = my_seeds[offset:offset + args.batch_size]
        if args.skip_existing_batches and batch_complete(image_dir, batch_classes, batch_seeds):
            skipped_existing += len(batch_classes)
            if progress is not None:
                progress.update(1)
            continue
        labels = torch.tensor(batch_classes, device=device, dtype=torch.long)
        start = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16, enabled=str(device).startswith("cuda")):
            images = var.autoregressive_infer_cfg(
                B=len(batch_classes),
                label_B=labels,
                g_seed=batch_seeds[0],
                cfg=args.cfg,
                top_k=900,
                top_p=0.96,
                more_smooth=False,
            )
        elapsed = time.time() - start
        stats.add_generation(elapsed, tokens_per_image * len(batch_classes), len(batch_classes))
        stats.add_counts(collect_counts(var))
        save_images(images, batch_classes, batch_seeds, image_dir)
        if progress is not None:
            progress.update(1)
    if progress is not None:
        progress.close()
    if use_dist:
        dist.barrier()
    if dist.is_master():
        stats_data = {
            "model_depth": args.model_depth,
            "cfg": args.cfg,
            "num_classes": args.num_classes,
            "images_per_class": args.images_per_class,
            "total_images": len(all_classes),
            "batch_size": args.batch_size,
            "skipped_existing_images_on_rank0": skipped_existing,
            "patch_nums": PATCH_NUMS,
            "moe": moe_meta,
            "world_size": world_size,
            "distributed": use_dist,
            "performance": stats.as_dict(),
        }
        with open(os.path.join(args.output_dir, args.stats_file), "w") as f:
            json.dump(stats_data, f, indent=2)
        print(f"Saved images to {image_dir}")
    if use_dist:
        dist.finalize()


if __name__ == "__main__":
    main()
