import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd):
    print(" ".join(str(x) for x in cmd), flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run([str(x) for x in cmd], check=True, cwd=ROOT, env=env)


def main():
    p = argparse.ArgumentParser("Convert a dense VAR checkpoint into a Prism-MoE checkpoint.")
    p.add_argument("--dense-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--imagenet-dir", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--calib-samples", type=int, default=200)
    p.add_argument("--calib-batch-size", type=int, default=4)
    p.add_argument("--skip-stage2", action="store_true")
    args = p.parse_args()

    output = Path(args.output)
    stage1_ckpt = output if args.skip_stage2 else output.with_name(output.stem + "_stage1.pth")

    run([
        sys.executable, "initialization/stage1.py",
        "--var_ckpt", args.dense_ckpt,
        "--vae_ckpt", args.vae_ckpt,
        "--output_path", stage1_ckpt,
        "--model_depth", args.depth,
        "--nexperts", 12,
        "--topk", 2,
        "--shared_ratio", 0.25,
        "--hard_mode",
        "--use_two_stage",
        "--candidate_multiplier", 2.0,
        "--shared_second_score", "trajectory_contribution_energy",
        "--expert_assignment", "trajectory_profile_kmeans",
        "--router_init", "trajectory_energy",
        "--router_force_bias",
        "--router_balance_calib", "trajectory",
        "--router_balance_strength", 0.25,
        "--nsamples", args.calib_samples,
        "--batch_size", args.calib_batch_size,
        "--use_images",
        "--imagenet_dir", args.imagenet_dir,
    ])

    if args.skip_stage2:
        return

    run([
        sys.executable, "initialization/stage2.py",
        "--dense_ckpt", args.dense_ckpt,
        "--vae_ckpt", args.vae_ckpt,
        "--moe_ckpt", stage1_ckpt,
        "--save_refined_ckpt", output,
        "--depth", args.depth,
        "--num_calib", args.calib_samples,
        "--calib_bs", args.calib_batch_size,
        "--use_images",
        "--imagenet_dir", args.imagenet_dir,
        "--ridge_lambda_shared", 10.0,
        "--max_delta_norm", 0.05,
        "--refine_experts",
        "--ridge_lambda_expert", 20.0,
        "--max_delta_norm_expert", 0.02,
        "--min_tokens_per_expert", 4096,
    ])


if __name__ == "__main__":
    main()
