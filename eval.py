import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser("Generate samples and compute FID for a Prism-MoE checkpoint.")
    p.add_argument("--moe-ckpt", required=True)
    p.add_argument("--vae-ckpt", required=True)
    p.add_argument("--ref-npz", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--depth", type=int, default=16)
    p.add_argument("--gpus", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--images-per-class", type=int, default=50)
    p.add_argument("--cfg", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--keep-images", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")

    gen_cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node", args.gpus,
        "engines/fid_engine.py",
        "--dist",
        "--moe_ckpt", args.moe_ckpt,
        "--vae_ckpt", args.vae_ckpt,
        "--model_depth", args.depth,
        "--images_per_class", args.images_per_class,
        "--batch_size", args.batch_size,
        "--topk", 2,
        "--routing_mode", "checkpoint",
        "--cfg", args.cfg,
        "--seed", args.seed,
        "--output_dir", out,
        "--stats_file", "generation_stats.json",
        "--skip_existing_batches",
    ]
    subprocess.run([str(x) for x in gen_cmd], check=True, cwd=ROOT, env=env)

    fid_cmd = [
        sys.executable, "-m", "pytorch_fid",
        out / "fid_outputs",
        args.ref_npz,
        "--device", "cuda:0",
        "--batch-size", 50,
        "--num-workers", 8,
    ]
    with open(out / "fid_result.txt", "w") as f:
        subprocess.run([str(x) for x in fid_cmd], check=True, cwd=ROOT, stdout=f, env=env)

    if not args.keep_images:
        subprocess.run(["rm", "-rf", str(out / "fid_outputs")], check=True)


if __name__ == "__main__":
    main()
