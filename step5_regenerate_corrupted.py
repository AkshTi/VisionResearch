#!/usr/bin/env python3
"""
Step 5: Regenerate videos with corrupted pose conditioning.

For each corruption scale (from config.CORRUPTION_SCALES):
  1. Monkey-patches DFoT to apply rotation perturbation to future frame conditions
  2. Runs DFoT validation (same dataset, same clips, same model)
  3. Extracts generated frames from output GIFs
  4. Saves to runs/action_mismatch/phase2/sample_XXXX_scaleY.Y/gen_frames/

The monkey-patch works at the `on_after_batch_transfer` level:
  - DFoT's dataset returns conditions as (B, T, 16) = [4 intrinsics | 12 flattened extrinsics]
  - We perturb the rotation part of future-frame extrinsics by the measured drift x scale
  - Intrinsics and translation are left unchanged
"""

import sys
import os
import json
import shutil
import subprocess
import textwrap
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DFOT_REPO, RUNS_DIR, N_SAMPLES, K_HISTORY, T_FUTURE,
    FRAME_SKIP, DFOT_CHECKPOINT, HISTORY_GUIDANCE_NAME,
    HISTORY_GUIDANCE_SCALE, SEED, WANDB_ENTITY, CORRUPTION_SCALES
)


# ============================================================
# Monkey-patch runner script (written into DFoT repo at runtime)
# ============================================================

RUNNER_SCRIPT = textwrap.dedent('''\
"""
Phase 2 runner: DFoT with corrupted pose conditioning.

Uses runpy.run_module("main", alter_sys=True) to emulate `python -m main`
exactly, so Hydra resolves configurations/ correctly. The monkey-patch is
applied at class level before run_module executes, so sys.modules caching
ensures DFoTVideoPose picks up the patched method.
"""
import sys
import os
import runpy
import numpy as np

# NumPy 2.0 removed np.sctypes; restore it so imgaug (pulled in by pyiqa/DFoT) doesn\'t crash.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int":     [np.int8, np.int16, np.int32, np.int64],
        "uint":    [np.uint8, np.uint16, np.uint32, np.uint64],
        "float":   [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others":  [bool, object, bytes, str, np.void],
    }

import torch
from scipy.spatial.transform import Rotation

CORRUPTION_SCALE = float(os.environ.get("PHASE2_SCALE", "0"))
DRIFT_MEDIAN = float(os.environ.get("PHASE2_DRIFT", "35.13"))
K_HISTORY = int(os.environ.get("PHASE2_K_HISTORY", "4"))
SEED = int(os.environ.get("PHASE2_SEED", "42"))

np.random.seed(SEED + int(CORRUPTION_SCALE * 1000))

# --- Monkey-patch at class level ---
from algorithms.dfot.dfot_video import DFoTVideo
from algorithms.dfot.dfot_video_pose import DFoTVideoPose

_original = DFoTVideo.on_after_batch_transfer


def _patched(self, batch, dataloader_idx):
    """Apply rotation corruption to future-frame conditions."""
    xs, conditions, masks, gt_videos = _original(self, batch, dataloader_idx)

    if conditions is not None and CORRUPTION_SCALE > 0:
        B, T, D = conditions.shape
        n_future = T - K_HISTORY

        if n_future > 0:
            for b in range(B):
                for t in range(K_HISTORY, T):
                    # conditions[b, t] is 16-dim: [4 intrinsics | 12 flattened extrinsics]
                    # extrinsics layout (row-major 3x4):
                    #   [R00 R01 R02 tx  R10 R11 R12 ty  R20 R21 R22 tz]
                    ext_flat = conditions[b, t, 4:].clone()
                    ext = ext_flat.reshape(3, 4)
                    R = ext[:3, :3].cpu().numpy()

                    # Linear perturbation: frame_idx/n_future * drift * scale
                    frame_idx = t - K_HISTORY + 1
                    per_frame_drift = DRIFT_MEDIAN / n_future
                    perturbation_deg = per_frame_drift * frame_idx * CORRUPTION_SCALE

                    axis = np.random.randn(3)
                    axis /= (np.linalg.norm(axis) + 1e-8)
                    angle_rad = np.radians(perturbation_deg)
                    delta_R = Rotation.from_rotvec(axis * angle_rad).as_matrix()
                    R_corrupted = delta_R @ R

                    ext[:3, :3] = torch.from_numpy(R_corrupted).float().to(conditions.device)
                    conditions[b, t, 4:] = ext.reshape(12)

    return xs, conditions, masks, gt_videos


DFoTVideoPose.on_after_batch_transfer = _patched
print(f"[Phase2] Patch applied (scale={CORRUPTION_SCALE})")

# --- Process @-shortcuts then run main exactly as `python -m main` would ---
from utils.hydra_utils import unwrap_shortcuts
sys.argv = unwrap_shortcuts(sys.argv, config_path="configurations", config_name="config")

# alter_sys=True makes runpy set sys.argv[0] to main.py\'s path,
# which is what Hydra needs to resolve config_path="configurations"
runpy.run_module("main", run_name="__main__", alter_sys=True)
''')


# ============================================================
# Helpers
# ============================================================

def create_runner(dfot_repo: Path) -> Path:
    """Write the monkey-patch runner script into the DFoT repo."""
    runner_path = dfot_repo / "_phase2_runner.py"
    runner_path.write_text(RUNNER_SCRIPT)
    return runner_path


def find_latest_run_gifs(dfot_repo: Path) -> list:
    """Find prediction GIFs from the most recent DFoT output run."""
    latest = dfot_repo / "outputs" / "latest-run"
    if not latest.exists():
        return []
    run_dir = latest.resolve()
    return sorted(run_dir.rglob("prediction_vis/video_*.gif"))


def extract_frames_from_gif(gif_path: Path, k_history: int = 4):
    """
    Extract frames from a DFoT prediction GIF.

    DFoT GIFs are [GT | Predicted] side-by-side (512x256), 12 frames total
    (4 context + 8 future). Context frames are identical on both sides.

    Returns:
        predicted_future: list of PIL.Image (right half, future frames only)
        gt_future: list of PIL.Image (left half, future frames only)
    """
    gif = Image.open(gif_path)
    width = gif.size[0]
    half_w = width // 2

    predicted_future = []
    gt_future = []

    for i in range(gif.n_frames):
        gif.seek(i)
        frame = gif.convert("RGB")

        if i >= k_history:
            # Right half = predicted
            predicted_future.append(frame.crop((half_w, 0, width, frame.size[1])))
            # Left half = GT
            gt_future.append(frame.crop((0, 0, half_w, frame.size[1])))

    return predicted_future, gt_future


def run_dfot_with_corruption(
    dfot_repo: Path,
    scale: float,
    drift_median: float,
    n_samples: int,
) -> int:
    """Run DFoT validation with the corrupted pose monkey-patch."""
    n_frames = K_HISTORY + T_FUTURE

    cmd = [
        "python", "_phase2_runner.py",
        f"+name=phase2_scale{scale:.1f}",
        "dataset=realestate10k_mini",
        "algorithm=dfot_video_pose",
        "experiment=video_generation",
        "@diffusion/continuous",
        f"load=pretrained:{DFOT_CHECKPOINT}",
        "algorithm.checkpoint.strict=false",
        "experiment.tasks=[validation]",
        "experiment.validation.data.shuffle=False",
        f"dataset.context_length={K_HISTORY}",
        f"dataset.frame_skip={FRAME_SKIP}",
        f"dataset.n_frames={n_frames}",
        "experiment.validation.batch_size=1",
        f"dataset.num_eval_videos={n_samples}",
        f"algorithm.tasks.prediction.history_guidance.name={HISTORY_GUIDANCE_NAME}",
        f"+algorithm.tasks.prediction.history_guidance.guidance_scale={HISTORY_GUIDANCE_SCALE}",
        f"wandb.entity={WANDB_ENTITY}",
        "wandb.mode=offline",
        "dataset.subdataset_size=null",
        "experiment.reload_dataloaders_every_n_epochs=0",
        "algorithm.backbone.channels=[128,256,576,1152]",
        "algorithm.backbone.num_updown_blocks=[3,3,6]",
        "algorithm.backbone.num_mid_blocks=20",
        "algorithm.backbone.num_heads=9",
        "++algorithm.diffusion.training_schedule.name=cosine",
        "++algorithm.diffusion.training_schedule.shift=0.125",
        "++algorithm.diffusion.loss_weighting.strategy=sigmoid",
        "++algorithm.diffusion.loss_weighting.sigmoid_bias=-1.0",
    ]

    env = os.environ.copy()
    env["PHASE2_SCALE"] = str(scale)
    env["PHASE2_DRIFT"] = str(drift_median)
    env["PHASE2_K_HISTORY"] = str(K_HISTORY)
    env["PHASE2_SEED"] = str(SEED)

    print(f"\n  [RUN] DFoT with corruption scale={scale:.1f}")
    print(f"    cwd: {dfot_repo}")
    print(f"    Expected final corruption: ~{drift_median * scale:.1f} deg")

    result = subprocess.run(
        cmd,
        cwd=str(dfot_repo),
        env=env,
        capture_output=True,
        text=True,
        timeout=7200,
    )

    if result.returncode != 0:
        print(f"\n  [FAILED] scale={scale:.1f}")
        print("  stderr (tail):\n", result.stderr[-3000:])
        print("  stdout (tail):\n", result.stdout[-1500:])
    else:
        print(f"  [OK] scale={scale:.1f} completed")
        print("  stdout (tail):\n", result.stdout[-500:])

    return result.returncode


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("STEP 5: Regenerate videos with corrupted poses")
    print("=" * 60)

    phase2_dir = RUNS_DIR / "phase2"

    # Load drift statistics from Phase 1
    manifest_path = phase2_dir / "phase2_manifest.json"
    if not manifest_path.exists():
        print(f"  ERROR: {manifest_path} not found. Run step4 first.")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    drift_median = manifest["drift_statistics"]["p50"]
    print(f"  Phase 1 median drift: {drift_median:.2f} deg")
    print(f"  Corruption scales: {CORRUPTION_SCALES}")

    # Create the monkey-patch runner in the DFoT repo
    runner_path = create_runner(DFOT_REPO)
    print(f"  Runner: {runner_path}")

    # ----------------------------------------------------------------
    # Extract GT future frames from the clean baseline (step1 GIFs)
    # These are used by step6 for LPIPS evaluation.
    # ----------------------------------------------------------------
    print("\n--- Extracting GT future frames from clean baseline ---")
    clean_gen_dir = RUNS_DIR / "generated"

    for i in range(N_SAMPLES):
        sample_dir = clean_gen_dir / f"sample_{i:04d}"
        gif_dir = sample_dir / "videos"
        if not gif_dir.exists():
            continue

        gifs = sorted(gif_dir.glob("*.gif"))
        if not gifs:
            continue

        _, gt_frames = extract_frames_from_gif(gifs[0], k_history=K_HISTORY)

        gt_out = phase2_dir / f"gt_frames_sample_{i:04d}"
        gt_out.mkdir(parents=True, exist_ok=True)
        for j, frame in enumerate(gt_frames):
            frame.save(gt_out / f"frame_{j:04d}.png")

    print(f"  GT frames saved to {phase2_dir}/gt_frames_sample_*/")

    # ----------------------------------------------------------------
    # Run DFoT for each corruption scale
    # ----------------------------------------------------------------
    scales_ok = []

    for scale in CORRUPTION_SCALES:
        print(f"\n{'='*60}")
        print(f"  CORRUPTION SCALE = {scale:.1f}  "
              f"(~{drift_median * scale:.1f} deg final)")
        print(f"{'='*60}")

        rc = run_dfot_with_corruption(
            DFOT_REPO, scale, drift_median, N_SAMPLES
        )

        if rc != 0:
            print(f"  Skipping frame extraction for scale={scale:.1f}")
            continue

        # Find the generated GIFs from the latest run
        gifs = find_latest_run_gifs(DFOT_REPO)
        if not gifs:
            print(f"  WARNING: No GIFs found for scale={scale:.1f}")
            continue

        print(f"  Found {len(gifs)} GIFs")

        for i, gif_path in enumerate(gifs[:N_SAMPLES]):
            predicted_frames, _ = extract_frames_from_gif(
                gif_path, k_history=K_HISTORY
            )

            out_dir = phase2_dir / f"sample_{i:04d}_scale{scale:.1f}" / "gen_frames"
            out_dir.mkdir(parents=True, exist_ok=True)

            for j, frame in enumerate(predicted_frames):
                frame.save(out_dir / f"frame_{j:04d}.png")

            print(f"    sample_{i:04d}: saved {len(predicted_frames)} frames")

        scales_ok.append(scale)

    # ----------------------------------------------------------------
    # Cleanup and update manifest
    # ----------------------------------------------------------------
    if runner_path.exists():
        runner_path.unlink()

    manifest["regeneration_complete"] = True
    manifest["scales_generated"] = scales_ok
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Step 5 complete!")
    print(f"  Scales generated: {scales_ok}")
    print(f"  Next: run step6_evaluate_phase2.py")


if __name__ == "__main__":
    main()
