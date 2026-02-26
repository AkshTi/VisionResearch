#!/usr/bin/env python3
"""
Step 5: Regenerate videos with corrupted pose conditioning.

For each corruption scale (0.0 = clean baseline, plus config.CORRUPTION_SCALES):
  1. Monkey-patches DFoT to apply rotation perturbation to future frame conditions
  2. Runs DFoT validation (same dataset, same clips, same model)
  3. Extracts generated frames from raw NPZ files (not GIFs — GIF encoding
     loses prediction data, producing all-black left halves)
  4. Saves to runs/action_mismatch/phase2/sample_XXXX_scaleY.Y/gen_frames/

The monkey-patch works at the on_after_batch_transfer level:
  - DFoT's dataset returns conditions as (B, T, 16) = [4 intrinsics | 12 flattened extrinsics]
  - We perturb the rotation part of future-frame extrinsics by the measured drift x scale
  - The corrupted raw conditions flow through DFoT's _process_conditions
    (normalize_by_first → rays → positional encoding) naturally
  - scale_within_bounds only affects translations (not rotations) and bound=null in config
"""

import sys
import os
import json
import shutil
import subprocess
import tempfile
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

Patches DFoTVideo.on_after_batch_transfer to perturb the rotation part of
future-frame extrinsics in raw condition space (B, T, 16).  The corrupted
conditions then flow through DFoT\'s _process_conditions pipeline:
  CameraPose.from_vectors → normalize_by_first → rays → pos_encoding
Rotation corruption survives this pipeline because:
  - normalize_by_first computes R_t @ R_0^T  (corruption in R_t is preserved)
  - scale_within_bounds only scales translations, never rotations
  - bound=null in the default config, so it\'s not even called
"""
import sys
import os
import runpy
import numpy as np

# NumPy 2.0 removed several aliases; restore them all before any DFoT import.
if not hasattr(np, "sctypes"):
    np.sctypes = {
        "int":     [np.int8, np.int16, np.int32, np.int64],
        "uint":    [np.uint8, np.uint16, np.uint32, np.uint64],
        "float":   [np.float16, np.float32, np.float64],
        "complex": [np.complex64, np.complex128],
        "others":  [bool, object, bytes, str, np.void],
    }
if not hasattr(np, "float_"):   np.float_   = np.float64
if not hasattr(np, "complex_"): np.complex_ = np.complex128
if not hasattr(np, "int_"):     np.int_     = np.intp
if not hasattr(np, "object_"):  np.object_  = object

import torch
from scipy.spatial.transform import Rotation

CORRUPTION_SCALE = float(os.environ.get("PHASE2_SCALE", "0"))
DRIFT_MEDIAN = float(os.environ.get("PHASE2_DRIFT", "35.13"))
K_HISTORY = int(os.environ.get("PHASE2_K_HISTORY", "4"))
SEED = int(os.environ.get("PHASE2_SEED", "42"))

np.random.seed(SEED + int(CORRUPTION_SCALE * 1000))

# --- Monkey-patch on_after_batch_transfer (raw conditions space) ---
from algorithms.dfot.dfot_video import DFoTVideo

_original = DFoTVideo.on_after_batch_transfer
_call_count = [0]


def _patched(self, batch, dataloader_idx):
    xs, conditions, masks, gt_videos = _original(self, batch, dataloader_idx)

    if conditions is None or CORRUPTION_SCALE <= 0:
        return xs, conditions, masks, gt_videos

    _call_count[0] += 1
    B, T, D = conditions.shape
    n_future = T - K_HISTORY
    if n_future <= 0:
        return xs, conditions, masks, gt_videos

    intr_offset = D - 12

    if _call_count[0] <= 2:
        print(f"[Phase2 DEBUG] _patched called (call #{_call_count[0]}): "
              f"conditions.shape={conditions.shape}, D={D}, intr_offset={intr_offset}")
        print(f"[Phase2 DEBUG]   conditions[0,0,:5] = {conditions[0, 0, :5].tolist()}")

    for b in range(B):
        for t in range(K_HISTORY, T):
            frame_idx = t - K_HISTORY + 1
            perturbation_deg = (DRIFT_MEDIAN / n_future) * frame_idx * CORRUPTION_SCALE

            ext_flat = conditions[b, t, intr_offset:].clone()
            ext = ext_flat.reshape(3, 4)
            R = ext[:3, :3].cpu().numpy()

            axis = np.random.randn(3)
            axis /= np.linalg.norm(axis) + 1e-8
            delta_R = Rotation.from_rotvec(
                axis * np.radians(perturbation_deg)
            ).as_matrix()
            R_corrupted = (delta_R @ R).astype(np.float32)

            if t == K_HISTORY and _call_count[0] <= 2:
                actual_diff = np.degrees(np.arccos(np.clip(
                    (np.trace(R_corrupted @ R.T) - 1) / 2, -1, 1)))
                print(f"[Phase2 DEBUG]   b={b} t={t}: "
                      f"perturbation_deg={perturbation_deg:.2f}, "
                      f"actual_diff={actual_diff:.2f}")

            ext[:3, :3] = torch.from_numpy(R_corrupted).to(conditions.device)
            conditions[b, t, intr_offset:] = ext.reshape(12)

    return xs, conditions, masks, gt_videos


DFoTVideo.on_after_batch_transfer = _patched
print(f"[Phase2] on_after_batch_transfer patch applied (scale={CORRUPTION_SCALE})")

# --- Process @-shortcuts then run main.py directly ---
from utils.hydra_utils import unwrap_shortcuts
sys.argv = unwrap_shortcuts(sys.argv, config_path="configurations", config_name="config")

main_path = os.path.abspath("main.py")
sys.argv[0] = main_path
runpy.run_path(main_path, run_name="__main__")
''')


# ============================================================
# Helpers
# ============================================================

def _numpy_fix_env() -> dict:
    """Return an env dict with a sitecustomize.py that patches NumPy 2.0 aliases.

    PyTorch Lightning spawns one worker process per GPU rank.  Each spawned
    process gets a fresh Python interpreter, so inline patches in the runner
    script are invisible to ranks > 0.  sitecustomize.py is auto-executed by
    *every* Python process at startup, making it the only reliable injection
    point for multi-GPU jobs.
    """
    fix_code = textwrap.dedent("""\
        import numpy as np
        if not hasattr(np, 'sctypes'):
            np.sctypes = {
                'int':     [np.int8, np.int16, np.int32, np.int64],
                'uint':    [np.uint8, np.uint16, np.uint32, np.uint64],
                'float':   [np.float16, np.float32, np.float64],
                'complex': [np.complex64, np.complex128],
                'others':  [bool, object, bytes, str, np.void],
            }
        if not hasattr(np, 'float_'):   np.float_   = np.float64
        if not hasattr(np, 'complex_'): np.complex_ = np.complex128
        if not hasattr(np, 'int_'):     np.int_     = np.intp
        if not hasattr(np, 'bool_'):    np.bool_    = np.bool_
        if not hasattr(np, 'object_'):  np.object_  = object
        if not hasattr(np, 'str_'):     np.str_     = np.str_
    """)
    tmpdir = tempfile.mkdtemp(prefix="np_fix_")
    (Path(tmpdir) / "sitecustomize.py").write_text(fix_code)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = tmpdir + (":" + existing if existing else "")
    return env


def create_runner(dfot_repo: Path) -> Path:
    """Write the monkey-patch runner script into the DFoT repo."""
    runner_path = dfot_repo / "_phase2_runner.py"
    runner_path.write_text(RUNNER_SCRIPT)
    return runner_path


def snapshot_wandb_dirs(dfot_repo: Path) -> set:
    """Return the set of existing wandb offline-run-* dirs right now."""
    wandb_root = dfot_repo / "wandb"
    if not wandb_root.exists():
        return set()
    return set(wandb_root.glob("offline-run-*"))


def find_run_gifs(run_dir: Path, dfot_repo: Path, pre_run_wandb: set) -> list:
    """Find prediction GIFs produced by a single DFoT run.

    Uses a set-difference approach: we snapshot wandb dirs BEFORE the run
    starts, then after it finishes we take only the *new* ones.  This is
    immune to file-sync tools resetting mtime on old dirs.
    """
    gifs: list = []

    # 1. Hydra output dir (DFoT sometimes writes GIFs there too)
    if run_dir and run_dir.exists():
        found = list(run_dir.rglob("*.gif"))
        print(f"    [GIF search] Hydra dir {run_dir}: {len(found)} GIFs")
        gifs += found

    # 2. New wandb offline runs (dirs that didn't exist before this run)
    wandb_root = dfot_repo / "wandb"
    if wandb_root.exists():
        current = set(wandb_root.glob("offline-run-*"))
        new_runs = current - pre_run_wandb
        print(f"    [GIF search] New wandb runs: {[p.name for p in new_runs]}")
        for wandb_run in sorted(new_runs):
            media = wandb_run / "files" / "media"
            if media.exists():
                found = list(media.rglob("*.gif"))
                print(f"    [GIF search]   {wandb_run.name}/files/media: {len(found)} GIFs")
                gifs += found

    return sorted(set(gifs))


def parse_output_dir(stdout: str) -> Path:
    """Extract the Hydra output directory from DFoT's stdout.

    DFoT prints (with ANSI color codes):
      '\x1b[36mOutputs will be saved to:\x1b[39m /path/to/outputs/...'
    We strip all ANSI escape sequences before matching.
    """
    import re
    clean = re.sub(r"\x1b\[[0-9;]*m", "", stdout)
    match = re.search(r"Outputs will be saved to:\s*(.+)", clean)
    if match:
        return Path(match.group(1).strip())
    return Path("")


def extract_frames_from_gif(gif_path: Path, k_history: int = 4):
    """
    Extract frames from a DFoT prediction GIF.

    DFoT log_video does: torch.cat([prediction, gt], dim=-1)
    so GIF layout is [Predicted | GT] — left half = predicted, right = GT.

    Returns:
        predicted_future: list of PIL.Image (left half, future frames only)
        gt_future: list of PIL.Image (right half, future frames only)
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
            predicted_future.append(frame.crop((0, 0, half_w, frame.size[1])))
            gt_future.append(frame.crop((half_w, 0, width, frame.size[1])))

    return predicted_future, gt_future


def run_dfot_with_corruption(
    dfot_repo: Path,
    scale: float,
    drift_median: float,
    n_samples: int,
    raw_dir: str,
) -> tuple:
    """Run DFoT validation with the corrupted pose monkey-patch.
    Returns (returncode, output_dir)."""
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
        # Default max_num_videos=8 in dfot_video.yaml caps NPZ output regardless of
        # num_eval_videos. Must match n_samples so all clips get saved to raw_dir.
        f"algorithm.logging.max_num_videos={n_samples}",
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
        # Sampling noise schedule matching RE10K training checkpoint.
        # Default beta_schedule=cosine/shift=1.0 gives wrong logsnr values during inference
        # causing all-black generated frames. RE10K was trained with cosine_simple_diffusion
        # shifted=0.125 (see configurations/dataset_experiment/realestate10k_video_generation.yaml).
        "algorithm.diffusion.beta_schedule=cosine_simple_diffusion",
        "++algorithm.diffusion.schedule_fn_kwargs.shifted=0.125",
        "++algorithm.diffusion.schedule_fn_kwargs.interpolated=false",
        f"++algorithm.logging.raw_dir={raw_dir}",
    ]

    env = _numpy_fix_env()
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
        for line in result.stdout.splitlines():
            if "[Phase2" in line:
                print(f"  {line}")
        print("  stderr (tail):\n", result.stderr[-3000:])
        print("  stdout (tail):\n", result.stdout[-1500:])
        return result.returncode, Path("")

    print(f"  [OK] scale={scale:.1f} completed")
    # Print all Phase2 DEBUG lines from stdout
    for line in result.stdout.splitlines():
        if "[Phase2" in line:
            print(f"  {line}")
    print("  stdout (tail):\n", result.stdout[-500:])

    output_dir = parse_output_dir(result.stdout)
    print(f"  Output dir: {output_dir}")
    return result.returncode, output_dir


# ============================================================
# NPZ-based frame extraction (replaces broken GIF extraction)
# ============================================================

def extract_frames_from_npz(raw_dir: Path, sample_idx: int, k_history: int = 4):
    """
    Extract gen and gt future frames from DFoT's raw NPZ output.

    When algorithm.logging.raw_dir is set, DFoT saves data.npz with:
        gt:  (T, C, H, W) uint8 [0-255]
        gen: (T, C, H, W) uint8 [0-255]

    Returns:
        gen_future: list of PIL.Image (future prediction frames)
        gt_future:  list of PIL.Image (future ground-truth frames)
    """
    npz_path = raw_dir / str(sample_idx) / "data.npz"
    if not npz_path.exists():
        print(f"    WARNING: {npz_path} not found")
        return [], []

    data = np.load(npz_path)
    if "gen" not in data or "gt" not in data:
        print(f"    WARNING: NPZ missing expected keys (has: {list(data.keys())})")
        return [], []
    gen = data["gen"]  # (T, C, H, W)
    gt = data["gt"]    # (T, C, H, W)

    # Warn about black frames
    for t in range(gen.shape[0]):
        if gen[t].mean() < 1.0:
            print(f"    WARNING: sample {sample_idx} gen frame {t} is black (mean={gen[t].mean():.2f})")

    gen_future = []
    gt_future = []
    for t in range(k_history, gen.shape[0]):
        gen_frame = np.transpose(gen[t], (1, 2, 0))  # (H, W, C)
        gt_frame = np.transpose(gt[t], (1, 2, 0))
        gen_future.append(Image.fromarray(gen_frame))
        gt_future.append(Image.fromarray(gt_frame))

    return gen_future, gt_future


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
    clean_ids = set(manifest.get("clean_sample_ids", []))
    clean_indices = sorted(int(sid.replace("sample_", "")) for sid in clean_ids) if clean_ids else list(range(N_SAMPLES))
    print(f"  Phase 1 median drift: {drift_median:.2f} deg")
    print(f"  Corruption scales: {CORRUPTION_SCALES}")
    print(f"  Clean samples: {len(clean_indices)}/{N_SAMPLES}")

    # Create the monkey-patch runner in the DFoT repo
    runner_path = create_runner(DFOT_REPO)
    print(f"  Runner: {runner_path}")

    # ----------------------------------------------------------------
    # Run DFoT for scale=0.0 (clean baseline) + each corruption scale.
    # Uses raw_dir NPZ files instead of GIFs — GIF encoding drops
    # prediction data (left half comes out all-black).
    # ----------------------------------------------------------------
    all_scales = [0.0] + list(CORRUPTION_SCALES)
    scales_ok = []

    for scale in all_scales:
        print(f"\n{'='*60}")
        if scale == 0.0:
            print(f"  CLEAN BASELINE (scale=0.0)")
        else:
            print(f"  CORRUPTION SCALE = {scale:.1f}  "
                  f"(~{drift_median * scale:.1f} deg final)")
        print(f"{'='*60}")

        raw_dir = str(DFOT_REPO.resolve() / f"_raw_scale{scale:.1f}")
        if Path(raw_dir).exists():
            shutil.rmtree(raw_dir)

        rc, run_output_dir = run_dfot_with_corruption(
            DFOT_REPO, scale, drift_median, N_SAMPLES, raw_dir
        )

        if rc != 0:
            print(f"  Skipping frame extraction for scale={scale:.1f}")
            continue

        raw_path = Path(raw_dir)
        npz_count = len(list(raw_path.rglob("data.npz")))
        print(f"  Found {npz_count} NPZ files in {raw_path}")

        if npz_count == 0:
            print(f"  WARNING: No NPZ files — raw_dir may not be supported")
            continue

        for i in clean_indices:
            gen_frames, gt_frames = extract_frames_from_npz(
                raw_path, i, k_history=K_HISTORY
            )

            if not gen_frames:
                print(f"    sample_{i:04d}: no frames extracted")
                continue

            # Save gen (prediction) frames
            out_dir = phase2_dir / f"sample_{i:04d}_scale{scale:.1f}" / "gen_frames"
            out_dir.mkdir(parents=True, exist_ok=True)
            for j, frame in enumerate(gen_frames):
                frame.save(out_dir / f"frame_{j:04d}.png")

            # Save GT frames (only needed once, from the clean baseline)
            if scale == 0.0:
                gt_out = phase2_dir / f"gt_frames_sample_{i:04d}"
                gt_out.mkdir(parents=True, exist_ok=True)
                for j, frame in enumerate(gt_frames):
                    frame.save(gt_out / f"frame_{j:04d}.png")

            frame0 = np.array(gen_frames[0])
            print(f"    sample_{i:04d}: saved {len(gen_frames)} frames "
                  f"(mean={frame0.mean():.1f}, std={frame0.std():.1f})")

        # Cleanup raw_dir to save disk space
        shutil.rmtree(raw_path, ignore_errors=True)
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
