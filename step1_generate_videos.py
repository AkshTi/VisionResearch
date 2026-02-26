#!/usr/bin/env python3
"""
Step 1: Generate videos with the pose-conditioned DFoT model.

Runs DFoT via its Hydra CLI, then auto-discovers outputs and copies them into:
  RUNS_DIR/generated/sample_XXXX/{videos,gen_frames,poses}/...

If DFoT fails, prints the *real* error and exits non-zero by default.
"""

import sys
import os
import json
import subprocess
import shutil
import re
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DFOT_REPO, RUNS_DIR, N_SAMPLES, K_HISTORY, T_FUTURE,
    FRAME_SKIP, DFOT_CHECKPOINT, HISTORY_GUIDANCE_NAME,
    HISTORY_GUIDANCE_SCALE, SEED, WANDB_ENTITY
)


# ============================================================
# Step 1 runner: embedded script written into DFoT repo
# ============================================================

STEP1_RUNNER_SCRIPT = '''\
"""
Step 1 runner: runs DFoT validation and saves GT future poses per sample.

Applies two patches before DFoT runs:
  1. NumPy 2.0 compatibility (np.sctypes, np.float_, etc.)
  2. on_after_batch_transfer intercept to save conditioning poses to disk

Pose format: the DFoT condition tensor is (B, T, 16) where each 16-dim
vector = [4 intrinsics | 12-dim row-major 3x4 extrinsic [R|t]].
We reconstruct 4x4 SE(3) matrices and save as poses_gt_future.npy.
"""
import sys
import os
import runpy
import numpy as np
from pathlib import Path

# --- NumPy 2.0 compatibility ---
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

K_HISTORY  = int(os.environ.get("STEP1_K_HISTORY",  "4"))
OUTPUT_DIR = Path(os.environ.get("STEP1_OUTPUT_DIR", "."))

# --- Patch: save GT future poses during validation ---
from algorithms.dfot.dfot_video import DFoTVideo
from algorithms.dfot.dfot_video_pose import DFoTVideoPose

_original = DFoTVideo.on_after_batch_transfer
_sample_counter = [0]


def _pose_saver(self, batch, dataloader_idx):
    xs, conditions, masks, gt_videos = _original(self, batch, dataloader_idx)

    if conditions is not None:
        B, T, D = conditions.shape
        n_future = T - K_HISTORY
        if n_future > 0:
            for b in range(B):
                # conditions[b, t] = [4 intrinsics | 12-dim row-major [R|t]]
                future_conds = conditions[b, K_HISTORY:, 4:].cpu().numpy()  # (n_future, 12)
                poses_gt = np.zeros((n_future, 4, 4), dtype=np.float32)
                for t in range(n_future):
                    poses_gt[t, :3, :] = future_conds[t].reshape(3, 4)
                    poses_gt[t, 3, 3] = 1.0

                out_dir = OUTPUT_DIR / f"sample_{_sample_counter[0]:04d}"
                out_dir.mkdir(parents=True, exist_ok=True)
                np.save(out_dir / "poses_gt_future.npy", poses_gt)
                print(f"[Step1Runner] GT poses saved: sample_{_sample_counter[0]:04d} "
                      f"({n_future} future frames)")
                _sample_counter[0] += 1

    return xs, conditions, masks, gt_videos


DFoTVideoPose.on_after_batch_transfer = _pose_saver
print(f"[Step1Runner] Pose-saving patch applied (K_HISTORY={K_HISTORY})")

# --- Run main.py via run_path so Hydra resolves configurations/ correctly ---
from utils.hydra_utils import unwrap_shortcuts
sys.argv = unwrap_shortcuts(sys.argv, config_path="configurations", config_name="config")
main_path = os.path.abspath("main.py")
sys.argv[0] = main_path
runpy.run_path(main_path, run_name="__main__")
'''


def create_step1_runner(dfot_repo: Path) -> Path:
    """Write the step 1 pose-saving runner into the DFoT repo."""
    runner_path = dfot_repo / "_step1_runner.py"
    runner_path.write_text(STEP1_RUNNER_SCRIPT)
    return runner_path


# ----------------------------
# Utilities
# ----------------------------

def _numpy_fix_env() -> dict:
    """
    Return an env dict that injects a sitecustomize.py restoring np.sctypes.

    NumPy 2.0 removed np.sctypes, which breaks imgaug (pulled in by pyiqa/DFoT).
    sitecustomize.py is executed automatically by Python at startup before any
    other import, so the patch is in place before DFoT's imports run.
    """
    fix_code = (
        "import numpy as np\n"
        # np.sctypes removed in NumPy 2.0 (used by imgaug via pyiqa)
        "if not hasattr(np, 'sctypes'):\n"
        "    np.sctypes = {\n"
        "        'int':     [np.int8, np.int16, np.int32, np.int64],\n"
        "        'uint':    [np.uint8, np.uint16, np.uint32, np.uint64],\n"
        "        'float':   [np.float16, np.float32, np.float64],\n"
        "        'complex': [np.complex64, np.complex128],\n"
        "        'others':  [bool, object, bytes, str, np.void],\n"
        "    }\n"
        # scalar type aliases removed in NumPy 2.0 (used by torchmetrics FID)
        "if not hasattr(np, 'float_'):   np.float_   = np.float64\n"
        "if not hasattr(np, 'complex_'): np.complex_ = np.complex128\n"
        "if not hasattr(np, 'int_'):     np.int_     = np.intp\n"
        "if not hasattr(np, 'bool_'):    np.bool_    = np.bool_\n"
        "if not hasattr(np, 'object_'):  np.object_  = object\n"
        "if not hasattr(np, 'str_'):     np.str_     = np.str_\n"
    )
    tmpdir = tempfile.mkdtemp(prefix="np_fix_")
    (Path(tmpdir) / "sitecustomize.py").write_text(fix_code)
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = tmpdir + (":" + existing if existing else "")
    return env


def _run(cmd: List[str], cwd: Path, timeout: int = 3600,
         env: Optional[dict] = None) -> subprocess.CompletedProcess:
    """Run a command and return CompletedProcess. Does NOT swallow errors."""
    print("\n[RUN]")
    print(f"  cwd: {cwd}")
    print("  cmd:", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def preflight_or_die(output_dir: Path, env: Optional[dict] = None) -> None:
    """Validate repo, python, likely deps, and directory structure."""
    print("\n[Preflight]")

    if not DFOT_REPO.exists():
        raise FileNotFoundError(f"DFOT_REPO not found at: {DFOT_REPO}")

    main_py = DFOT_REPO / "main.py"
    if not main_py.exists():
        # DFoT runs via `python -m main`, so it should have main.py at repo root.
        raise FileNotFoundError(f"Expected DFoT entrypoint missing: {main_py}")

    # Create output dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check that python can import DFoT's main module when run from repo
    test = _run(["python", "-c", "import main; print('ok')"], cwd=DFOT_REPO,
                timeout=120, env=env)
    if test.returncode != 0:
        print("\n[Preflight FAIL] Cannot import DFoT main module.")
        print("stdout:\n", test.stdout[-2000:])
        print("stderr:\n", test.stderr[-2000:])
        raise RuntimeError("DFoT environment is not set up (import main failed).")

    # Check pillow ONLY because mock mode uses it
    try:
        import PIL  # noqa: F401
    except Exception:
        print("[Warn] pillow not installed; mock output generation will fail if triggered.")
        print("      Install with: pip install pillow")

    # Check checkpoint string looks like DFoT pretrained identifier or file
    ckpt = str(DFOT_CHECKPOINT)
    if not ckpt:
        raise ValueError("DFOT_CHECKPOINT is empty in config.py")

    # If it's a path, verify it exists. If it's a name, allow (DFoT may resolve it).
    ckpt_path = Path(ckpt)
    if ("/" in ckpt or ckpt.endswith(".ckpt")) and ckpt_path.exists() is False:
        print(f"[Warn] DFOT_CHECKPOINT looks like a file/path but does not exist: {ckpt_path}")
        print("      If you intended pretrained name (e.g. DFoT_RE10K.ckpt), set it exactly like DFoT wiki uses.")


def find_latest_dir(root: Path) -> Optional[Path]:
    if not root.exists():
        return None
    dirs = [p for p in root.rglob("*") if p.is_dir()]
    if not dirs:
        return None
    return max(dirs, key=lambda p: p.stat().st_mtime)


def collect_media_from_outputs(dfot_repo: Path, n_samples: int) -> Dict[str, Any]:
    """
    Search DFoT repo for recent hydra outputs and wandb offline artifacts.
    Returns dict with discovered files.
    """
    found: Dict[str, Any] = {
        "hydra_runs": [],
        "wandb_runs": [],
        "videos": [],
        "gifs": [],
        "frames_dirs": [],
        "poses": [],
    }

    # Hydra outputs folder (DFoT uses this commonly)
    outputs_root = dfot_repo / "outputs"
    if outputs_root.exists():
        # Get most recent run directories (top-level children usually are date folders)
        candidates = sorted(outputs_root.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True)
        # Keep only dirs that look like a run (contain .hydra or config files)
        run_dirs = []
        for p in candidates:
            if p.is_dir() and ((p / ".hydra").exists() or (p / "log.txt").exists()):
                run_dirs.append(p)
            if len(run_dirs) >= 5:
                break
        found["hydra_runs"] = [str(p) for p in run_dirs]

        # Search within newest run dirs for media
        for run in run_dirs[:3]:
            for ext in ("*.mp4", "*.webm"):
                found["videos"] += [str(x) for x in run.rglob(ext)]
            found["gifs"] += [str(x) for x in run.rglob("*.gif")]
            # Common pose dumps
            for ext in ("*.npy", "*.npz"):
                for x in run.rglob(ext):
                    name = x.name.lower()
                    if "pose" in name or "poses" in name:
                        found["poses"].append(str(x))

    # W&B offline folder inside DFoT repo (common)
    wandb_root = dfot_repo / "wandb"
    if wandb_root.exists():
        # Look for offline runs
        run_dirs = sorted(
            [p for p in wandb_root.glob("offline-run-*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )[:5]
        found["wandb_runs"] = [str(p) for p in run_dirs]
        for run in run_dirs[:3]:
            media = run / "files" / "media"
            if media.exists():
                found["videos"] += [str(x) for x in media.rglob("*.mp4")]
                found["videos"] += [str(x) for x in media.rglob("*.webm")]
                found["gifs"] += [str(x) for x in media.rglob("*.gif")]
            # Sometimes pose arrays get logged as files
            files = run / "files"
            if files.exists():
                for x in files.rglob("*.npy"):
                    if "pose" in x.name.lower() or "poses" in x.name.lower():
                        found["poses"].append(str(x))

    # De-dup
    for k in ("videos", "gifs", "poses"):
        found[k] = sorted(set(found[k]))

    # Cap lists to something reasonable
    found["videos"] = found["videos"][: max(50, n_samples * 5)]
    found["gifs"] = found["gifs"][: max(50, n_samples * 5)]
    found["poses"] = found["poses"][: max(200, n_samples * 10)]

    return found


def stage_into_samples(output_dir: Path, discovered: Dict[str, Any]) -> Dict[str, Any]:
    """
    Copy discovered artifacts into output_dir/sample_XXXX/...
    We can't guarantee 1:1 mapping to samples without knowing DFoT naming,
    but we at least make a deterministic staging so downstream scripts can run.
    """
    staged = {"samples": []}

    videos = [Path(p) for p in discovered.get("videos", [])]
    gifs = [Path(p) for p in discovered.get("gifs", [])]
    poses = [Path(p) for p in discovered.get("poses", [])]

    # Prefer mp4/webm videos for staging
    media = videos if videos else gifs

    if not media:
        print("\n[Stage] No videos/gifs discovered to stage.")
        return staged

    # Stage first N_SAMPLES items
    for i, src in enumerate(media[:N_SAMPLES]):
        sample_dir = output_dir / f"sample_{i:04d}"
        vid_dir = sample_dir / "videos"
        pose_dir = sample_dir / "poses"
        vid_dir.mkdir(parents=True, exist_ok=True)
        pose_dir.mkdir(parents=True, exist_ok=True)

        dst = vid_dir / src.name
        shutil.copy2(src, dst)

        # Naively associate poses: copy any pose files that look related by time proximity
        # (Better: parse DFoT naming once you see one real run.)
        # Here we just copy up to 3 pose files for convenience.
        for j, p in enumerate(poses[i*3:(i+1)*3]):
            shutil.copy2(p, pose_dir / p.name)

        staged["samples"].append({
            "id": i,
            "video": str(dst),
            "poses_dir": str(pose_dir),
            "note": "Pose association is heuristic; adjust once you inspect DFoT filenames."
        })

    return staged


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    print("=" * 60)
    print("STEP 1: Generate videos with DFoT")
    print("=" * 60)

    output_dir = RUNS_DIR / "generated"
    n_frames = K_HISTORY + T_FUTURE

    dfot_env = _numpy_fix_env()

    preflight_or_die(output_dir, dfot_env)

    # Write the pose-saving runner into the DFoT repo
    runner_path = create_step1_runner(DFOT_REPO)
    print(f"  Runner: {runner_path}")

    # Pass output dir and K_HISTORY to the runner via env
    dfot_env["STEP1_OUTPUT_DIR"] = str(output_dir.resolve())
    dfot_env["STEP1_K_HISTORY"] = str(K_HISTORY)

    # Checkpoint was trained with realestate10k_video_generation.yaml overrides
    # Must match EXACT architecture: channels [128,256,576,1152], 20 mid blocks, 9 heads
    cmd = [
        "python", "_step1_runner.py",
        "+name=action_mismatch_step1",
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
        f"experiment.validation.batch_size=1",
        f"dataset.num_eval_videos={N_SAMPLES}",
        # Default max_num_videos=8 in dfot_video.yaml caps NPZ output regardless of
        # num_eval_videos. Must match N_SAMPLES so all clips get saved to raw_dir.
        f"algorithm.logging.max_num_videos={N_SAMPLES}",
        f"algorithm.tasks.prediction.history_guidance.name={HISTORY_GUIDANCE_NAME}",
        f"+algorithm.tasks.prediction.history_guidance.guidance_scale={HISTORY_GUIDANCE_SCALE}",
        # Wandb config
        f"wandb.entity={WANDB_ENTITY}",
        "wandb.mode=offline",
        # Fix resumable dataloader check (both must agree: both off for validation-only)
        "dataset.subdataset_size=null",
        "experiment.reload_dataloaders_every_n_epochs=0",
        # Backbone architecture matching the TRAINED checkpoint
        "algorithm.backbone.channels=[128,256,576,1152]",
        "algorithm.backbone.num_updown_blocks=[3,3,6]",
        "algorithm.backbone.num_mid_blocks=20",
        "algorithm.backbone.num_heads=9",
        # Continuous diffusion params (required by code, not in @diffusion/continuous)
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
        f"++algorithm.logging.raw_dir={str((output_dir / '_raw_outputs').resolve())}",
    ]

    # Run DFoT via runner
    result = _run(cmd, cwd=DFOT_REPO, timeout=3600, env=dfot_env)

    if result.returncode != 0:
        if runner_path.exists():
            runner_path.unlink()
        print("\n[DFoT FAILED]")
        print("stdout (tail):\n", result.stdout[-4000:])
        print("stderr (tail):\n", result.stderr[-4000:])
        raise RuntimeError(
            "DFoT run failed. The error above is the real cause; fix that first "
            "(usually env/deps/checkpoint/dataset path)."
        )

    print("\n[DFoT OK]")
    print("stdout (tail):\n", result.stdout[-1500:])

    # Clean up runner
    if runner_path.exists():
        runner_path.unlink()

    # Verify GT poses were saved
    pose_files = sorted(output_dir.glob("sample_*/poses_gt_future.npy"))
    print(f"\n[GT Poses] Saved {len(pose_files)} poses_gt_future.npy files")
    if len(pose_files) == 0:
        print("  WARNING: No GT poses saved — check runner output above for errors.")

    # Discover outputs
    discovered = collect_media_from_outputs(DFOT_REPO, N_SAMPLES)
    print("\n[Discovered]")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in discovered.items()}, indent=2))

    # Stage outputs
    staged = stage_into_samples(output_dir, discovered)

    # Extract predicted future frames from NPZ into gen_frames/
    print("\n[Frame Extraction] Extracting predicted frames from NPZ...")
    extract_count = 0
    raw_dir = output_dir / '_raw_outputs'
    for i in range(N_SAMPLES):
        sample_dir = output_dir / f"sample_{i:04d}"
        frames_dir = sample_dir / "gen_frames"
        npz_path = raw_dir / str(i) / "data.npz"
        
        if not npz_path.exists():
            continue
            
        try:
            from PIL import Image
            frames_dir.mkdir(exist_ok=True)
            data = np.load(npz_path)
            if "gen" not in data:
                print(f"  {sample_dir.name}: NPZ missing 'gen' key (keys: {list(data.keys())})")
                continue
            gen = data["gen"]  # (T, C, H, W)

            # Validate frames aren't all black
            for t in range(gen.shape[0]):
                if gen[t].mean() < 1.0:
                    print(f"  WARNING: {sample_dir.name} frame {t} is black (mean={gen[t].mean():.2f})")

            frame_idx = 0
            for t in range(K_HISTORY, gen.shape[0]):
                gen_frame = np.transpose(gen[t], (1, 2, 0))  # (H, W, C)
                img = Image.fromarray(gen_frame)
                img.save(frames_dir / f"frame_{frame_idx:04d}.png")
                frame_idx += 1

            print(f"  {sample_dir.name}: extracted {frame_idx} frames → gen_frames/")
            extract_count += 1
        except Exception as e:
            print(f"  {sample_dir.name}: frame extraction failed — {e}")

    # Clean up raw npz files
    if raw_dir.exists():
        try:
            shutil.rmtree(raw_dir)
        except Exception as e:
            print(f"  [WARN] Could not remove {raw_dir}: {e} (non-fatal, continuing)")

    print(f"[Frame Extraction] Done ({extract_count}/{N_SAMPLES} samples)")

    manifest = {
        "n_samples": N_SAMPLES,
        "k_history": K_HISTORY,
        "t_future": T_FUTURE,
        "frame_skip": FRAME_SKIP,
        "dfot_checkpoint": str(DFOT_CHECKPOINT),
        "discovered": discovered,
        "staged": staged,
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[Done] Wrote manifest: {output_dir / 'manifest.json'}")
    print(f"[Done] Staged samples under: {output_dir}")


if __name__ == "__main__":
    main()
