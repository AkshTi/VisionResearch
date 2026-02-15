#!/usr/bin/env python3
"""
Step 1: Generate videos with the pose-conditioned DFoT model.

Runs DFoT via its Hydra CLI, then auto-discovers outputs and copies them into:
  RUNS_DIR/generated/sample_XXXX/{videos,gen_frames,poses}/...

If DFoT fails, prints the *real* error and exits non-zero by default.
"""

import sys
import json
import subprocess
import shutil
import re
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


# ----------------------------
# Utilities
# ----------------------------

def _run(cmd: List[str], cwd: Path, timeout: int = 3600) -> subprocess.CompletedProcess:
    """Run a command and return CompletedProcess. Does NOT swallow errors."""
    print("\n[RUN]")
    print(f"  cwd: {cwd}")
    print("  cmd:", " ".join(cmd))
    return subprocess.run(
        cmd,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout
    )


def preflight_or_die(output_dir: Path) -> None:
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
    test = _run(["python", "-c", "import main; print('ok')"], cwd=DFOT_REPO, timeout=120)
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

    preflight_or_die(output_dir)

    # README command + explicit backbone architecture + continuous diffusion params
    # Force backbone to match pretrained checkpoint: channels [128,256,512,1024], mid_blocks 16
    cmd = [
        "python", "-m", "main",
        "+name=action_mismatch_step1",
        "dataset=realestate10k_mini",
        "algorithm=dfot_video_pose",
        "experiment=video_generation",
        "@diffusion/continuous",
        f"load=pretrained:{DFOT_CHECKPOINT}",
        "experiment.tasks=[validation]",
        "experiment.validation.data.shuffle=False",
        f"dataset.context_length={K_HISTORY}",
        f"dataset.frame_skip={FRAME_SKIP}",
        f"dataset.n_frames={n_frames}",
        f"experiment.validation.batch_size=1",
        f"dataset.num_eval_videos={N_SAMPLES}",
        f"algorithm.tasks.prediction.history_guidance.name={HISTORY_GUIDANCE_NAME}",
        f"+algorithm.tasks.prediction.history_guidance.guidance_scale={HISTORY_GUIDANCE_SCALE}",
        # Wandb config
        f"wandb.entity={WANDB_ENTITY}",
        "wandb.mode=offline",
        # Explicitly set backbone to match pretrained checkpoint (prevent overrides)
        "algorithm.backbone.channels=[128,256,512,1024]",
        "algorithm.backbone.num_mid_blocks=16",
        "algorithm.backbone.num_updown_blocks=[3,3,3]",
        "algorithm.backbone.num_heads=4",
        # Add missing continuous diffusion params (required by code)
        "++algorithm.diffusion.training_schedule.name=cosine",
        "++algorithm.diffusion.training_schedule.shift=0.125",
        "++algorithm.diffusion.loss_weighting.strategy=sigmoid",
        "++algorithm.diffusion.loss_weighting.sigmoid_bias=-1.0",
    ]

    # Run DFoT
    result = _run(cmd, cwd=DFOT_REPO, timeout=3600)

    if result.returncode != 0:
        print("\n[DFoT FAILED]")
        print("stdout (tail):\n", result.stdout[-4000:])
        print("stderr (tail):\n", result.stderr[-4000:])
        raise RuntimeError(
            "DFoT run failed. The error above is the real cause; fix that first "
            "(usually env/deps/checkpoint/dataset path)."
        )

    print("\n[DFoT OK]")
    print("stdout (tail):\n", result.stdout[-1500:])

    # Discover outputs
    discovered = collect_media_from_outputs(DFOT_REPO, N_SAMPLES)
    print("\n[Discovered]")
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in discovered.items()}, indent=2))

    # Stage outputs
    staged = stage_into_samples(output_dir, discovered)

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
