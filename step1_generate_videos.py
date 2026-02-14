#!/usr/bin/env python3
"""
Step 1: Generate videos with the pose-conditioned DFoT model.

Hyunwoo's instruction:
  "First, take the pose-conditioned DFoT (history-guided diffusion) 
   pretrained RE10k dataset as your world model."

This script generates N videos using DFoT conditioned on RE10k poses,
saving the generated frames and the ground-truth poses used for conditioning.
"""

import sys
import json
import subprocess
import shutil
import numpy as np
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DFOT_REPO, RUNS_DIR, N_SAMPLES, K_HISTORY, T_FUTURE,
    FRAME_SKIP, DFOT_CHECKPOINT, HISTORY_GUIDANCE_NAME,
    HISTORY_GUIDANCE_SCALE, SEED, WANDB_ENTITY
)


def main():
    print("=" * 60)
    print("STEP 1: Generate videos with DFoT")
    print("=" * 60)
    
    # Create output directory
    output_dir = RUNS_DIR / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    n_frames = K_HISTORY + T_FUTURE
    
    # ================================================================
    # OPTION A: Use DFoT's standard CLI (recommended for first run)
    # ================================================================
    print(f"\nGenerating {N_SAMPLES} videos with DFoT...")
    print(f"  History frames: {K_HISTORY}")
    print(f"  Future frames:  {T_FUTURE}")
    print(f"  Frame skip:     {FRAME_SKIP}")
    print(f"  Checkpoint:     {DFOT_CHECKPOINT}")
    
    # Build the DFoT command
    # DFoT needs training_schedule even for inference - add it with + prefix
    cmd = [
        "python", "main.py",
        f"+name=action_mismatch_step1",
        "dataset=realestate10k_mini",
        "algorithm=dfot_video_pose",
        "experiment=video_generation",
        "@diffusion/continuous",
        f"load=pretrained:{DFOT_CHECKPOINT}",
        f"wandb.entity={WANDB_ENTITY}",
        # Add training_schedule (required by model init, even for inference)
        "+algorithm.diffusion.training_schedule.name=cosine",
        "+algorithm.diffusion.training_schedule.shift=0.125",
        # Experiment settings
        "experiment.tasks=[validation]",
        "experiment.validation.data.shuffle=False",
        f"experiment.validation.batch_size=1",
        f"dataset.context_length={K_HISTORY}",
        f"dataset.frame_skip={FRAME_SKIP}",
        f"dataset.n_frames={n_frames}",
        f"dataset.num_eval_videos={N_SAMPLES}",
        f"algorithm.tasks.prediction.history_guidance.name={HISTORY_GUIDANCE_NAME}",
        f"+algorithm.tasks.prediction.history_guidance.guidance_scale={HISTORY_GUIDANCE_SCALE}",
    ]
    
    print(f"\n  Full command:")
    print(f"  cd {DFOT_REPO} && \\")
    print(f"  {' '.join(cmd)}")
    
    # Check if DFoT repo exists
    if not DFOT_REPO.exists():
        print(f"\n  ERROR: DFoT repo not found at {DFOT_REPO}")
        print(f"  Please clone it:")
        print(f"  git clone https://github.com/kwsong0113/diffusion-forcing-transformer.git {DFOT_REPO}")
        print(f"\n  Then edit config.py to set DFOT_REPO correctly.")
        return create_mock_outputs(output_dir)
    
    # Run DFoT
    try:
        result = subprocess.run(
            cmd,
            cwd=str(DFOT_REPO),
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            print(f"\n  DFoT failed. Last 1000 chars of stderr:")
            print(result.stderr[-1000:])
            print(f"\n  Falling back to mock outputs for development...")
            return create_mock_outputs(output_dir)
        
        print(f"\n  DFoT generation complete!")
        print(f"  Check wandb for logged videos and poses.")
        
        # Parse DFoT output to find generated files
        # DFoT logs to wandb; you'll need to find the wandb run directory
        return find_and_organize_dfot_outputs(output_dir, result.stdout)
        
    except FileNotFoundError:
        print(f"\n  Could not run DFoT (python not found or wrong env).")
        print(f"  Make sure you've activated the dfot conda env:")
        print(f"  conda activate mech_interp_gpu")
        return create_mock_outputs(output_dir)
    except subprocess.TimeoutExpired:
        print(f"\n  DFoT generation timed out after 1 hour.")
        return create_mock_outputs(output_dir)


def find_and_organize_dfot_outputs(output_dir: Path, stdout: str) -> dict:
    """
    After DFoT runs, find its outputs and organize them.
    
    DFoT saves generated videos and poses via wandb logging.
    The exact output location depends on the wandb run.
    """
    # Look for wandb run ID in stdout
    import re
    wandb_match = re.search(r'wandb: Run data is saved locally in (.+)', stdout)
    
    if wandb_match:
        wandb_dir = Path(wandb_match.group(1))
        print(f"  Found wandb outputs at: {wandb_dir}")
    
    # Also check the DFoT outputs directory
    dfot_outputs = DFOT_REPO / "outputs"
    if dfot_outputs.exists():
        # Find the most recent run
        runs = sorted(dfot_outputs.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
        if runs:
            print(f"  Latest DFoT output: {runs[0]}")
    
    manifest = {
        "n_samples": N_SAMPLES,
        "k_history": K_HISTORY,
        "t_future": T_FUTURE,
        "frame_skip": FRAME_SKIP,
        "note": "Check wandb for generated videos. Copy them to runs/action_mismatch/generated/",
    }
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n  Manifest saved to {output_dir / 'manifest.json'}")
    print(f"\n  NEXT STEPS:")
    print(f"  1. Find the generated videos in your wandb run or DFoT outputs/")
    print(f"  2. Copy/symlink them into {output_dir}/sample_XXXX/")
    print(f"  3. Also extract the GT poses that DFoT used for conditioning")
    print(f"  4. Then run step2_estimate_poses.py")
    
    return manifest


def create_mock_outputs(output_dir: Path) -> dict:
    """
    Create mock outputs for development/testing without GPU.
    
    This lets you develop steps 2-4 while waiting for DFoT access.
    The mock data has realistic structure but random content.
    """
    print(f"\n  Creating MOCK outputs for development...")
    
    np.random.seed(SEED)
    
    manifest = {"samples": []}
    
    for i in range(N_SAMPLES):
        sample_dir = output_dir / f"sample_{i:04d}"
        frames_dir = sample_dir / "gen_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock frames (small gray images with some variation)
        for t in range(T_FUTURE):
            img = np.random.randint(100, 200, (256, 256, 3), dtype=np.uint8)
            from PIL import Image
            Image.fromarray(img).save(frames_dir / f"frame_{t:04d}.png")
        
        # Create mock GT poses (smooth camera trajectory)
        poses_gt = np.zeros((T_FUTURE, 4, 4))
        for t in range(T_FUTURE):
            poses_gt[t] = np.eye(4)
            # Add smooth rotation around Y axis
            angle = np.radians(t * 5)  # 5 degrees per frame
            c, s = np.cos(angle), np.sin(angle)
            poses_gt[t, 0, 0] = c
            poses_gt[t, 0, 2] = s
            poses_gt[t, 2, 0] = -s
            poses_gt[t, 2, 2] = c
            # Add smooth translation
            poses_gt[t, 0, 3] = t * 0.1
            poses_gt[t, 2, 3] = t * 0.05
        
        np.save(sample_dir / "poses_gt_future.npy", poses_gt)
        
        # Save metadata
        meta = {
            "sample_id": i,
            "n_frames": T_FUTURE,
            "is_mock": True,
            "k_history": K_HISTORY,
        }
        with open(sample_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        manifest["samples"].append({
            "id": i,
            "frames_dir": str(frames_dir),
            "poses_gt_path": str(sample_dir / "poses_gt_future.npy"),
        })
    
    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"  Created {N_SAMPLES} mock samples in {output_dir}")
    print(f"  These have RANDOM data — replace with real DFoT outputs!")
    
    return manifest


if __name__ == "__main__":
    main()
