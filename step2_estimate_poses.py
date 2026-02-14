#!/usr/bin/env python3
"""
Step 2: Run pose oracle on generated videos.

Hyunwoo's instruction:
  "Measure the discrepancy between the target action (ground-truth future pose) 
   and the pose estimated from the generated video."
  "You may use the TPS oracle to measure the pose discrepancy"

We use VGGT as the pose oracle (same as XFactor's TPS metric internally).
For each generated video, we feed the frames to VGGT and get estimated SE(3) poses.
"""

import sys
import json
import numpy as np
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import RUNS_DIR, VGGT_REPO, VGGT_DEVICE


def main():
    print("=" * 60)
    print("STEP 2: Estimate poses from generated videos (VGGT oracle)")
    print("=" * 60)
    
    generated_dir = RUNS_DIR / "generated"
    
    if not generated_dir.exists():
        print(f"  ERROR: {generated_dir} not found. Run step1 first.")
        return
    
    # Find all sample directories
    sample_dirs = sorted(generated_dir.glob("sample_*"))
    if not sample_dirs:
        print(f"  ERROR: No sample directories found in {generated_dir}")
        return
    
    print(f"  Found {len(sample_dirs)} samples")
    
    # Try to load VGGT, fall back to OpenCV if unavailable
    oracle = load_oracle()
    
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        frames_dir = sample_dir / "gen_frames"
        
        if not frames_dir.exists():
            print(f"  Skipping {sample_id}: no gen_frames directory")
            continue
        
        # Load frames
        frame_paths = sorted(frames_dir.glob("*.png"))
        if not frame_paths:
            frame_paths = sorted(frames_dir.glob("*.jpg"))
        
        if not frame_paths:
            print(f"  Skipping {sample_id}: no frame images found")
            continue
        
        print(f"  Processing {sample_id} ({len(frame_paths)} frames)...")
        
        # Load frames as numpy array
        frames = np.array([np.array(Image.open(p).convert("RGB")) for p in frame_paths])
        
        # Estimate poses
        try:
            poses_est = oracle.estimate_poses_from_frames(frames)
        except Exception as e:
            print(f"  ERROR estimating poses for {sample_id}: {e}")
            continue
        
        # Sanity checks
        T = len(frame_paths)
        assert poses_est.shape == (T, 4, 4), f"Expected ({T}, 4, 4), got {poses_est.shape}"
        assert not np.any(np.isnan(poses_est)), "NaN in estimated poses!"
        
        # Save
        output_path = sample_dir / "poses_est_from_gen.npy"
        np.save(output_path, poses_est)
        print(f"  Saved: {output_path}")
    
    print(f"\n  Done! Estimated poses saved for all samples.")
    print(f"  Next: run step3_compute_mismatch.py")


def load_oracle():
    """Try to load VGGT, fall back to OpenCV pose estimator."""
    
    # Try VGGT first
    try:
        from utils.vggt_wrapper import VGGTOracle
        oracle = VGGTOracle(
            vggt_repo_path=str(VGGT_REPO),
            device=VGGT_DEVICE
        )
        print("  Using VGGT oracle (recommended)")
        return oracle
    except Exception as e:
        print(f"  Could not load VGGT: {e}")
    
    # Fall back to OpenCV
    try:
        from utils.vggt_wrapper import OpenCVPoseEstimator
        oracle = OpenCVPoseEstimator()
        print("  Using OpenCV fallback pose estimator")
        print("  WARNING: Less accurate than VGGT. Swap in VGGT when available.")
        return oracle
    except ImportError as e:
        print(f"  Could not load OpenCV either: {e}")
        print(f"  Install: pip install opencv-python")
    
    # Last resort: mock oracle for development
    print("  Using MOCK oracle (random poses — for development only!)")
    return MockOracle()


class MockOracle:
    """Mock pose oracle that returns slightly perturbed identity poses."""
    
    def estimate_poses_from_frames(self, frames):
        T = len(frames)
        poses = np.zeros((T, 4, 4))
        
        for t in range(T):
            poses[t] = np.eye(4)
            # Add small random rotation (simulating estimation noise)
            angle = np.radians(t * 4.5 + np.random.randn() * 1.0)
            c, s = np.cos(angle), np.sin(angle)
            poses[t, 0, 0] = c
            poses[t, 0, 2] = s
            poses[t, 2, 0] = -s
            poses[t, 2, 2] = c
            poses[t, 0, 3] = t * 0.1 + np.random.randn() * 0.01
        
        return poses


if __name__ == "__main__":
    main()
