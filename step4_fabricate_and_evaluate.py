#!/usr/bin/env python3
"""
Step 4: Fabricate mismatched history and measure degradation.

Hyunwoo's instructions:
  "Using this discrepancy statistics, fabricate a history with ground-truth 
   video and inaccurate action."
  "Assess the impact of this inaccurate pose in terms of 
   1) perceptual quality and 2) future pose accuracy."

This script:
  1. Takes drift statistics from Phase 1 (step3 outputs)
  2. Creates corrupted pose histories (GT video + wrong poses)
  3. Regenerates videos with corrupted vs clean history
  4. Evaluates:
     a) Perceptual quality (LPIPS between generated and GT frames)
     b) Future pose accuracy (rotation error on newly generated futures)
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RUNS_DIR, DFOT_REPO, VGGT_REPO, VGGT_DEVICE,
    N_PHASE2_CLIPS, CORRUPTION_SCALES, SEED, T_FUTURE, K_HISTORY
)
from utils.pose_utils import (
    apply_rotation_perturbation,
    compute_cumulative_rotation_error,
    compute_rotation_errors_over_time,
)


def main():
    print("=" * 60)
    print("STEP 4: Fabricate mismatched history + measure impact")
    print("=" * 60)
    
    aggregate_dir = RUNS_DIR / "aggregate"
    phase2_dir = RUNS_DIR / "phase2"
    phase2_dir.mkdir(parents=True, exist_ok=True)
    
    # ================================================================
    # Step 4.1: Build drift library from Phase 1
    # ================================================================
    print("\n--- Step 4.1: Building drift library from Phase 1 ---")
    
    csv_path = aggregate_dir / "mismatch_all.csv"
    if not csv_path.exists():
        print(f"  ERROR: {csv_path} not found. Run step3 first.")
        return
    
    df = pd.read_csv(csv_path)
    df_cum = df[df["error_type"] == "cumulative"]
    
    # Get final-frame errors for each sample
    final_errors = df_cum.groupby("sample_id")["rot_err_deg"].apply(lambda x: x.iloc[-1]).sort_values()
    
    print(f"  Drift statistics from Phase 1:")
    print(f"  10th percentile (low drift):  {final_errors.quantile(0.1):.2f}°")
    print(f"  50th percentile (median):     {final_errors.quantile(0.5):.2f}°")
    print(f"  90th percentile (high drift): {final_errors.quantile(0.9):.2f}°")
    
    # Extract representative drift trajectories
    # Low drift: 10th percentile sample
    low_drift_sample = final_errors.index[max(0, int(len(final_errors) * 0.1))]
    # High drift: 90th percentile sample
    high_drift_sample = final_errors.index[min(len(final_errors) - 1, int(len(final_errors) * 0.9))]
    
    # Get the cumulative error trajectory for these samples
    drift_low = df_cum[df_cum["sample_id"] == low_drift_sample].sort_values("t")["rot_err_deg"].values
    drift_high = df_cum[df_cum["sample_id"] == high_drift_sample].sort_values("t")["rot_err_deg"].values
    
    # Use median drift across all samples as the "typical" drift
    drift_median = df_cum.groupby("t")["rot_err_deg"].median().values
    
    print(f"  Low-drift sample:  {low_drift_sample} (final err: {drift_low[-1]:.2f}°)")
    print(f"  High-drift sample: {high_drift_sample} (final err: {drift_high[-1]:.2f}°)")
    
    # ================================================================
    # Step 4.2: Corrupt pose history
    # ================================================================
    print("\n--- Step 4.2: Creating corrupted pose histories ---")
    
    generated_dir = RUNS_DIR / "generated"
    sample_dirs = sorted(generated_dir.glob("sample_*"))[:N_PHASE2_CLIPS]
    
    if not sample_dirs:
        print(f"  ERROR: No samples found. Run step1 first.")
        return
    
    results = []
    
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        gt_pose_path = sample_dir / "poses_gt_future.npy"
        
        if not gt_pose_path.exists():
            continue
        
        poses_gt = np.load(gt_pose_path)
        T = len(poses_gt)
        
        # For each corruption level, create corrupted poses
        for scale in CORRUPTION_SCALES:
            # Use median drift trajectory, scaled
            drift = drift_median[:T] if len(drift_median) >= T else np.pad(
                drift_median, (0, T - len(drift_median)), mode='edge'
            )
            
            corrupted_poses = apply_rotation_perturbation(poses_gt, drift, scale=scale)
            
            # Save corrupted poses
            corrupt_dir = phase2_dir / f"{sample_id}_scale{scale:.1f}"
            corrupt_dir.mkdir(parents=True, exist_ok=True)
            np.save(corrupt_dir / "poses_corrupted.npy", corrupted_poses)
            np.save(corrupt_dir / "poses_clean.npy", poses_gt)
            
            # Compute the actual corruption magnitude
            cum_err = compute_cumulative_rotation_error(corrupted_poses, poses_gt)
            
            results.append({
                "sample_id": sample_id,
                "corruption_scale": scale,
                "corruption_final_deg": cum_err[-1],
            })
            
            print(f"  {sample_id} scale={scale:.1f}: final corruption = {cum_err[-1]:.2f}°")
    
    # ================================================================
    # Step 4.3: Regenerate videos with corrupted history
    # ================================================================
    print("\n--- Step 4.3: Regenerating with corrupted history ---")
    print("  ")
    print("  To regenerate videos with corrupted poses, you have two options:")
    print("  ")
    print("  OPTION A (Recommended): Modify RE10k data files")
    print("    1. Copy the RE10k clip data to a new directory")
    print("    2. Replace the pose values with corrupted ones")
    print("    3. Run DFoT on this modified dataset")
    print("    This keeps DFoT's code unchanged.")
    print("  ")
    print("  OPTION B: Direct model API (requires more DFoT internals)")
    print("    1. Load the DFoT model programmatically")  
    print("    2. Call its generation function with custom pose tensors")
    print("    3. Requires understanding DFoT's internal sample() API")
    print("  ")
    
    # Create helper script for Option A
    create_corruption_helper(phase2_dir, sample_dirs)
    
    # ================================================================
    # Step 4.4: Evaluate impact
    # ================================================================
    print("\n--- Step 4.4: Evaluation template ---")
    
    # Save Phase 2 manifest
    phase2_manifest = {
        "drift_statistics": {
            "p10": float(final_errors.quantile(0.1)),
            "p50": float(final_errors.quantile(0.5)),
            "p90": float(final_errors.quantile(0.9)),
        },
        "corruption_scales": CORRUPTION_SCALES,
        "n_clips": len(sample_dirs),
        "results": results,
    }
    
    with open(phase2_dir / "phase2_manifest.json", "w") as f:
        json.dump(phase2_manifest, f, indent=2)
    
    # Save results CSV
    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(phase2_dir / "phase2_results.csv", index=False)
        print(f"\n  Saved: {phase2_dir / 'phase2_results.csv'}")
    
    # Generate evaluation comparison plot
    make_phase2_plots(phase2_dir, results)
    
    print(f"\n  ✅ Phase 2 corruption setup complete!")
    print(f"  Corrupted poses saved to {phase2_dir}/")
    print(f"  ")
    print(f"  TO COMPLETE PHASE 2:")
    print(f"  1. Regenerate videos with corrupted poses (see options above)")
    print(f"  2. Run step2 (pose estimation) on the new videos")
    print(f"  3. Compare clean vs corrupted pose accuracy & perceptual quality")


def create_corruption_helper(phase2_dir: Path, sample_dirs: list):
    """
    Create a helper script that modifies RE10k pose files for corrupted generation.
    """
    helper_path = phase2_dir / "regenerate_with_corruption.sh"
    
    script = """#!/bin/bash
# Helper script to regenerate videos with corrupted poses.
#
# This creates a modified RE10k dataset with corrupted poses,
# then runs DFoT on it.
#
# Usage: bash regenerate_with_corruption.sh

DFOT_REPO="{dfot_repo}"
PHASE2_DIR="{phase2_dir}"

echo "=== Regenerating with corrupted poses ==="

# For each corruption condition, run DFoT generation:
# 1. Clean history (baseline)
echo "Generating with CLEAN poses (baseline)..."
cd $DFOT_REPO
python -m main +name=phase2_clean dataset=realestate10k_mini algorithm=dfot_video_pose \\
  experiment=video_generation @diffusion/continuous load=pretrained:DFoT_RE10K.ckpt \\
  'experiment.tasks=[validation]' experiment.validation.batch_size=1 \\
  dataset.context_length={k_hist} dataset.frame_skip=20 dataset.n_frames={n_frames} \\
  dataset.num_eval_videos={n_clips} \\
  algorithm.tasks.prediction.history_guidance.name=vanilla \\
  +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0

echo ""
echo "=== NEXT STEP ==="
echo "To generate with CORRUPTED poses, you need to modify the"
echo "RE10k dataset loader in DFoT to accept custom pose overrides."
echo ""
echo "Look at: datasets/realestate10k/ in the DFoT repo"
echo "The dataset __getitem__ returns poses — you can intercept there."
echo ""
echo "Alternatively, write a custom script that:"
echo "  1. Loads the DFoT model"
echo "  2. Loads history frames from RE10k"  
echo "  3. Feeds corrupted poses from $PHASE2_DIR"
echo "  4. Runs generation"
""".format(
        dfot_repo=str(DFOT_REPO),
        phase2_dir=str(phase2_dir),
        k_hist=K_HISTORY,
        n_frames=K_HISTORY + T_FUTURE,
        n_clips=N_PHASE2_CLIPS,
    )
    
    with open(helper_path, "w") as f:
        f.write(script)
    
    helper_path.chmod(0o755)
    print(f"  Helper script: {helper_path}")


def make_phase2_plots(phase2_dir: Path, results: list):
    """Generate Phase 2 comparison plots."""
    
    if not results:
        return
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    
    df = pd.DataFrame(results)
    
    # Plot: Corruption magnitude vs scale factor
    fig, ax = plt.subplots(figsize=(8, 5))
    
    for scale in sorted(df["corruption_scale"].unique()):
        subset = df[df["corruption_scale"] == scale]
        vals = subset["corruption_final_deg"].values
        ax.scatter(
            [scale] * len(vals), vals, 
            alpha=0.6, s=50, label=f"Scale {scale:.1f}"
        )
        ax.plot(scale, np.median(vals), "k_", markersize=20, markeredgewidth=3)
    
    ax.set_xlabel("Corruption Scale Factor", fontsize=14)
    ax.set_ylabel("Applied Rotation Corruption (degrees)", fontsize=14)
    ax.set_title("Fabricated Pose Corruption Magnitudes", fontsize=16)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plot_path = phase2_dir / "corruption_magnitudes.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
