#!/usr/bin/env python3
"""
Step 3: Compute pose mismatch between GT and estimated poses.

Hyunwoo's instruction:
  "the first step is to identify the amount of imprecision caused by 
   the accumulation of divergence between the actions and observations"

This script:
  1. Loads GT poses (what we asked DFoT to generate) and estimated poses (what it actually generated)
  2. Computes rotation error at each timestep
  3. Saves mismatch_all.csv with columns: sample_id, t, rot_err_deg
  4. Generates two plots:
     - Plot 1: rot_err_deg vs timestep (median + quartiles across samples)
     - Plot 2: histogram of rot_err_deg at the final timestep
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import RUNS_DIR, MAX_BASELINE_DRIFT
from utils.pose_utils import (
    rotation_error_deg,
    compute_rotation_errors_over_time,
    compute_cumulative_rotation_error,
    extrinsic_to_R_t,
)


def main():
    print("=" * 60)
    print("STEP 3: Compute pose mismatch")
    print("=" * 60)
    
    generated_dir = RUNS_DIR / "generated"
    aggregate_dir = RUNS_DIR / "aggregate"
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all samples that have both GT and estimated poses
    sample_dirs = sorted(generated_dir.glob("sample_*"))
    
    all_rows = []
    cumulative_errors_all = []
    
    for sample_dir in sample_dirs:
        sample_id = sample_dir.name
        
        gt_path = sample_dir / "poses_gt_future.npy"
        est_path = sample_dir / "poses_est_from_gen.npy"
        
        if not gt_path.exists():
            print(f"  Skipping {sample_id}: no GT poses")
            continue
        if not est_path.exists():
            print(f"  Skipping {sample_id}: no estimated poses (run step2 first)")
            continue
        
        poses_gt = np.load(gt_path)
        poses_est = np.load(est_path)
        
        T = min(len(poses_gt), len(poses_est))
        poses_gt = poses_gt[:T]
        poses_est = poses_est[:T]
        
        print(f"  {sample_id}: {T} frames")
        
        # Compute per-frame relative rotation errors
        rel_errors = compute_rotation_errors_over_time(poses_est, poses_gt, use_relative=True)
        
        # Compute cumulative rotation drift
        cum_errors = compute_cumulative_rotation_error(poses_est, poses_gt)
        cumulative_errors_all.append(cum_errors)
        
        # Store relative errors (T-1 values since they're frame-to-frame)
        for t_idx, err in enumerate(rel_errors):
            all_rows.append({
                "sample_id": sample_id,
                "t": t_idx + 1,  # t=0 is the first frame, relative error starts at t=1
                "rot_err_deg": err,
                "error_type": "relative",
            })
        
        # Store cumulative errors (T values)
        for t_idx, err in enumerate(cum_errors):
            all_rows.append({
                "sample_id": sample_id,
                "t": t_idx,
                "rot_err_deg": err,
                "error_type": "cumulative",
            })
        
        # Print summary for this sample
        print(f"    Relative rot error: mean={rel_errors.mean():.2f}°, max={rel_errors.max():.2f}°")
        print(f"    Cumulative drift at final frame: {cum_errors[-1]:.2f}°")
    
    if not all_rows:
        print("\n  ERROR: No samples with both GT and estimated poses found.")
        print("  Make sure step1 and step2 completed successfully.")
        return
    
    # Save CSV
    df = pd.DataFrame(all_rows)
    csv_path = aggregate_dir / "mismatch_all.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")
    
    # Print overall statistics
    df_cum = df[df["error_type"] == "cumulative"]
    df_rel = df[df["error_type"] == "relative"]
    
    print(f"\n  OVERALL STATISTICS:")
    print(f"  Samples analyzed: {df_cum['sample_id'].nunique()}")
    
    if len(df_rel) > 0:
        print(f"\n  Per-frame relative rotation error:")
        print(f"    Mean:   {df_rel['rot_err_deg'].mean():.2f}°")
        print(f"    Median: {df_rel['rot_err_deg'].median():.2f}°")
        print(f"    Std:    {df_rel['rot_err_deg'].std():.2f}°")
    
    # Final-frame cumulative drift
    final_frame_errors = df_cum.groupby("sample_id")["rot_err_deg"].apply(lambda x: x.iloc[-1])
    print(f"\n  Cumulative drift at final frame:")
    print(f"    Mean:   {final_frame_errors.mean():.2f}°")
    print(f"    Median: {final_frame_errors.median():.2f}°")
    print(f"    10th %: {final_frame_errors.quantile(0.1):.2f}°")
    print(f"    90th %: {final_frame_errors.quantile(0.9):.2f}°")
    
    # Filter clean samples (low baseline drift) for Phase 2
    clean_ids = final_frame_errors[final_frame_errors <= MAX_BASELINE_DRIFT].index.tolist()
    clean_indices = [int(sid.replace("sample_", "")) for sid in clean_ids]

    clean_manifest = {
        "threshold_deg": MAX_BASELINE_DRIFT,
        "n_total": len(final_frame_errors),
        "n_clean": len(clean_ids),
        "clean_sample_ids": clean_ids,
        "clean_sample_indices": clean_indices,
    }
    clean_path = aggregate_dir / "clean_samples.json"
    with open(clean_path, "w") as f:
        json.dump(clean_manifest, f, indent=2)

    print(f"\n  Drift filtering (threshold={MAX_BASELINE_DRIFT}°):")
    print(f"    {len(clean_ids)}/{len(final_frame_errors)} samples pass → saved {clean_path}")
    if len(clean_ids) > 0:
        clean_drifts = final_frame_errors[final_frame_errors <= MAX_BASELINE_DRIFT]
        print(f"    Clean subset median drift: {clean_drifts.median():.2f}°")

    # Generate plots
    make_plots(df, aggregate_dir)

    print(f"\n  ✅ Phase 1 complete!")
    print(f"  You've quantified the accumulation of divergence.")
    print(f"  Next: run step4_fabricate_and_evaluate.py for Phase 2.")


def make_plots(df: pd.DataFrame, output_dir: Path):
    """Generate the two key plots for Phase 1."""
    
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available, skipping plots")
        print("  Install: pip install matplotlib")
        return
    
    # ================================================================
    # Plot 1: Cumulative rotation error vs timestep
    # ================================================================
    df_cum = df[df["error_type"] == "cumulative"]
    
    if len(df_cum) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by timestep, compute statistics
        stats = df_cum.groupby("t")["rot_err_deg"].agg(["median", "mean"])
        q25 = df_cum.groupby("t")["rot_err_deg"].quantile(0.25)
        q75 = df_cum.groupby("t")["rot_err_deg"].quantile(0.75)
        
        t_vals = stats.index.values
        
        ax.plot(t_vals, stats["median"], "b-", linewidth=2, label="Median")
        ax.fill_between(t_vals, q25.values, q75.values, alpha=0.3, color="blue", label="25th–75th percentile")
        ax.plot(t_vals, stats["mean"], "r--", linewidth=1, label="Mean")
        
        ax.set_xlabel("Timestep", fontsize=14)
        ax.set_ylabel("Cumulative Rotation Error (degrees)", fontsize=14)
        ax.set_title("Accumulation of Pose-Video Divergence Over Time", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        
        plot1_path = output_dir / "rot_err_vs_time.png"
        fig.savefig(plot1_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot 1 saved: {plot1_path}")
    
    # ================================================================
    # Plot 2: Histogram of rotation error at final timestep
    # ================================================================
    final_errors = df_cum.groupby("sample_id")["rot_err_deg"].apply(lambda x: x.iloc[-1])
    
    if len(final_errors) > 0:
        fig, ax = plt.subplots(figsize=(8, 5))
        
        ax.hist(final_errors.values, bins=20, edgecolor="black", alpha=0.7, color="steelblue")
        ax.axvline(final_errors.median(), color="red", linestyle="--", linewidth=2,
                    label=f"Median: {final_errors.median():.1f}°")
        
        ax.set_xlabel("Cumulative Rotation Error at Final Frame (degrees)", fontsize=14)
        ax.set_ylabel("Count", fontsize=14)
        ax.set_title("Distribution of Final-Frame Pose Drift", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plot2_path = output_dir / "rot_err_histogram.png"
        fig.savefig(plot2_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot 2 saved: {plot2_path}")
    
    # ================================================================
    # Plot 3: Per-frame relative error (bonus)
    # ================================================================
    df_rel = df[df["error_type"] == "relative"]
    
    if len(df_rel) > 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        stats = df_rel.groupby("t")["rot_err_deg"].agg(["median", "mean"])
        q25 = df_rel.groupby("t")["rot_err_deg"].quantile(0.25)
        q75 = df_rel.groupby("t")["rot_err_deg"].quantile(0.75)
        
        t_vals = stats.index.values
        
        ax.plot(t_vals, stats["median"], "g-", linewidth=2, label="Median")
        ax.fill_between(t_vals, q25.values, q75.values, alpha=0.3, color="green", label="25th–75th percentile")
        
        ax.set_xlabel("Timestep", fontsize=14)
        ax.set_ylabel("Per-Frame Relative Rotation Error (degrees)", fontsize=14)
        ax.set_title("Per-Frame Pose Accuracy Over Time", fontsize=16)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plot3_path = output_dir / "rel_rot_err_vs_time.png"
        fig.savefig(plot3_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Plot 3 saved: {plot3_path}")


if __name__ == "__main__":
    main()
