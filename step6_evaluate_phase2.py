#!/usr/bin/env python3
"""
Step 6: Evaluate Phase 2 — Impact of inaccurate pose conditioning.

Hyunwoo's criteria:
  1) Perceptual quality:  LPIPS between generated and GT frames
  2) Future pose accuracy: rotation error of VGGT-estimated poses vs GT poses

Compares across corruption scales (0=clean, 0.5, 1.0, 2.0).
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    RUNS_DIR, VGGT_REPO, VGGT_DEVICE,
    CORRUPTION_SCALES, K_HISTORY, T_FUTURE, N_SAMPLES,
)
from utils.pose_utils import (
    compute_rotation_errors_over_time,
    compute_cumulative_rotation_error,
)


# ============================================================
# Helpers
# ============================================================

def load_frames(frames_dir: Path):
    """Load PNG frames from a directory as numpy arrays."""
    frame_paths = sorted(frames_dir.glob("*.png"))
    if not frame_paths:
        return []
    frames = []
    for p in frame_paths:
        arr = np.array(Image.open(p).convert("RGB"))
        if arr.mean() < 1.0:
            print(f"  WARNING: {p.name} is black (mean={arr.mean():.2f}) — "
                  "this will corrupt evaluation metrics")
        frames.append(arr)
    return frames


def compute_lpips_scores(frames_a, frames_b, device="cuda"):
    """
    Compute per-frame LPIPS between two lists of numpy frames.
    Returns list of per-frame scores, or None if lpips is unavailable.
    """
    try:
        import torch
        import lpips
    except ImportError:
        print("  WARNING: lpips not installed. Run: pip install lpips")
        return None

    loss_fn = lpips.LPIPS(net="alex").to(device)

    scores = []
    for fa, fb in zip(frames_a, frames_b):
        # (H, W, 3) uint8 -> (1, 3, H, W) float in [-1, 1]
        ta = torch.from_numpy(fa).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
        tb = torch.from_numpy(fb).permute(2, 0, 1).float().unsqueeze(0) / 127.5 - 1.0
        with torch.no_grad():
            score = loss_fn(ta.to(device), tb.to(device)).item()
        scores.append(score)

    return scores


def load_oracle():
    """Load VGGT (or fallback) pose oracle — same as step2."""
    try:
        from utils.vggt_wrapper import VGGTOracle
        oracle = VGGTOracle(vggt_repo_path=str(VGGT_REPO), device=VGGT_DEVICE)
        print("  Pose oracle: VGGT")
        return oracle
    except Exception as e:
        print(f"  Could not load VGGT: {e}")

    try:
        from utils.vggt_wrapper import OpenCVPoseEstimator
        oracle = OpenCVPoseEstimator()
        print("  Pose oracle: OpenCV fallback")
        return oracle
    except ImportError:
        pass

    raise RuntimeError("No pose oracle available. Install VGGT or OpenCV.")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("STEP 6: Evaluate Phase 2 — Impact of Pose Corruption")
    print("=" * 60)

    phase2_dir = RUNS_DIR / "phase2"

    # Load pose oracle
    oracle = load_oracle()

    # Determine LPIPS device
    lpips_device = VGGT_DEVICE  # typically "cuda"

    all_results = []

    # Evaluate: clean baseline (scale=0) + each corruption scale
    conditions = [0.0] + CORRUPTION_SCALES

    for scale in conditions:
        print(f"\n--- Scale = {scale:.1f} {'(clean baseline)' if scale == 0 else ''} ---")

        for i in range(N_SAMPLES):
            sample_id = f"sample_{i:04d}"

            # --- Locate generated frames ---
            # All scales (including 0.0 = clean baseline) are now in phase2/
            gen_dir = phase2_dir / f"{sample_id}_scale{scale:.1f}" / "gen_frames"

            if not gen_dir.exists():
                print(f"  {sample_id} scale={scale:.1f}: no gen_frames — skipping")
                continue

            gen_frames = load_frames(gen_dir)
            if not gen_frames:
                print(f"  {sample_id} scale={scale:.1f}: empty gen_frames — skipping")
                continue

            # --- Locate GT future frames ---
            gt_dir = phase2_dir / f"gt_frames_{sample_id}"
            gt_frames = load_frames(gt_dir) if gt_dir.exists() else []

            # --- 1) LPIPS: generated vs GT ---
            lpips_mean = None
            if gt_frames and len(gt_frames) == len(gen_frames):
                scores = compute_lpips_scores(gen_frames, gt_frames, device=lpips_device)
                if scores is not None:
                    lpips_mean = float(np.mean(scores))

            # --- 2) Pose accuracy: VGGT(gen_frames) vs VGGT(gt_frames) ---
            # GT poses come from VGGT on the clean gt_frames (scale=0.0 from step5 NPZ).
            # This is fully self-consistent regardless of frame_skip used in step5.
            pose_err_final = None
            gt_poses_npy = phase2_dir / f"gt_poses_sample_{i:04d}.npy"

            if gt_frames and not gt_poses_npy.exists():
                try:
                    gt_poses = oracle.estimate_poses_from_frames(np.stack(gt_frames))
                    np.save(gt_poses_npy, gt_poses)
                except Exception as e:
                    print(f"  {sample_id}: GT pose estimation error: {e}")

            if gt_poses_npy.exists() and gen_frames:
                try:
                    frames_np = np.stack(gen_frames)
                    poses_est = oracle.estimate_poses_from_frames(frames_np)
                    poses_gt = np.load(gt_poses_npy)

                    T_min = min(len(poses_est), len(poses_gt))
                    cum_err = compute_cumulative_rotation_error(
                        poses_gt[:T_min], poses_est[:T_min]
                    )
                    pose_err_final = float(cum_err[-1]) if len(cum_err) > 0 else None

                    out_dir = phase2_dir / f"{sample_id}_scale{scale:.1f}"
                    np.save(out_dir / "poses_est_phase2.npy", poses_est)

                except Exception as e:
                    print(f"  {sample_id} scale={scale:.1f}: pose estimation error: {e}")

            # Log
            parts = [f"  {sample_id} scale={scale:.1f}:"]
            if lpips_mean is not None:
                parts.append(f"LPIPS={lpips_mean:.4f}")
            if pose_err_final is not None:
                parts.append(f"pose_err={pose_err_final:.2f} deg")
            print("  ".join(parts))

            all_results.append({
                "sample_id": sample_id,
                "corruption_scale": scale,
                "lpips": lpips_mean,
                "final_pose_error_deg": pose_err_final,
            })

    # ----------------------------------------------------------------
    # Save results
    # ----------------------------------------------------------------
    if not all_results:
        print("\n  No results collected. Check that step5 outputs exist.")
        return

    df = pd.DataFrame(all_results)
    csv_path = phase2_dir / "phase2_evaluation.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ----------------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("PHASE 2 EVALUATION SUMMARY")
    print("=" * 60)

    agg = {}
    if df["lpips"].notna().any():
        agg["lpips"] = ["mean", "std"]
    if df["final_pose_error_deg"].notna().any():
        agg["final_pose_error_deg"] = ["mean", "std"]

    if agg:
        summary = df.groupby("corruption_scale").agg(agg).round(4)
        print(summary.to_string())
    else:
        print("  (no numeric results to summarise)")

    # ----------------------------------------------------------------
    # Plots
    # ----------------------------------------------------------------
    make_plots(phase2_dir, df)

    # ----------------------------------------------------------------
    # Update manifest
    # ----------------------------------------------------------------
    manifest_path = phase2_dir / "phase2_manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        manifest = {}

    manifest["evaluation_complete"] = True
    manifest["evaluation_csv"] = str(csv_path)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n  Phase 2 evaluation complete!")


# ============================================================
# Plots
# ============================================================

def make_plots(phase2_dir: Path, df: pd.DataFrame):
    """Generate Phase 2 evaluation plots."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Could not generate plots (matplotlib not installed)")
        return

    has_lpips = df["lpips"].notna().any()
    has_pose = df["final_pose_error_deg"].notna().any()

    if not has_lpips and not has_pose:
        return

    n_plots = int(has_lpips) + int(has_pose)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    plot_idx = 0

    # --- Plot 1: LPIPS vs corruption scale ---
    if has_lpips:
        ax = axes[plot_idx]
        plot_idx += 1
        grouped = df.groupby("corruption_scale")["lpips"]
        means = grouped.mean()
        stds = grouped.std().fillna(0)

        ax.errorbar(
            means.index, means.values, yerr=stds.values,
            fmt="o-", capsize=5, markersize=8, linewidth=2, color="#2196F3",
        )
        # Scatter individual points
        for scale in means.index:
            pts = df[df["corruption_scale"] == scale]["lpips"].dropna()
            ax.scatter([scale] * len(pts), pts, alpha=0.3, color="#2196F3", s=30)

        ax.set_xlabel("Corruption Scale", fontsize=13)
        ax.set_ylabel("LPIPS (lower = better quality)", fontsize=13)
        ax.set_title("1) Perceptual Quality vs Pose Corruption", fontsize=14)
        ax.grid(True, alpha=0.3)

    # --- Plot 2: Pose error vs corruption scale ---
    if has_pose:
        ax = axes[plot_idx]
        grouped = df.groupby("corruption_scale")["final_pose_error_deg"]
        means = grouped.mean()
        stds = grouped.std().fillna(0)

        ax.errorbar(
            means.index, means.values, yerr=stds.values,
            fmt="s-", capsize=5, markersize=8, linewidth=2, color="#FF9800",
        )
        for scale in means.index:
            pts = df[df["corruption_scale"] == scale]["final_pose_error_deg"].dropna()
            ax.scatter([scale] * len(pts), pts, alpha=0.3, color="#FF9800", s=30)

        ax.set_xlabel("Corruption Scale", fontsize=13)
        ax.set_ylabel("Cumulative Pose Error (degrees)", fontsize=13)
        ax.set_title("2) Future Pose Accuracy vs Pose Corruption", fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = phase2_dir / "phase2_evaluation.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {plot_path}")


if __name__ == "__main__":
    main()
