"""
Pose utilities: rotation error, SE(3) operations, scale-invariant comparisons.

Key metric: geodesic distance on SO(3) — this is scale-free and robust,
which addresses Hyunwoo's concern about SE(3) scale ambiguity.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def rotation_error_deg(R_est: np.ndarray, R_gt: np.ndarray) -> float:
    """
    Compute geodesic distance between two rotation matrices in degrees.
    
    rot_err = arccos((trace(R_est @ R_gt^T) - 1) / 2)
    
    This is scale-free — no translation ambiguity.
    
    Args:
        R_est: (3, 3) estimated rotation matrix
        R_gt:  (3, 3) ground-truth rotation matrix
    
    Returns:
        Error in degrees
    """
    R_diff = R_est @ R_gt.T
    # Clamp trace for numerical stability
    trace_val = np.clip((np.trace(R_diff) - 1.0) / 2.0, -1.0, 1.0)
    angle_rad = np.arccos(trace_val)
    return np.degrees(angle_rad)


def translation_direction_error_deg(t_est: np.ndarray, t_gt: np.ndarray) -> float:
    """
    Compute angular error between translation directions (scale-invariant).
    
    We normalize both translations to unit vectors and measure the angle between them.
    This avoids scale ambiguity in SE(3).
    
    Args:
        t_est: (3,) estimated translation vector
        t_gt:  (3,) ground-truth translation vector
    
    Returns:
        Error in degrees. Returns NaN if either vector is near-zero.
    """
    norm_est = np.linalg.norm(t_est)
    norm_gt = np.linalg.norm(t_gt)
    
    if norm_est < 1e-8 or norm_gt < 1e-8:
        return float('nan')
    
    cos_angle = np.clip(
        np.dot(t_est / norm_est, t_gt / norm_gt), -1.0, 1.0
    )
    return np.degrees(np.arccos(cos_angle))


def extrinsic_to_R_t(extrinsic: np.ndarray):
    """
    Extract rotation and translation from a 4x4 extrinsic matrix.
    
    Args:
        extrinsic: (4, 4) camera-from-world transformation
    
    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
    """
    R = extrinsic[:3, :3]
    t = extrinsic[:3, 3]
    return R, t


def relative_pose(ext_a: np.ndarray, ext_b: np.ndarray) -> np.ndarray:
    """
    Compute relative pose from frame A to frame B.
    
    T_rel = T_b @ T_a^{-1}
    
    Args:
        ext_a: (4, 4) extrinsic of frame A
        ext_b: (4, 4) extrinsic of frame B
    
    Returns:
        (4, 4) relative transformation
    """
    return ext_b @ np.linalg.inv(ext_a)


def compute_rotation_errors_over_time(
    poses_est: np.ndarray, 
    poses_gt: np.ndarray,
    use_relative: bool = True
) -> np.ndarray:
    """
    Compute rotation error at each timestep.
    
    Args:
        poses_est: (T, 4, 4) estimated extrinsic matrices
        poses_gt:  (T, 4, 4) ground-truth extrinsic matrices
        use_relative: If True, compare relative poses (frame-to-frame).
                      If False, compare absolute poses (but these have
                      a global alignment issue).
    
    Returns:
        errors: (T,) or (T-1,) array of rotation errors in degrees
    """
    T = len(poses_est)
    assert len(poses_gt) == T, f"Length mismatch: {len(poses_est)} vs {len(poses_gt)}"
    
    if use_relative:
        # Compare relative (frame-to-frame) rotations — more robust
        errors = []
        for i in range(1, T):
            rel_est = relative_pose(poses_est[i-1], poses_est[i])
            rel_gt = relative_pose(poses_gt[i-1], poses_gt[i])
            R_est, _ = extrinsic_to_R_t(rel_est)
            R_gt, _ = extrinsic_to_R_t(rel_gt)
            errors.append(rotation_error_deg(R_est, R_gt))
        return np.array(errors)
    else:
        # Compare absolute poses (need global alignment first)
        # Align by setting first frame as identity for both
        T0_est_inv = np.linalg.inv(poses_est[0])
        T0_gt_inv = np.linalg.inv(poses_gt[0])
        
        errors = []
        for i in range(T):
            aligned_est = poses_est[i] @ T0_est_inv
            aligned_gt = poses_gt[i] @ T0_gt_inv
            R_est, _ = extrinsic_to_R_t(aligned_est)
            R_gt, _ = extrinsic_to_R_t(aligned_gt)
            errors.append(rotation_error_deg(R_est, R_gt))
        return np.array(errors)


def compute_cumulative_rotation_error(
    poses_est: np.ndarray,
    poses_gt: np.ndarray
) -> np.ndarray:
    """
    Compute CUMULATIVE rotation drift: error of pose at time t relative to first frame.
    
    This measures how much total drift has accumulated, which is what
    Hyunwoo means by "accumulation of divergence."
    
    Args:
        poses_est: (T, 4, 4) estimated extrinsics
        poses_gt:  (T, 4, 4) ground-truth extrinsics
    
    Returns:
        errors: (T,) array — error[0] = 0 by construction
    """
    T = len(poses_est)
    
    # Align so that first frame is identity in both
    T0_est_inv = np.linalg.inv(poses_est[0])
    T0_gt_inv = np.linalg.inv(poses_gt[0])
    
    errors = np.zeros(T)
    for i in range(T):
        aligned_est = poses_est[i] @ T0_est_inv
        aligned_gt = poses_gt[i] @ T0_gt_inv
        R_est, _ = extrinsic_to_R_t(aligned_est)
        R_gt, _ = extrinsic_to_R_t(aligned_gt)
        errors[i] = rotation_error_deg(R_est, R_gt)
    
    return errors


def apply_rotation_perturbation(
    poses: np.ndarray,
    drift_trajectory: np.ndarray,
    scale: float = 1.0
) -> np.ndarray:
    """
    Apply a measured drift trajectory to corrupt a pose sequence.
    
    This is used in Phase 2 to fabricate mismatched history.
    
    Args:
        poses: (T, 4, 4) original pose sequence
        drift_trajectory: (T,) drift angles in degrees at each timestep
                         (from Phase 1 measurements)
        scale: multiply drift by this factor
    
    Returns:
        corrupted_poses: (T, 4, 4) poses with applied rotation perturbation
    """
    corrupted = poses.copy()
    
    for i in range(len(poses)):
        # Create a random rotation axis
        rng = np.random.RandomState(42 + i)
        axis = rng.randn(3)
        axis = axis / np.linalg.norm(axis)
        
        # Rotate by drift_trajectory[i] * scale degrees around this axis
        angle_deg = drift_trajectory[i] * scale
        R_perturb = Rotation.from_rotvec(
            np.radians(angle_deg) * axis
        ).as_matrix()
        
        # Apply perturbation to rotation part
        corrupted[i, :3, :3] = R_perturb @ poses[i, :3, :3]
    
    return corrupted


def re10k_txt_to_poses(filepath: str) -> np.ndarray:
    """
    Parse a RE10k pose file (.txt) into extrinsic matrices.
    
    RE10k format: each line is:
        timestamp fx fy cx cy [16 values of 4x4 extrinsic matrix row-major]
    
    Args:
        filepath: path to the RE10k .txt pose file
    
    Returns:
        poses: (N, 4, 4) array of extrinsic matrices
        timestamps: (N,) array of timestamps
    """
    poses = []
    timestamps = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 21:  # timestamp + 4 intrinsics + 16 extrinsic values
                continue
            
            timestamp = int(parts[0])
            # fx, fy, cx, cy = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            # 16 values of the extrinsic matrix (row-major 4x4)
            ext_values = [float(x) for x in parts[5:21]]
            ext_matrix = np.array(ext_values).reshape(4, 4)
            
            poses.append(ext_matrix)
            timestamps.append(timestamp)
    
    return np.array(poses), np.array(timestamps)
