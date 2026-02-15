"""
Wrapper around VGGT (Visual Geometry Grounded Transformer) for camera pose estimation.

VGGT is the oracle used by XFactor's TPS metric. It takes a set of frames and
outputs SE(3) camera extrinsics directly — no COLMAP, no optimization.

This is what Hyunwoo means by "TPS oracle" — VGGT is the backbone.

Reference:
  - VGGT paper: https://arxiv.org/abs/2503.11651 (CVPR 2025 Best Paper)
  - XFactor uses VGGT: "our implementation uses VGGT as the oracle"
  - GitHub: https://github.com/facebookresearch/vggt
"""

import sys
import numpy as np
from pathlib import Path
from typing import Optional

import torch
from PIL import Image


class VGGTOracle:
    """
    Camera pose oracle using VGGT.
    
    Given a set of video frames, estimates SE(3) camera extrinsics
    (camera-from-world transformations) for each frame.
    """
    
    def __init__(
        self,
        vggt_repo_path: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float16
    ):
        """
        Args:
            vggt_repo_path: Path to cloned VGGT repo
            device: "cuda" or "cpu"
            dtype: torch.float16 for speed, torch.float32 for accuracy
        """
        self.device = device
        self.dtype = dtype
        
        # Add VGGT repo to path
        vggt_path = Path(vggt_repo_path)
        if str(vggt_path) not in sys.path:
            sys.path.insert(0, str(vggt_path))
        
        # Import VGGT
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        
        self._load_and_preprocess = load_and_preprocess_images
        self._pose_enc_to_extri_intri = pose_encoding_to_extri_intri
        
        # Load model
        print("Loading VGGT model...")
        self.model = VGGT.from_pretrained("facebook/VGGT-1B")
        self.model = self.model.to(device)
        self.model.eval()
        print("VGGT loaded successfully.")
    
    def estimate_poses_from_frames(
        self, 
        frames: list,
        return_numpy: bool = True
    ) -> np.ndarray:
        """
        Estimate camera poses from a list of video frames.
        
        Args:
            frames: List of PIL Images, or list of numpy arrays (H, W, 3) uint8,
                    or numpy array (T, H, W, 3)
            return_numpy: If True, return numpy array
        
        Returns:
            extrinsics: (T, 4, 4) camera-from-world extrinsic matrices
        """
        # Convert frames to the format VGGT expects
        if isinstance(frames, np.ndarray) and frames.ndim == 4:
            # (T, H, W, 3) numpy array -> list of PIL
            pil_frames = [Image.fromarray(f) for f in frames]
        elif len(frames) > 0 and isinstance(frames[0], np.ndarray):
            pil_frames = [Image.fromarray(f) for f in frames]
        else:
            pil_frames = frames  # Already PIL
        
        # VGGT preprocessing: load images into tensor
        # VGGT expects images as a tensor of shape (S, 3, H, W) normalized to [0, 1]
        images = self._preprocess_frames(pil_frames)
        images = images.to(self.device)
        
        with torch.no_grad():
            with torch.cuda.amp.autocast(dtype=self.dtype):
                images_batch = images[None]  # Add batch dimension: (1, S, 3, H, W)
                aggregated_tokens_list, ps_idx = self.model.aggregator(images_batch)
                
                # Get camera poses
                pose_enc = self.model.camera_head(aggregated_tokens_list)[-1]
                
                # Convert pose encoding to extrinsic and intrinsic matrices
                extrinsic, intrinsic = self._pose_enc_to_extri_intri(
                    pose_enc, images_batch.shape[-2:]
                )

        # VGGT returns (B, T, 3, 4) - need to add bottom row [0,0,0,1]
        extrinsic_np = extrinsic.squeeze(0).cpu().numpy()  # (T, 3, 4)

        if return_numpy:
            # Convert (T, 3, 4) to (T, 4, 4) by adding bottom row
            T = extrinsic_np.shape[0]
            extrinsic_full = np.zeros((T, 4, 4))
            extrinsic_full[:, :3, :] = extrinsic_np
            extrinsic_full[:, 3, 3] = 1.0  # Homogeneous coordinate
            return extrinsic_full

        # Add bottom row for torch output too
        T = extrinsic.shape[1]
        bottom_row = torch.zeros((1, T, 1, 4), device=extrinsic.device)
        bottom_row[:, :, :, 3] = 1.0
        extrinsic_full = torch.cat([extrinsic, bottom_row], dim=2)
        return extrinsic_full.squeeze(0)
    
    def estimate_poses_from_paths(
        self, 
        frame_paths: list
    ) -> np.ndarray:
        """
        Estimate camera poses from a list of frame file paths.
        
        Args:
            frame_paths: List of paths to image files
        
        Returns:
            extrinsics: (T, 4, 4) camera-from-world extrinsic matrices
        """
        frames = [Image.open(p).convert("RGB") for p in sorted(frame_paths)]
        return self.estimate_poses_from_frames(frames)
    
    def estimate_poses_from_directory(
        self,
        frame_dir: str,
        extension: str = "png"
    ) -> np.ndarray:
        """
        Estimate poses from all frames in a directory.
        
        Args:
            frame_dir: Path to directory containing frame images
            extension: File extension to look for
        
        Returns:
            extrinsics: (T, 4, 4) camera-from-world extrinsic matrices
        """
        frame_dir = Path(frame_dir)
        paths = sorted(frame_dir.glob(f"*.{extension}"))
        if not paths:
            raise FileNotFoundError(f"No .{extension} files in {frame_dir}")
        return self.estimate_poses_from_paths(paths)
    
    def _preprocess_frames(self, pil_frames: list) -> torch.Tensor:
        """
        Preprocess PIL frames for VGGT input.
        
        VGGT expects: (S, 3, H, W) tensor, float32, values in [0, 1]
        Images should be resized to a standard size.
        """
        import torchvision.transforms.functional as TF
        
        tensors = []
        for img in pil_frames:
            # Resize to VGGT's expected size (518 x 518 seems standard for DINOv2 backbone)
            # But VGGT handles variable sizes; we resize to keep memory manageable
            img_resized = img.resize((518, 518), Image.BILINEAR)
            t = TF.to_tensor(img_resized)  # (3, H, W), [0, 1]
            tensors.append(t)
        
        return torch.stack(tensors)  # (S, 3, H, W)


# ============================================================
# Fallback: Simple relative pose estimation using OpenCV
# Use this if VGGT is not available yet
# ============================================================

class OpenCVPoseEstimator:
    """
    Fallback pose estimator using OpenCV feature matching + essential matrix.
    
    Less accurate than VGGT but zero additional dependencies beyond opencv-python.
    Use this to get Phase 1 running today, then swap in VGGT.
    """
    
    def __init__(self):
        try:
            import cv2
            self.cv2 = cv2
        except ImportError:
            raise ImportError("pip install opencv-python")
        
        # SIFT feature detector
        self.detector = cv2.SIFT_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Assume a rough focal length (pixels) — will be wrong in absolute scale
        # but relative rotations should be approximately correct
        self.focal_default = 500.0
    
    def estimate_poses_from_frames(
        self,
        frames: np.ndarray,
        return_numpy: bool = True
    ) -> np.ndarray:
        """
        Estimate relative poses between consecutive frames.
        Returns absolute poses in the coordinate frame of the first frame.
        
        Args:
            frames: (T, H, W, 3) uint8 numpy array
        
        Returns:
            poses: (T, 4, 4) extrinsic matrices (first frame = identity)
        """
        T = len(frames)
        H, W = frames[0].shape[:2]
        cx, cy = W / 2, H / 2
        focal = self.focal_default
        K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
        
        poses = np.zeros((T, 4, 4))
        poses[0] = np.eye(4)  # First frame is identity
        
        for i in range(1, T):
            R, t = self._estimate_relative(frames[i-1], frames[i], K)
            
            # Accumulate: T_i = T_rel @ T_{i-1}
            T_rel = np.eye(4)
            T_rel[:3, :3] = R
            T_rel[:3, 3] = t.flatten()
            poses[i] = T_rel @ poses[i-1]
        
        return poses
    
    def _estimate_relative(self, img1, img2, K):
        """Estimate relative R, t between two frames."""
        cv2 = self.cv2
        
        gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        
        kp1, des1 = self.detector.detectAndCompute(gray1, None)
        kp2, des2 = self.detector.detectAndCompute(gray2, None)
        
        if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
            return np.eye(3), np.zeros((3, 1))
        
        # Match features
        matches = self.matcher.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good = []
        for m_pair in matches:
            if len(m_pair) == 2:
                m, n = m_pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)
        
        if len(good) < 8:
            return np.eye(3), np.zeros((3, 1))
        
        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])
        
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
        
        if E is None:
            return np.eye(3), np.zeros((3, 1))
        
        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K)
        
        return R, t
