"""
Wrapper around DFoT for programmatic video generation.

DFoT (Diffusion Forcing Transformer) is normally invoked via command-line with Hydra.
This wrapper provides two approaches:
  1. Subprocess: call DFoT's main.py as a subprocess (simplest, always works)
  2. Direct: import DFoT modules and call them programmatically (faster, needs more setup)

For this experiment, we use approach 1 (subprocess) by default since it requires
no modifications to DFoT's codebase.
"""

import subprocess
import json
import shutil
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
from PIL import Image


class DFoTSubprocessWrapper:
    """
    Generate videos by calling DFoT as a subprocess.
    
    This is the safest approach — it uses DFoT exactly as documented
    and avoids import/dependency conflicts.
    """
    
    def __init__(
        self,
        dfot_repo_path: str,
        checkpoint: str = "DFoT_RE10K.ckpt",
        conda_env: str = "dfot",
    ):
        self.repo_path = Path(dfot_repo_path)
        self.checkpoint = checkpoint
        self.conda_env = conda_env
        
        # Verify repo exists
        if not (self.repo_path / "main.py").exists():
            raise FileNotFoundError(
                f"DFoT main.py not found at {self.repo_path}. "
                f"Did you clone the repo? "
                f"git clone https://github.com/kwsong0113/diffusion-forcing-transformer.git"
            )
    
    def generate_batch(
        self,
        name: str = "action_mismatch_eval",
        n_frames: int = 8,
        context_length: int = 1,
        frame_skip: int = 20,
        batch_size: int = 1,
        num_videos: Optional[int] = None,
        guidance_name: str = "vanilla",
        guidance_scale: float = 4.0,
        dataset: str = "realestate10k_mini",
        output_dir: Optional[str] = None,
        extra_args: Optional[list] = None,
    ) -> Path:
        """
        Generate a batch of videos using DFoT.
        
        This calls DFoT's main.py with the appropriate arguments.
        Generated videos and poses will be saved by DFoT's own logging.
        
        Args:
            name: Run name for wandb logging
            n_frames: Total number of frames (history + generated)
            context_length: Number of context (history) frames
            frame_skip: Temporal stride in the video
            batch_size: Batch size for generation
            num_videos: Number of videos to generate (None = all in dataset)
            guidance_name: History guidance method
            guidance_scale: Guidance strength
            dataset: Dataset config name
            output_dir: Where to save outputs (None = DFoT default)
            extra_args: Additional command-line args
        
        Returns:
            Path to the output directory
        """
        cmd = [
            "python", "-m", "main",
            f"+name={name}",
            f"dataset={dataset}",
            "algorithm=dfot_video_pose",
            "experiment=video_generation",
            "@diffusion/continuous",
            f"load=pretrained:{self.checkpoint}",
            "experiment.tasks=[validation]",
            f"experiment.validation.batch_size={batch_size}",
            f"dataset.context_length={context_length}",
            f"dataset.frame_skip={frame_skip}",
            f"dataset.n_frames={n_frames}",
            f"algorithm.tasks.prediction.history_guidance.name={guidance_name}",
            f"+algorithm.tasks.prediction.history_guidance.guidance_scale={guidance_scale}",
        ]
        
        if num_videos is not None:
            cmd.append(f"dataset.num_eval_videos={num_videos}")
        
        if extra_args:
            cmd.extend(extra_args)
        
        print(f"Running DFoT generation...")
        print(f"  Command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            cwd=str(self.repo_path),
            capture_output=True,
            text=True,
        )
        
        if result.returncode != 0:
            print(f"DFoT stderr:\n{result.stderr[-2000:]}")
            raise RuntimeError(f"DFoT generation failed with code {result.returncode}")
        
        # DFoT saves outputs via wandb; parse the output to find the run directory
        # The output directory is typically in outputs/ or logged to wandb
        print("DFoT generation complete.")
        print(f"DFoT stdout (last 500 chars):\n{result.stdout[-500:]}")
        
        return self.repo_path / "outputs"


class DFoTDirectWrapper:
    """
    Direct programmatic interface to DFoT.
    
    This imports DFoT's modules directly for more control over the generation
    process, which is needed for Phase 2 (feeding custom pose histories).
    
    IMPORTANT: This requires running from within the DFoT repo directory
    with the dfot conda environment activated.
    """
    
    def __init__(self, dfot_repo_path: str, device: str = "cuda"):
        import sys
        self.repo_path = Path(dfot_repo_path)
        if str(self.repo_path) not in sys.path:
            sys.path.insert(0, str(self.repo_path))
        
        self.device = device
        self._model = None
        self._config = None
    
    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Load the DFoT model.
        
        If checkpoint_path is None, uses the pretrained:DFoT_RE10K.ckpt mechanism.
        """
        # This is a template — the exact import path depends on DFoT's internal structure.
        # You may need to adjust based on your DFoT version.
        try:
            from algorithms.dfot.dfot_video_pose import DFoTVideoPose
            from hydra import compose, initialize_config_dir
            
            # Load config
            config_dir = str(self.repo_path / "configurations")
            with initialize_config_dir(config_dir=config_dir, version_base=None):
                cfg = compose(
                    config_name="config",
                    overrides=[
                        "algorithm=dfot_video_pose",
                        "dataset=realestate10k_mini",
                        "experiment=video_generation",
                        "@diffusion/continuous",
                    ]
                )
            
            self._config = cfg
            # Model loading would happen here — this is highly dependent on
            # DFoT's internal API which may change.
            print("Direct DFoT loading — adjust import paths as needed for your version.")
            
        except ImportError as e:
            print(f"Could not import DFoT modules: {e}")
            print("Make sure you're running from within the DFoT repo with the dfot env active.")
            raise
    
    def generate_with_custom_poses(
        self,
        history_frames: np.ndarray,
        history_poses: np.ndarray,
        future_poses: np.ndarray,
    ) -> np.ndarray:
        """
        Generate future frames given custom history frames and poses.
        
        THIS IS THE KEY FUNCTION FOR PHASE 2.
        
        It allows feeding "corrupted" history poses while keeping
        history frames clean — exactly what Hyunwoo asked for.
        
        Args:
            history_frames: (K, H, W, 3) uint8 history frames
            history_poses:  (K, 4, 4) history poses (can be corrupted!)
            future_poses:   (T, 4, 4) target future poses
        
        Returns:
            generated_frames: (T, H, W, 3) uint8 generated frames
        """
        # This is a TEMPLATE. The exact implementation depends on DFoT's API.
        # The key idea is:
        #   1. Encode history_frames with the VAE
        #   2. Set up the conditioning with history_poses + future_poses
        #   3. Run the diffusion sampling
        #   4. Decode the generated latents
        #
        # You'll need to look at DFoT's validation loop to see how it does this
        # internally, then replicate with your custom poses.
        
        raise NotImplementedError(
            "Direct generation with custom poses requires inspecting DFoT's internals.\n"
            "Look at algorithms/dfot/dfot_video_pose.py -> validation_step() or sample()\n"
            "to see how poses are fed to the model during generation.\n\n"
            "For Phase 2, the alternative approach is to:\n"
            "1. Modify the RE10k dataset files to contain corrupted poses\n"
            "2. Run DFoT's standard generation pipeline on the modified data\n"
            "See step4_fabricate_and_evaluate.py for this approach."
        )


# ============================================================
# Helper: Extract frames and poses from DFoT's output
# ============================================================

def extract_dfot_outputs(
    wandb_run_dir: str,
    output_dir: str,
    sample_ids: Optional[list] = None
) -> dict:
    """
    Extract generated frames and GT poses from DFoT's wandb output.
    
    DFoT saves validation outputs to wandb. This function finds them
    and organizes them into our experiment structure.
    
    Args:
        wandb_run_dir: Path to the wandb run directory
        output_dir: Where to save extracted outputs
        sample_ids: Which samples to extract (None = all)
    
    Returns:
        Dictionary mapping sample_id -> {frames_dir, poses_gt_path}
    """
    wandb_dir = Path(wandb_run_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # DFoT typically saves videos as .mp4 and logs poses via wandb
    # The exact output format depends on the experiment config
    # Look for video files and pose data in the wandb media directory
    
    video_files = list(wandb_dir.rglob("*.mp4"))
    print(f"Found {len(video_files)} video files in {wandb_dir}")
    
    for idx, vf in enumerate(video_files):
        if sample_ids and idx not in sample_ids:
            continue
        
        sample_dir = out_dir / f"sample_{idx:04d}"
        sample_dir.mkdir(exist_ok=True)
        
        # Copy video
        shutil.copy2(vf, sample_dir / "gen.mp4")
        
        # Extract frames from video
        frames_dir = sample_dir / "gen_frames"
        frames_dir.mkdir(exist_ok=True)
        extract_frames_from_video(vf, frames_dir)
        
        results[idx] = {
            "frames_dir": str(frames_dir),
            "video_path": str(sample_dir / "gen.mp4"),
        }
    
    return results


def extract_frames_from_video(video_path, output_dir, extension="png"):
    """Extract individual frames from an mp4 file."""
    try:
        import imageio.v3 as iio
        frames = iio.imread(str(video_path), plugin="pyav")
        for i, frame in enumerate(frames):
            img = Image.fromarray(frame)
            img.save(Path(output_dir) / f"frame_{i:04d}.{extension}")
    except ImportError:
        # Fallback to ffmpeg
        subprocess.run([
            "ffmpeg", "-i", str(video_path),
            "-q:v", "2",
            str(Path(output_dir) / f"frame_%04d.{extension}")
        ], capture_output=True)
