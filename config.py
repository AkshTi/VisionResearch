"""
Configuration for the action-video mismatch experiment.
Edit the paths below to match your local setup.
"""
from pathlib import Path

# ============================================================
# PATHS — Edit these to match your machine
# ============================================================

# Path to the cloned DFoT repo
DFOT_REPO = Path("./diffusion-forcing-transformer")

# Path to the cloned VGGT repo
VGGT_REPO = Path("./vggt")

# Path to RE10k data (DFoT will auto-download realestate10k_mini,
# but if you have the full dataset, point here)
RE10K_DATA = DFOT_REPO / "data"

# Output directory for this experiment
RUNS_DIR = Path("./runs/action_mismatch")

# ============================================================
# EXPERIMENT HYPERPARAMETERS
# ============================================================

# Number of samples to generate (start with 5 for debugging, scale to 50+)
N_SAMPLES = 50  # Set to 50+ for real experiment

# Number of history (context) frames given to DFoT
# MUST be 1: RE10k pretrained model was trained for single-image conditioning.
K_HISTORY = 1

# Number of future frames to generate
# MUST be 7: RE10k model has max_frames=8, context_frames=1, so it generates
# exactly 7 future frames. Asking for more causes sliding-window autoregression
# past the training horizon, which collapses to all-black outputs.
T_FUTURE = 7

# Frame skip (temporal stride in the RE10k clip)
# MUST be 20: This is the default used by the DFoT authors for single-image-to-short
FRAME_SKIP = 20

# DFoT pretrained checkpoint name
DFOT_CHECKPOINT = "DFoT_RE10K.ckpt"

# History guidance settings
HISTORY_GUIDANCE_NAME = "vanilla"
HISTORY_GUIDANCE_SCALE = 4.0

# ============================================================
# VGGT SETTINGS
# ============================================================

# VGGT model name (auto-downloads from HuggingFace)
VGGT_MODEL_NAME = "facebook/VGGT-1B"

# Device for VGGT inference
VGGT_DEVICE = "cuda"  # or "cpu" if no GPU

# ============================================================
# PHASE 2: CORRUPTION SETTINGS
# ============================================================

# Number of clips to evaluate in Phase 2
N_PHASE2_CLIPS = 10

# Maximum baseline cumulative drift (degrees) for a sample to be included in
# Phase 2.  Clips with higher drift under clean pose conditioning are excluded
# because VGGT cannot reliably track them, which inflates the median drift and
# miscalibrates the corruption scaling (see 50-sample run analysis).
MAX_BASELINE_DRIFT = 2.0

# Corruption levels (scale factors applied to measured drift)
CORRUPTION_SCALES = [10.0, 50.0, 100.0]#[0.5, 1.0, 2.0]

# ============================================================
# EVALUATION SETTINGS
# ============================================================

# Whether to compute LPIPS (requires lpips package)
COMPUTE_LPIPS = True

# Random seed for reproducibility
SEED = 42

# ============================================================
# WANDB SETTINGS (needed for DFoT)
# ============================================================

WANDB_ENTITY = "akshatatiwari55" 
WANDB_PROJECT = "action-mismatch-experiment"
