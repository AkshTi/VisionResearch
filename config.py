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
N_SAMPLES = 5  # Set to 50+ for real experiment

# Number of history (context) frames given to DFoT
K_HISTORY = 4  # DFoT RE10k default uses 4 history frames

# Number of future frames to generate
T_FUTURE = 8  # Start small; DFoT RE10k uses 4 predicted frames for LPIPS eval

# Frame skip (temporal stride in the RE10k clip)
FRAME_SKIP = 20  # Matches DFoT's default for short generation

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
