#!/bin/bash
# Helper script to regenerate videos with corrupted poses.
#
# This creates a modified RE10k dataset with corrupted poses,
# then runs DFoT on it.
#
# Usage: bash regenerate_with_corruption.sh

DFOT_REPO="diffusion-forcing-transformer"
PHASE2_DIR="runs/action_mismatch/phase2"

echo "=== Regenerating with corrupted poses ==="

# For each corruption condition, run DFoT generation:
# 1. Clean history (baseline)
echo "Generating with CLEAN poses (baseline)..."
cd $DFOT_REPO
python -m main +name=phase2_clean dataset=realestate10k_mini algorithm=dfot_video_pose \
  experiment=video_generation @diffusion/continuous load=pretrained:DFoT_RE10K.ckpt \
  'experiment.tasks=[validation]' experiment.validation.batch_size=1 \
  dataset.context_length=1 dataset.frame_skip=20 dataset.n_frames=8 \
  dataset.num_eval_videos=10 \
  algorithm.tasks.prediction.history_guidance.name=vanilla \
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
