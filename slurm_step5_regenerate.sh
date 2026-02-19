#!/bin/bash
#SBATCH --job-name=step5_regen
#SBATCH --output=results/slurm_%j_step5_regenerate.out
#SBATCH --error=results/slurm_%j_step5_regenerate.err
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/orcd/home/002/akshatat/VisionResearch

# Step 5: Regenerate videos with corrupted pose conditioning
# Runs DFoT 3 times (one per corruption scale: 0.5, 1.0, 2.0)
# Submit with: sbatch slurm_step5_regenerate.sh

echo "========================================="
echo "Step 5: Regenerate with Corrupted Poses"
echo "Action-Video Mismatch Experiment"
echo "========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

source ~/.bashrc
conda activate mech_interp_gpu

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Install scipy if needed (used by monkey-patch for rotation perturbation)
pip install scipy -q

nvidia-smi
echo ""

export WANDB_MODE=online
export PYTHONUNBUFFERED=1
mkdir -p results

echo "========================================="
echo "Starting corrupted pose regeneration..."
echo "========================================="
echo ""

python step5_regenerate_corrupted.py

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Step 5 Complete"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

echo "Phase 2 outputs:"
ls -lh runs/action_mismatch/phase2/

exit $EXIT_CODE
