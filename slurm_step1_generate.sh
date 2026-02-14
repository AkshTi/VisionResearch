#!/bin/bash
#SBATCH --job-name=step1_gen_dfot
#SBATCH --output=results/slurm_%j_step1_generate.out
#SBATCH --error=results/slurm_%j_step1_generate.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8

# Step 1: Generate videos with pose-conditioned DFoT
# Submit with: sbatch slurm_step1_generate.sh
# Monitor with: tail -f results/slurm_<JOBID>_step1_generate.out

echo "========================================="
echo "Step 1: Generate Videos with DFoT"
echo "Action-Video Mismatch Experiment"
echo "========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load modules (adjust for MIT ORCD)
module load python/3.10
module load cuda/12.1

# Activate DFoT environment
# Replace with your actual conda env path
source ~/.bashrc
conda activate dfot

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Verify GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Set environment variables
export WANDB_MODE=online
export PYTHONUNBUFFERED=1

# Create results directory
mkdir -p results

echo "========================================="
echo "Starting DFoT video generation..."
echo "========================================="
echo ""

# Run step 1
python step1_generate_videos.py

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Step 1 Complete"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

# Check outputs
echo "Generated outputs:"
ls -lh runs/action_mismatch/generated/

exit $EXIT_CODE
