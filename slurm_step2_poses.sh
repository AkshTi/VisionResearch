#!/bin/bash
#SBATCH --job-name=step2_pose_est
#SBATCH --output=results/slurm_%j_step2_poses.out
#SBATCH --error=results/slurm_%j_step2_poses.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4

# Step 2: Estimate poses from generated videos using VGGT
# Submit with: sbatch slurm_step2_poses.sh
# Monitor with: tail -f results/slurm_<JOBID>_step2_poses.out

echo "========================================="
echo "Step 2: Estimate Poses (VGGT Oracle)"
echo "Action-Video Mismatch Experiment"
echo "========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load modules
module load python/3.10
module load cuda/12.1

# Activate environment (use VGGT env if separate, or same as DFoT)
source ~/.bashrc
conda activate mech_interp_gpu  # or conda activate vggt

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Verify GPU availability
echo "Checking GPU availability..."
nvidia-smi
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Starting pose estimation with VGGT..."
echo "========================================="
echo ""

# Check input exists
if [ ! -d "runs/action_mismatch/generated" ]; then
    echo "ERROR: Generated videos not found!"
    echo "Run step1 first."
    exit 1
fi

# Run step 2
python step2_estimate_poses.py

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Step 2 Complete"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

# Check outputs
echo "Estimated poses saved to:"
find runs/action_mismatch/generated -name "poses_est_from_gen.npy" | head -5

exit $EXIT_CODE
