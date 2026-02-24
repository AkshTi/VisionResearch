#!/bin/bash
#SBATCH --job-name=step5_regen
#SBATCH --output=results/slurm_%j_step5_regenerate.out
#SBATCH --error=results/slurm_%j_step5_regenerate.err
#SBATCH --time=06:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/orcd/home/002/akshatat/VisionResearch

# Step 5: Regenerate videos with corrupted pose conditioning
# Runs DFoT 4x (clean baseline + 3 corruption scales)
# Submit with: sbatch slurm_step5_regenerate.sh
# Monitor with: tail -f results/slurm_<JOBID>_step5_regenerate.out

echo "========================================="
echo "Step 5: Regenerate with Corrupted Poses"
echo "Action-Video Mismatch Experiment - Phase 2"
echo "========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Load modules (commented out - conda env has everything)
# module load python/3.10
# module load cuda/12.1

# Activate DFoT environment
source ~/.bashrc
conda activate mech_interp_gpu

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
echo "Starting corrupted pose regeneration..."
echo "========================================="
echo ""

# Check Phase 2 setup exists
if [ ! -f "runs/action_mismatch/phase2/phase2_manifest.json" ]; then
    echo "ERROR: Phase 2 manifest not found!"
    echo "Run steps 1-4 first."
    exit 1
fi

# Run step 5
python step5_regenerate_corrupted.py

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Step 5 Complete"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

# Check outputs
echo "Phase 2 generated frames:"
find runs/action_mismatch/phase2 -name "frame_0000.png" | head -20

exit $EXIT_CODE
