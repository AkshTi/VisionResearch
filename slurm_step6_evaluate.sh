#!/bin/bash
#SBATCH --job-name=step6_eval
#SBATCH --output=results/slurm_%j_step6_evaluate.out
#SBATCH --error=results/slurm_%j_step6_evaluate.err
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --chdir=/orcd/home/002/akshatat/VisionResearch

# Step 6: Evaluate Phase 2 (LPIPS + future pose accuracy)
# Submit with: sbatch slurm_step6_evaluate.sh
# Monitor with: tail -f results/slurm_<JOBID>_step6_evaluate.out
# NOTE: Run AFTER step5 completes.

echo "========================================="
echo "Step 6: Phase 2 Evaluation"
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

# Activate environment
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
echo "Starting Phase 2 evaluation..."
echo "========================================="
echo ""

# Check step 5 outputs exist
if [ ! -d "runs/action_mismatch/phase2" ]; then
    echo "ERROR: Phase 2 outputs not found!"
    echo "Run step5 first."
    exit 1
fi

# Run step 6
python step6_evaluate_phase2.py

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Step 6 Complete - PHASE 2 DONE!"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

# Display results
echo "Evaluation results:"
if [ -f "runs/action_mismatch/phase2/phase2_evaluation.csv" ]; then
    cat runs/action_mismatch/phase2/phase2_evaluation.csv
fi

exit $EXIT_CODE
