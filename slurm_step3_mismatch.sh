#!/bin/bash
#SBATCH --job-name=step3_mismatch
#SBATCH --output=results/slurm_%j_step3_mismatch.out
#SBATCH --error=results/slurm_%j_step3_mismatch.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Step 3: Compute pose mismatch statistics (Phase 1 analysis)
# Submit with: sbatch slurm_step3_mismatch.sh
# Monitor with: tail -f results/slurm_<JOBID>_step3_mismatch.out

echo "========================================="
echo "Step 3: Compute Pose Mismatch"
echo "Action-Video Mismatch Experiment - Phase 1"
echo "========================================="
echo ""
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Load modules (commented out - conda env has everything)
# module load python/3.10

# Activate environment
source ~/.bashrc
conda activate mech_interp_gpu

echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""

# Set environment variables
export PYTHONUNBUFFERED=1

echo "========================================="
echo "Computing rotation errors and drift..."
echo "========================================="
echo ""

# Check inputs exist
if [ ! -d "runs/action_mismatch/generated" ]; then
    echo "ERROR: Generated videos not found!"
    echo "Run step1 and step2 first."
    exit 1
fi

# Run step 3
python step3_compute_mismatch.py

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Step 3 Complete - PHASE 1 DONE!"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

# Display results
echo "Results saved to:"
ls -lh runs/action_mismatch/aggregate/

echo ""
echo "Drift statistics:"
if [ -f "runs/action_mismatch/aggregate/mismatch_all.csv" ]; then
    head -20 runs/action_mismatch/aggregate/mismatch_all.csv
fi

exit $EXIT_CODE
