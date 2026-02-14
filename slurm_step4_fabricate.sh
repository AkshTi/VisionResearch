#!/bin/bash
#SBATCH --job-name=step4_fabricate
#SBATCH --output=results/slurm_%j_step4_fabricate.out
#SBATCH --error=results/slurm_%j_step4_fabricate.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

# Step 4: Fabricate corrupted pose histories (Phase 2 setup)
# Submit with: sbatch slurm_step4_fabricate.sh
# Monitor with: tail -f results/slurm_<JOBID>_step4_fabricate.out

echo "========================================="
echo "Step 4: Fabricate Corrupted Histories"
echo "Action-Video Mismatch Experiment - Phase 2"
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
echo "Creating corrupted pose histories..."
echo "========================================="
echo ""

# Check Phase 1 results exist
if [ ! -f "runs/action_mismatch/aggregate/mismatch_all.csv" ]; then
    echo "ERROR: Phase 1 results not found!"
    echo "Run steps 1-3 first."
    exit 1
fi

# Run step 4
python step4_fabricate_and_evaluate.py

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Step 4 Complete - Phase 2 Setup Done"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

# Display results
echo "Corrupted poses saved to:"
ls -lh runs/action_mismatch/phase2/

echo ""
echo "Next: Regenerate videos with corrupted poses"
echo "See runs/action_mismatch/phase2/regenerate_with_corruption.sh"

exit $EXIT_CODE
