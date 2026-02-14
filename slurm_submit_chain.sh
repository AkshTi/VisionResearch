#!/bin/bash

# Submit all steps with SLURM job dependencies
# This ensures each step waits for the previous one to complete successfully
#
# Usage: bash slurm_submit_chain.sh

echo "========================================="
echo "Submitting Action-Video Mismatch Pipeline"
echo "with SLURM job dependencies"
echo "========================================="
echo ""

# Submit Step 1
STEP1_ID=$(sbatch --parsable slurm_step1_generate.sh)
echo "Step 1 (Generate Videos):  Job ID $STEP1_ID"

# Submit Step 2 (depends on Step 1)
STEP2_ID=$(sbatch --parsable --dependency=afterok:$STEP1_ID slurm_step2_poses.sh)
echo "Step 2 (Estimate Poses):   Job ID $STEP2_ID (waits for $STEP1_ID)"

# Submit Step 3 (depends on Step 2)
STEP3_ID=$(sbatch --parsable --dependency=afterok:$STEP2_ID slurm_step3_mismatch.sh)
echo "Step 3 (Compute Mismatch): Job ID $STEP3_ID (waits for $STEP2_ID)"

# Submit Step 4 (depends on Step 3)
STEP4_ID=$(sbatch --parsable --dependency=afterok:$STEP3_ID slurm_step4_fabricate.sh)
echo "Step 4 (Fabricate):        Job ID $STEP4_ID (waits for $STEP3_ID)"

echo ""
echo "========================================="
echo "All jobs submitted!"
echo "========================================="
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f results/slurm_${STEP1_ID}_step1_generate.out"
echo ""
echo "Cancel all with:"
echo "  scancel $STEP1_ID $STEP2_ID $STEP3_ID $STEP4_ID"
echo ""
