#!/bin/bash
#SBATCH --job-name=action_mismatch_pipeline
#SBATCH --output=results/slurm_%j_pipeline.out
#SBATCH --error=results/slurm_%j_pipeline.err
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/orcd/home/002/akshatat/VisionResearch

# Full pipeline: All 6 steps in sequence (N_SAMPLES=50)
# Submit with: sbatch slurm_run_all.sh
# Monitor with: tail -f results/slurm_<JOBID>_pipeline.out

echo "========================================="
echo "Action-Video Mismatch Experiment"
echo "FULL PIPELINE (Steps 1-6)"
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

# Verify GPU
echo "GPU availability:"
nvidia-smi
echo ""

# Set environment variables
export WANDB_MODE=online
export PYTHONUNBUFFERED=1

# Create results directory
mkdir -p results

# Track overall success
OVERALL_SUCCESS=0

# =========================================
# STEP 1: Generate Videos
# =========================================
echo ""
echo "========================================="
echo "STEP 1: Generating videos with DFoT..."
echo "========================================="
echo "Start: $(date)"
echo ""

python step1_generate_videos.py
STEP1_EXIT=$?

if [ $STEP1_EXIT -ne 0 ]; then
    echo "ERROR: Step 1 failed with exit code $STEP1_EXIT"
    echo "Aborting pipeline."
    exit $STEP1_EXIT
fi

echo "✓ Step 1 complete: $(date)"

# =========================================
# STEP 2: Estimate Poses
# =========================================
echo ""
echo "========================================="
echo "STEP 2: Estimating poses with VGGT..."
echo "========================================="
echo "Start: $(date)"
echo ""

python step2_estimate_poses.py
STEP2_EXIT=$?

if [ $STEP2_EXIT -ne 0 ]; then
    echo "ERROR: Step 2 failed with exit code $STEP2_EXIT"
    echo "Aborting pipeline."
    exit $STEP2_EXIT
fi

echo "✓ Step 2 complete: $(date)"

# =========================================
# STEP 3: Compute Mismatch (Phase 1)
# =========================================
echo ""
echo "========================================="
echo "STEP 3: Computing pose mismatch..."
echo "========================================="
echo "Start: $(date)"
echo ""

python step3_compute_mismatch.py
STEP3_EXIT=$?

if [ $STEP3_EXIT -ne 0 ]; then
    echo "ERROR: Step 3 failed with exit code $STEP3_EXIT"
    echo "Aborting pipeline."
    exit $STEP3_EXIT
fi

echo "✓ Step 3 complete (Phase 1 DONE): $(date)"

# =========================================
# STEP 4: Fabricate Corruption (Phase 2 Setup)
# =========================================
echo ""
echo "========================================="
echo "STEP 4: Fabricating corrupted histories..."
echo "========================================="
echo "Start: $(date)"
echo ""

python step4_fabricate_and_evaluate.py
STEP4_EXIT=$?

if [ $STEP4_EXIT -ne 0 ]; then
    echo "ERROR: Step 4 failed with exit code $STEP4_EXIT"
    exit $STEP4_EXIT
fi

echo "✓ Step 4 complete (Phase 2 setup DONE): $(date)"

# =========================================
# STEP 5: Regenerate with Corrupted Poses
# =========================================
echo ""
echo "========================================="
echo "STEP 5: Regenerating with corrupted poses..."
echo "========================================="
echo "Start: $(date)"
echo ""

python step5_regenerate_corrupted.py
STEP5_EXIT=$?

if [ $STEP5_EXIT -ne 0 ]; then
    echo "ERROR: Step 5 failed with exit code $STEP5_EXIT"
    exit $STEP5_EXIT
fi

echo "✓ Step 5 complete: $(date)"

# =========================================
# STEP 6: Evaluate Phase 2
# =========================================
echo ""
echo "========================================="
echo "STEP 6: Evaluating Phase 2 (LPIPS + pose error)..."
echo "========================================="
echo "Start: $(date)"
echo ""

python step6_evaluate_phase2.py
STEP6_EXIT=$?

if [ $STEP6_EXIT -ne 0 ]; then
    echo "ERROR: Step 6 failed with exit code $STEP6_EXIT"
    exit $STEP6_EXIT
fi

echo "✓ Step 6 complete (Phase 2 DONE): $(date)"

# =========================================
# PIPELINE COMPLETE
# =========================================
echo ""
echo "========================================="
echo "✓✓✓ FULL PIPELINE COMPLETE (Steps 1-6) ✓✓✓"
echo "========================================="
echo ""
echo "End time: $(date)"
echo ""
echo "Results summary:"
echo "----------------"
echo "Generated videos:  runs/action_mismatch/generated/"
echo "Phase 1 analysis:  runs/action_mismatch/aggregate/"
echo "Phase 2 frames:    runs/action_mismatch/phase2/"
echo "Phase 2 results:   runs/action_mismatch/phase2/phase2_evaluation.csv"
echo "Phase 2 plot:      runs/action_mismatch/phase2/phase2_evaluation.png"
echo ""
echo "Key outputs:"
ls -lh runs/action_mismatch/aggregate/*.csv runs/action_mismatch/phase2/phase2_evaluation.csv 2>/dev/null
echo ""

exit 0
