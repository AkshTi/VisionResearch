#!/bin/bash
#SBATCH --job-name=step6_eval
#SBATCH --output=results/slurm_%j_step6_evaluate.out
#SBATCH --error=results/slurm_%j_step6_evaluate.err
#SBATCH --time=04:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --chdir=/orcd/home/002/akshatat/VisionResearch

# Step 6: Evaluate Phase 2 (LPIPS + pose accuracy)
# Submit with: sbatch slurm_step6_evaluate.sh
# NOTE: Run AFTER step5 completes.

echo "========================================="
echo "Step 6: Phase 2 Evaluation"
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

# Install evaluation dependencies
pip install lpips scipy -q

nvidia-smi
echo ""

export PYTHONUNBUFFERED=1
mkdir -p results

echo "========================================="
echo "Starting Phase 2 evaluation..."
echo "========================================="
echo ""

python step6_evaluate_phase2.py

EXIT_CODE=$?

echo ""
echo "========================================="
echo "Step 6 Complete"
echo "========================================="
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo ""

echo "Evaluation results:"
cat runs/action_mismatch/phase2/phase2_evaluation.csv 2>/dev/null || echo "(no CSV yet)"
echo ""
echo "Plot saved to:"
ls -lh runs/action_mismatch/phase2/phase2_evaluation.png 2>/dev/null || echo "(no plot yet)"

exit $EXIT_CODE
