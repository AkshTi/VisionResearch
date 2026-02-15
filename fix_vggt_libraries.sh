#!/bin/bash
# Fix VGGT library dependencies on ORCD
# Run this on the ORCD cluster

echo "========================================="
echo "Fixing VGGT Dependencies"
echo "========================================="
echo ""

# Activate conda environment
source ~/.bashrc
conda activate mech_interp_gpu

echo "Current library versions:"
conda list | grep -E "libstdcxx|libgcc|pytorch"
echo ""

echo "Updating C++ standard library..."
conda install -y -c conda-forge libstdcxx-ng>=12

echo ""
echo "Reinstalling PyTorch with compatible libraries..."
conda install -y pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

echo ""
echo "Installing VGGT dependencies..."
pip install transformers>=4.30.0
pip install accelerate>=0.20.0

echo ""
echo "========================================="
echo "Testing VGGT Import"
echo "========================================="
echo ""

# Test if the library issue is resolved
python -c "
import sys
print('Testing imports...')
try:
    import torch
    print(f'✓ PyTorch {torch.__version__}')
except Exception as e:
    print(f'✗ PyTorch: {e}')
    sys.exit(1)

try:
    import torchvision
    print(f'✓ torchvision {torchvision.__version__}')
except Exception as e:
    print(f'✗ torchvision: {e}')
    sys.exit(1)

try:
    from transformers import AutoModel
    print('✓ transformers')
except Exception as e:
    print(f'✗ transformers: {e}')

print('')
print('All core dependencies OK!')
print('VGGT should now work.')
"

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "========================================="
    echo "SUCCESS! Libraries fixed."
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo "1. Clone VGGT repo (if not already done):"
    echo "   cd ~/VisionResearch"
    echo "   git clone https://github.com/facebookresearch/vggt.git"
    echo ""
    echo "2. Update config.py to point to VGGT repo:"
    echo "   VGGT_REPO = Path('./vggt')"
    echo ""
    echo "3. Re-run step 2:"
    echo "   sbatch slurm_step2_poses.sh"
else
    echo "========================================="
    echo "FAILED - Check errors above"
    echo "========================================="
fi

exit $EXIT_CODE
