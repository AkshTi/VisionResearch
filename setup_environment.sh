#!/bin/bash
# One-time environment setup for ORCD
# Run this once: bash setup_environment.sh

echo "========================================="
echo "VisionResearch Environment Setup"
echo "========================================="
echo ""

# Activate conda environment
echo "1. Activating conda environment..."
source ~/.bashrc
conda activate mech_interp_gpu
echo "   ✓ Using: $(which python)"
echo ""

# Install DFoT requirements
echo "2. Installing DFoT dependencies..."
if [ -d "diffusion-forcing-transformer" ]; then
    cd diffusion-forcing-transformer
    pip install -r requirements.txt
    echo "   ✓ DFoT dependencies installed"
    cd ..
else
    echo "   ✗ Error: diffusion-forcing-transformer/ not found"
    echo "   Run: git clone https://github.com/kwsong0113/diffusion-forcing-transformer.git"
fi
echo ""

# Install VGGT requirements
echo "3. Installing VGGT dependencies..."
if [ -d "vggt" ]; then
    cd vggt
    if [ -f "requirements.txt" ]; then
        pip install -r requirements.txt
        echo "   ✓ VGGT dependencies installed"
    else
        echo "   ⚠ No requirements.txt found in vggt/"
    fi
    cd ..
else
    echo "   ✗ Error: vggt/ not found"
    echo "   Run: git clone https://github.com/facebookresearch/vggt.git"
fi
echo ""

# Install common dependencies
echo "4. Installing common dependencies..."
pip install numpy pandas pillow matplotlib lpips
echo "   ✓ Common packages installed"
echo ""

# Setup W&B
echo "5. W&B Setup"
echo "   Run: wandb login"
echo "   (You'll need to enter your API key)"
echo ""

# Verify installations
echo "========================================="
echo "Verification"
echo "========================================="
echo "Python: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'N/A')"
echo "Hydra: $(python -c 'import hydra; print(hydra.__version__)' 2>/dev/null || echo 'Not installed')"
echo "W&B: $(python -c 'import wandb; print(wandb.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""

echo "========================================="
echo "✓ Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "  1. wandb login"
echo "  2. Update config.py with your W&B username"
echo "  3. sbatch slurm_step1_generate.sh"
echo ""
