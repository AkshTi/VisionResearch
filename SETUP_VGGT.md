# Setup VGGT Oracle on ORCD

**Goal:** Fix the GLIBCXX library issue and enable VGGT pose estimation

---

## Quick Start (Copy-Paste)

```bash
# 1. Upload fix script
rsync -avz /Users/akshatatiwari/Desktop/VisionResearch/fix_vggt_libraries.sh orcd:~/VisionResearch/

# 2. SSH to ORCD
ssh orcd
cd ~/VisionResearch

# 3. Run fix script
bash fix_vggt_libraries.sh

# 4. Clone VGGT repo (if not already done)
git clone https://github.com/facebookresearch/vggt.git

# 5. Re-run Step 2
sbatch slurm_step2_poses.sh

# 6. Check results
tail -f results/slurm_*_step2_poses.out
```

---

## What the Fix Does

### Problem:
```
GLIBCXX_3.4.31' not found
```

VGGT requires a newer C++ standard library than what's on ORCD by default.

### Solution:

1. **Update libstdcxx-ng** → Provides GLIBCXX_3.4.31
2. **Reinstall PyTorch** → Ensures compatibility with new libraries
3. **Test imports** → Verifies everything works

---

## Expected Output (Success)

After running `fix_vggt_libraries.sh`, you should see:

```
Testing imports...
✓ PyTorch 2.x.x
✓ torchvision 0.x.x
✓ transformers

All core dependencies OK!
VGGT should now work.

=========================================
SUCCESS! Libraries fixed.
=========================================
```

---

## Expected Output (Step 2 with VGGT)

When you re-run Step 2, you should see:

```
Found 5 samples
Loading VGGT model...
VGGT loaded successfully.
  Using VGGT oracle (recommended)
  Processing sample_0000 (8 frames)...
  Saved: runs/action_mismatch/generated/sample_0000/poses_est_from_gen.npy
...
```

**No more "Could not load VGGT" message!**

---

## Troubleshooting

### If fix script fails:

**Error:** `conda: command not found`
```bash
source ~/.bashrc
conda activate mech_interp_gpu
```

**Error:** `PackagesNotFoundError`
```bash
# Use different channel
conda install -c anaconda libstdcxx-ng
```

**Error:** Still getting GLIBCXX error
```bash
# Check library path
ldd ~/.conda/envs/mech_interp_gpu/lib/python3.10/site-packages/optree/_C*.so

# May need to update LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

### If VGGT still doesn't load:

**Check VGGT repo exists:**
```bash
ls ~/VisionResearch/vggt/
# Should see: vggt/ folder with Python files
```

**Clone if missing:**
```bash
cd ~/VisionResearch
git clone https://github.com/facebookresearch/vggt.git
```

**Test VGGT directly:**
```bash
cd ~/VisionResearch
python -c "
import sys
sys.path.insert(0, './vggt')
from vggt.models.vggt import VGGT
model = VGGT.from_pretrained('facebook/VGGT-1B')
print('VGGT loaded successfully!')
"
```

---

## Alternative: Use Container

If library issues persist, use Singularity container:

```bash
# Pull PyTorch container with newer libraries
singularity pull docker://pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Run Step 2 in container
singularity exec --nv pytorch_2.1.0-cuda12.1-cudnn8-runtime.sif \
  python step2_estimate_poses.py
```

---

## Why VGGT Matters

**OpenCV (current):**
- ❌ Scale ambiguity
- ❌ No learned priors
- ❌ Constant ~5° drift (unrealistic)

**VGGT (after fix):**
- ✅ Learned from data
- ✅ Matches XFactor's TPS oracle
- ✅ Realistic drift patterns
- ✅ Variable error (not constant)

**For real science:** You need VGGT to match Hyunwoo's request for "TPS oracle" pose estimation.

---

## Next Steps After VGGT Works

1. **Delete old results:**
   ```bash
   rm runs/action_mismatch/generated/sample_*/poses_est_from_gen.npy
   rm runs/action_mismatch/aggregate/*
   ```

2. **Re-run Steps 2-3:**
   ```bash
   sbatch slurm_step2_poses.sh
   # Wait for completion
   python step3_compute_mismatch.py
   ```

3. **Compare results:**
   - OpenCV: 35° drift, perfectly linear
   - VGGT: Realistic, variable drift based on actual generation quality

---

## Time Estimate

- Library fix: ~5 minutes
- VGGT download: ~2 minutes (first time)
- Step 2 re-run: ~5 minutes (5 samples × 8 frames)
- **Total: ~12 minutes**

Good luck! 🚀
