#!/usr/bin/env python3
"""
Diagnostic: figure out WHERE the identical-output problem comes from.

Run on the cluster after step 5 completes:
    python diagnose.py
"""
import hashlib, sys, os
from pathlib import Path
from PIL import Image
import numpy as np

RUNS = Path("runs/action_mismatch")
PHASE2 = RUNS / "phase2"
GEN = RUNS / "generated"
SCALES = [0.0, 10.0, 50.0, 100.0]
N_SAMPLES = 5

def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()

def frame_stats(path):
    img = np.array(Image.open(path).convert("RGB"))
    return img.shape, float(img.mean()), float(img.std())

print("=" * 70)
print("DIAGNOSTIC: Are gen_frames truly identical across scales?")
print("=" * 70)

# ---- 1. Check gen_frames file identity ----
print("\n--- 1. MD5 checksums of gen_frames/frame_0004.png per scale ---")
for i in range(N_SAMPLES):
    hashes = {}
    for s in SCALES:
        if s == 0.0:
            d = GEN / f"sample_{i:04d}" / "gen_frames"
        else:
            d = PHASE2 / f"sample_{i:04d}_scale{s:.1f}" / "gen_frames"
        f = d / "frame_0004.png"
        if f.exists():
            h = md5(f)
            sz = f.stat().st_size
            hashes[s] = (h, sz)
        else:
            hashes[s] = ("MISSING", 0)
    unique = set(v[0] for v in hashes.values() if v[0] != "MISSING")
    status = "IDENTICAL" if len(unique) == 1 else f"DIFFER ({len(unique)} unique)"
    print(f"  sample_{i:04d}: {status}")
    for s, (h, sz) in hashes.items():
        print(f"    scale={s:>5.1f}: {h}  ({sz} bytes)")

# ---- 2. Check DFoT output GIFs directly ----
print("\n--- 2. MD5 checksums of raw DFoT GIFs (from outputs/) ---")
dfot = Path("diffusion-forcing-transformer")
outputs_dir = dfot / "outputs"
if outputs_dir.exists():
    run_dirs = sorted(outputs_dir.rglob("*.gif"))
    for g in run_dirs[-15:]:
        print(f"  {g.relative_to(dfot)}: {md5(g)} ({g.stat().st_size} bytes)")
else:
    print("  outputs/ not found — skipping")

# ---- 3. Check if gen_frames match the GIF extraction ----
print("\n--- 3. Frame pixel statistics (mean, std) for sample_0000 ---")
for s in SCALES:
    if s == 0.0:
        d = GEN / "sample_0000" / "gen_frames"
    else:
        d = PHASE2 / f"sample_0000_scale{s:.1f}" / "gen_frames"
    f = d / "frame_0004.png"
    if f.exists():
        shape, mean, std = frame_stats(f)
        print(f"  scale={s:>5.1f}: shape={shape}, mean={mean:.2f}, std={std:.2f}, size={f.stat().st_size}")
    else:
        print(f"  scale={s:>5.1f}: MISSING")

# ---- 4. Check gt_frames vs gen_frames ----
print("\n--- 4. Are gen_frames = gt_frames? (checking if extraction picked wrong half) ---")
for i in range(min(2, N_SAMPLES)):
    gt_f = PHASE2 / f"gt_frames_sample_{i:04d}" / "frame_0004.png"
    gen_f0 = GEN / f"sample_{i:04d}" / "gen_frames" / "frame_0004.png"
    gen_f10 = PHASE2 / f"sample_{i:04d}_scale10.0" / "gen_frames" / "frame_0004.png"
    results = {}
    for label, p in [("gt", gt_f), ("gen_s0", gen_f0), ("gen_s10", gen_f10)]:
        if p.exists():
            results[label] = md5(p)
        else:
            results[label] = "MISSING"
    print(f"  sample_{i:04d}:")
    for label, h in results.items():
        print(f"    {label:>8s}: {h}")
    if results.get("gt") == results.get("gen_s0"):
        print(f"    ** gt == gen_s0 => step1 extracted GT side, not prediction side!")
    if results.get("gen_s0") == results.get("gen_s10"):
        print(f"    ** gen_s0 == gen_s10 => scale 0 and scale 10 are byte-identical!")

# ---- 5. Check the step1 GIF extraction vs step5 ----
print("\n--- 5. GIF left-vs-right half check for sample_0000 ---")
for label, gif_dir in [("step1_clean", GEN / "sample_0000" / "videos")]:
    gifs = sorted(gif_dir.glob("*.gif")) if gif_dir.exists() else []
    if not gifs:
        print(f"  {label}: no GIF found")
        continue
    gif = Image.open(gifs[0])
    w = gif.size[0]
    half = w // 2
    gif.seek(4)
    frame = gif.convert("RGB")
    left = np.array(frame.crop((0, 0, half, frame.size[1])))
    right = np.array(frame.crop((half, 0, w, frame.size[1])))
    print(f"  {label} GIF frame 4:")
    print(f"    left  half (prediction): mean={left.mean():.2f}, std={left.std():.2f}")
    print(f"    right half (gt):         mean={right.mean():.2f}, std={right.std():.2f}")
    print(f"    identical? {np.array_equal(left, right)}")

    gen_f = GEN / "sample_0000" / "gen_frames" / "frame_0004.png"
    if gen_f.exists():
        saved = np.array(Image.open(gen_f).convert("RGB"))
        print(f"    saved gen_frame matches left?  {np.array_equal(saved, left)}")
        print(f"    saved gen_frame matches right? {np.array_equal(saved, right)}")

print("\n" + "=" * 70)
print("DONE. Look for IDENTICAL/DIFFER in section 1, and matching in section 4.")
print("=" * 70)
