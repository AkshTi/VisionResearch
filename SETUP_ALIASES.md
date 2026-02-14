# SLURM Log Viewing Aliases - Setup Guide

## 🚀 Quick Setup

### On MIT ORCD, run:

```bash
# Navigate to your project directory
cd ~/VisionResearch  # or wherever your project is

# Copy aliases to your home directory
cat slurm_aliases.sh >> ~/.bashrc

# Reload your shell
source ~/.bashrc
```

---

## 📖 Usage

### Basic Commands

```bash
# View output file for job 9243048
sout 9243048

# View error file for job 9243048
serr 9243048

# View both output and error files
slog 9243048

# Live monitoring of running job (Ctrl+C to exit)
stail 9243048

# Show last 50 lines of output
slast 9243048

# Show last 50 lines of errors
slasterr 9243048
```

### Advanced Commands

```bash
# List all your SLURM log files (most recent first)
sls

# Find logs matching a pattern
sfind step1        # Shows all step1 logs
sfind 9243         # Shows all logs with 9243 in the name

# Show your job queue + latest log files
sstatus

# Search for a pattern in both .out and .err
sgrep 9243048 "ERROR"
sgrep 9243048 "Step 1 Complete"

# Check for errors in all recent logs
schkerr
```

### Step-Specific Commands

```bash
# View latest Step 1 logs
s1logs

# View latest Step 2 logs
s2logs

# View latest Step 3 logs
s3logs

# View latest Step 4 logs
s4logs
```

---

## 💡 Common Workflows

### 1. Submit job and monitor it
```bash
sbatch slurm_step1_generate.sh
# Note the job ID from output: "Submitted batch job 9243048"

# Watch it run (live tail)
stail 9243048
```

### 2. Check completed job
```bash
# See full output
sout 9243048

# Check for errors
serr 9243048

# Or see both at once
slog 9243048
```

### 3. Debug a failed job
```bash
# Check error file first
serr 9243048

# Search for specific error patterns
sgrep 9243048 "ERROR"
sgrep 9243048 "Traceback"
sgrep 9243048 "FAIL"

# Check last 50 lines for quick diagnosis
slast 9243048
```

### 4. Monitor pipeline
```bash
# Submit chain
bash slurm_submit_chain.sh
# Output shows: 9243048, 9243049, 9243050, 9243051

# Check status of all jobs
sstatus

# Monitor step 1 as it runs
stail 9243048

# When step 1 finishes, check step 2
stail 9243049
```

---

## 📋 Alias Reference

| Alias | Description | Example |
|-------|-------------|---------|
| `sout` | View .out file | `sout 9243048` |
| `serr` | View .err file | `serr 9243048` |
| `slog` | View both files | `slog 9243048` |
| `stail` | Live tail .out | `stail 9243048` |
| `stailerr` | Live tail .err | `stailerr 9243048` |
| `slast` | Last 50 lines of .out | `slast 9243048` |
| `slasterr` | Last 50 lines of .err | `slasterr 9243048` |
| `sls` | List all log files | `sls` |
| `sfind` | Find logs by pattern | `sfind step1` |
| `sstatus` | Queue + recent logs | `sstatus` |
| `sgrep` | Search in logs | `sgrep 9243048 "ERROR"` |
| `s1logs` | Latest step1 logs | `s1logs` |
| `s2logs` | Latest step2 logs | `s2logs` |
| `s3logs` | Latest step3 logs | `s3logs` |
| `s4logs` | Latest step4 logs | `s4logs` |
| `schkerr` | Check for errors | `schkerr` |

---

## 🎯 Pro Tips

### Combine with other SLURM commands
```bash
# See running jobs + tail latest
squeue -u $USER && stail $(squeue -u $USER -h -o "%A" | head -1)

# Check status + view latest errors
sstatus && schkerr
```

### Use with watch for auto-refresh
```bash
# Auto-refresh job status every 2 seconds
watch -n 2 'squeue -u $USER'

# Auto-refresh last 20 lines every 5 seconds
watch -n 5 'tail -20 results/slurm_9243048_*.out'
```

### Grep for specific info
```bash
# Find when job started
sgrep 9243048 "Start time"

# Check GPU allocation
sgrep 9243048 "GPU"

# See Python version
sgrep 9243048 "Python version"

# Check exit codes
sgrep 9243048 "Exit code"
```

---

## 🔧 Customize

Edit `~/.bashrc` to modify aliases. For example, to change the number of lines shown:

```bash
# Change slast to show 100 lines instead of 50
alias slast='_slurm_last() { tail -100 results/slurm_"$1"_*.out 2>/dev/null || echo "No .out file found for job $1"; }; _slurm_last'
```

After editing, run:
```bash
source ~/.bashrc
```

---

## 📁 File Locations

All aliases assume your SLURM logs are in:
```
results/slurm_<JOBID>_<name>.out
results/slurm_<JOBID>_<name>.err
```

If your logs are elsewhere, modify the `results/` path in each alias.

---

## ✅ Quick Test

After setup, test with:
```bash
# Should show your recent log files
sls

# Should show your job queue
sstatus

# Try with a real job ID
sout 9243048
```

Enjoy easier log viewing! 🎉
