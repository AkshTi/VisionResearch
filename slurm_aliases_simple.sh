#!/bin/bash
# Simple SLURM Log Aliases - No Job ID Required!
# Just type: sout, serr, stail
#
# Add to ~/.bashrc on MIT ORCD:
#   cat slurm_aliases_simple.sh >> ~/.bashrc
#   source ~/.bashrc

# ============================================================
# AUTO-DETECT MOST RECENT JOB
# ============================================================

# View most recent .out file
alias sout='cat $(ls -t results/slurm_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No .out files found"'

# View most recent .err file
alias serr='cat $(ls -t results/slurm_*.err 2>/dev/null | head -1) 2>/dev/null || echo "No .err files found"'

# View both most recent .out and .err
alias slog='echo "========================================"; echo "LATEST OUTPUT:"; echo "========================================"; cat $(ls -t results/slurm_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No .out files"; echo ""; echo "========================================"; echo "LATEST ERRORS:"; echo "========================================"; cat $(ls -t results/slurm_*.err 2>/dev/null | head -1) 2>/dev/null || echo "No .err files"'

# Live tail most recent .out file
alias stail='tail -f $(ls -t results/slurm_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No .out files found"'

# Live tail most recent .err file
alias stailerr='tail -f $(ls -t results/slurm_*.err 2>/dev/null | head -1) 2>/dev/null || echo "No .err files found"'

# Last 50 lines of most recent .out
alias slast='tail -50 $(ls -t results/slurm_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No .out files found"'

# Last 50 lines of most recent .err
alias slasterr='tail -50 $(ls -t results/slurm_*.err 2>/dev/null | head -1) 2>/dev/null || echo "No .err files found"'

# ============================================================
# VIEW SPECIFIC RECENT JOBS (1st, 2nd, 3rd most recent)
# ============================================================

# 2nd most recent output
alias sout2='cat $(ls -t results/slurm_*.out 2>/dev/null | sed -n 2p) 2>/dev/null || echo "No 2nd .out file found"'

# 3rd most recent output
alias sout3='cat $(ls -t results/slurm_*.out 2>/dev/null | sed -n 3p) 2>/dev/null || echo "No 3rd .out file found"'

# 2nd most recent error
alias serr2='cat $(ls -t results/slurm_*.err 2>/dev/null | sed -n 2p) 2>/dev/null || echo "No 2nd .err file found"'

# 3rd most recent error
alias serr3='cat $(ls -t results/slurm_*.err 2>/dev/null | sed -n 3p) 2>/dev/null || echo "No 3rd .err file found"'

# ============================================================
# LIST AND STATUS
# ============================================================

# List all SLURM log files (most recent first)
alias sls='ls -lth results/slurm_*.{out,err} 2>/dev/null | head -20'

# Show which file is most recent
alias swhich='echo "Most recent .out: $(ls -t results/slurm_*.out 2>/dev/null | head -1)"; echo "Most recent .err: $(ls -t results/slurm_*.err 2>/dev/null | head -1)"'

# Quick status: job queue + latest log file info
alias sstatus='echo "========================================"; echo "YOUR JOBS"; echo "========================================"; squeue -u $USER; echo ""; echo "========================================"; echo "LATEST LOG FILES"; echo "========================================"; ls -lth results/slurm_*.{out,err} 2>/dev/null | head -6'

# ============================================================
# STEP-SPECIFIC (most recent for each step)
# ============================================================

# Latest step1 output
alias s1='cat $(ls -t results/slurm_*_step1_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No step1 .out found"'

# Latest step2 output
alias s2='cat $(ls -t results/slurm_*_step2_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No step2 .out found"'

# Latest step3 output
alias s3='cat $(ls -t results/slurm_*_step3_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No step3 .out found"'

# Latest step4 output
alias s4='cat $(ls -t results/slurm_*_step4_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No step4 .out found"'

# Latest step1 error
alias s1err='cat $(ls -t results/slurm_*_step1_*.err 2>/dev/null | head -1) 2>/dev/null || echo "No step1 .err found"'

# Latest step2 error
alias s2err='cat $(ls -t results/slurm_*_step2_*.err 2>/dev/null | head -1) 2>/dev/null || echo "No step2 .err found"'

# Latest step3 error
alias s3err='cat $(ls -t results/slurm_*_step3_*.err 2>/dev/null | head -1) 2>/dev/null || echo "No step3 .err found"'

# Latest step4 error
alias s4err='cat $(ls -t results/slurm_*_step4_*.err 2>/dev/null | head -1) 2>/dev/null || echo "No step4 .err found"'

# Live tail step1
alias s1tail='tail -f $(ls -t results/slurm_*_step1_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No step1 .out found"'

# Live tail step2
alias s2tail='tail -f $(ls -t results/slurm_*_step2_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No step2 .out found"'

# ============================================================
# SEARCH AND CHECK
# ============================================================

# Check most recent logs for errors
alias schkerr='echo "Checking most recent .err file..."; cat $(ls -t results/slurm_*.err 2>/dev/null | head -1) 2>/dev/null | grep -i "error\|fail\|exception\|traceback" || echo "No errors found (or no .err file)"'

# Grep in most recent output
alias sgrep='_sgrep() { grep --color=auto "$1" $(ls -t results/slurm_*.out 2>/dev/null | head -1) 2>/dev/null || echo "No .out file found"; }; _sgrep'

# Grep in most recent error
alias sgreperr='_sgreperr() { grep --color=auto "$1" $(ls -t results/slurm_*.err 2>/dev/null | head -1) 2>/dev/null || echo "No .err file found"; }; _sgreperr'

# ============================================================
# QUICK SUMMARY
# ============================================================

# Show key info from most recent run
alias ssum='_ssum() {
    OUT=$(ls -t results/slurm_*.out 2>/dev/null | head -1)
    ERR=$(ls -t results/slurm_*.err 2>/dev/null | head -1)
    echo "========================================";
    echo "LATEST JOB SUMMARY";
    echo "========================================";
    echo "Output file: $OUT";
    echo "Error file:  $ERR";
    echo "";
    echo "--- Job Info ---";
    grep -E "Job ID:|Start time:|End time:|Exit code:" "$OUT" 2>/dev/null;
    echo "";
    echo "--- Last 20 lines ---";
    tail -20 "$OUT" 2>/dev/null;
    echo "";
    echo "--- Errors (if any) ---";
    cat "$ERR" 2>/dev/null | grep -i "error\|fail\|exception" | head -10;
}; _ssum'

echo "✓ Simple SLURM aliases loaded!"
echo "Try: sout, serr, stail, sls, sstatus"
