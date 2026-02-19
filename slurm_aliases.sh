#!/bin/bash
# SLURM Log Viewing Aliases
# Add these to your ~/.bashrc on MIT ORCD
#
# Usage after adding to ~/.bashrc:
#   source ~/.bashrc
#   sout 9243048      # View .out file
#   serr 9243048      # View .err file
#   slog 9243048      # View both files
#   stail 9243048     # Live tail of .out file

# ============================================================
# SLURM LOG ALIASES
# ============================================================

# View SLURM .out file
alias sout='_slurm_out() { cat results/slurm_"$1"_*.out 2>/dev/null || echo "No .out file found for job $1"; }; _slurm_out'

# View SLURM .err file
alias serr='_slurm_err() { cat results/slurm_"$1"_*.err 2>/dev/null || echo "No .err file found for job $1"; }; _slurm_err'

# View both .out and .err files
alias slog='_slurm_log() {
    echo "========================================";
    echo "SLURM OUTPUT (job $1)";
    echo "========================================";
    cat results/slurm_"$1"_*.out 2>/dev/null || echo "No .out file found";
    echo "";
    echo "========================================";
    echo "SLURM ERRORS (job $1)";
    echo "========================================";
    cat results/slurm_"$1"_*.err 2>/dev/null || echo "No .err file found";
}; _slurm_log'

# Live tail of .out file (for monitoring running jobs)
alias stail='_slurm_tail() { tail -f results/slurm_"$1"_*.out 2>/dev/null || echo "No .out file found for job $1"; }; _slurm_tail'

# Live tail of .err file
alias stailerr='_slurm_tailerr() { tail -f results/slurm_"$1"_*.err 2>/dev/null || echo "No .err file found for job $1"; }; _slurm_tailerr'

# Show last 50 lines of .out
alias slast='_slurm_last() { tail -50 results/slurm_"$1"_*.out 2>/dev/null || echo "No .out file found for job $1"; }; _slurm_last'

# Show last 50 lines of .err
alias slasterr='_slurm_lasterr() { tail -50 results/slurm_"$1"_*.err 2>/dev/null || echo "No .err file found for job $1"; }; _slurm_lasterr'

# List all SLURM log files
alias sls='ls -lth results/slurm_*.{out,err} 2>/dev/null | head -20'

# Find logs by job ID pattern
alias sfind='_slurm_find() { ls -lth results/slurm_*"$1"*.{out,err} 2>/dev/null; }; _slurm_find'

# Quick status: show job queue + latest logs
alias sstatus='_slurm_status() {
    echo "========================================";
    echo "YOUR JOBS";
    echo "========================================";
    squeue -u $USER;
    echo "";
    echo "========================================";
    echo "LATEST LOG FILES";
    echo "========================================";
    ls -lth results/slurm_*.{out,err} 2>/dev/null | head -10;
}; _slurm_status'

# Grep in output file
alias sgrep='_slurm_grep() {
    if [ -z "$2" ]; then
        echo "Usage: sgrep <job_id> <pattern>";
        return 1;
    fi
    echo "Searching in .out file...";
    grep --color=auto "$2" results/slurm_"$1"_*.out 2>/dev/null;
    echo "";
    echo "Searching in .err file...";
    grep --color=auto "$2" results/slurm_"$1"_*.err 2>/dev/null;
}; _slurm_grep'

# ============================================================
# BONUS: Project-specific aliases
# ============================================================

# Quick view of all step1 logs
alias s1logs='ls -lth results/slurm_*_step1_*.{out,err} 2>/dev/null | head -5 && echo "" && echo "Latest step1 output:" && tail -50 results/slurm_*_step1_*.out 2>/dev/null | tail -50'

# Quick view of all step2 logs
alias s2logs='ls -lth results/slurm_*_step2_*.{out,err} 2>/dev/null | head -5 && echo "" && echo "Latest step2 output:" && tail -50 results/slurm_*_step2_*.out 2>/dev/null | tail -50'

# Quick view of all step3 logs
alias s3logs='ls -lth results/slurm_*_step3_*.{out,err} 2>/dev/null | head -5 && echo "" && echo "Latest step3 output:" && tail -50 results/slurm_*_step3_*.out 2>/dev/null | tail -50'

# Quick view of all step4 logs
alias s4logs='ls -lth results/slurm_*_step4_*.{out,err} 2>/dev/null | head -5 && echo "" && echo "Latest step4 output:" && tail -50 results/slurm_*_step4_*.out 2>/dev/null | tail -50'

# Quick view of all step5 logs
alias s5logs='ls -lth results/slurm_*_step5_*.{out,err} 2>/dev/null | head -5 && echo "" && echo "Latest step5 output:" && tail -50 results/slurm_*_step5_*.out 2>/dev/null | tail -50'

# Quick view of all step6 logs
alias s6logs='ls -lth results/slurm_*_step6_*.{out,err} 2>/dev/null | head -5 && echo "" && echo "Latest step6 output:" && tail -50 results/slurm_*_step6_*.out 2>/dev/null | tail -50'

# Check for errors in latest logs
alias schkerr='_slurm_checkerr() {
    echo "Checking for ERROR patterns in recent logs...";
    echo "";
    grep -i "error\|fail\|exception\|traceback" results/slurm_*.err 2>/dev/null | tail -20;
}; _slurm_checkerr'

echo "SLURM aliases loaded! ✓"
echo "Try: sout <job_id>, serr <job_id>, slog <job_id>, stail <job_id>"
