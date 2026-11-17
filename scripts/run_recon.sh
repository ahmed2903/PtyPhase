#!/bin/bash
# Per-streak reconstruction of real data (recon.py, ptyphase.runner.FourierPtychoEngine) as one
# Slurm job on Maxwell. All arguments are passed straight to recon.py (see `python recon.py --help`);
# anything not given uses the script's defaults (shared pupil, no Zernike projection).
#
# Submit from the folder the relative paths refer to (outputs go there too):
#   cd /gpfs/cfel/user/mohahmed/BP_data_recons_Apr26/inp_nanowires
#   sbatch /gpfs/cfel/user/mohahmed/ptyphase/scripts/run_recon.sh \
#          --input newData/roi-A_scan-484_binned4.h5 --steps dm:300:0.9 er:50
#
# Independent pupils (one per streak), after 100 shared-pupil iterations:
#   OUTDIR=recon_indep sbatch .../run_recon.sh --input ... --steps dm:300 er:50 \
#          --independent_pupils --independent_from 100
#
# Shared vs independent side by side:
#   sbatch .../run_recon.sh --input newData/roi-A_scan-484_binned4.h5 --steps dm:300 er:50
#   sbatch .../run_recon.sh --input newData/roi-A_scan-484_binned4.h5 --steps dm:300 er:50 --independent_pupils
#
#SBATCH --job-name=fp_recon
#SBATCH --time=48:00:00
#SBATCH --partition=allcpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmed.mokhtar@desy.de
#SBATCH --output=/gpfs/cfel/user/mohahmed/BP_data_recons_Apr26/logs/recon_%j.out
#SBATCH --error=/gpfs/cfel/user/mohahmed/BP_data_recons_Apr26/logs/recon_%j.err

set -euo pipefail

SCRIPT=/gpfs/cfel/user/mohahmed/ptyphase/scripts/recon.py
PYTHON=/gpfs/cfel/user/mohahmed/envs/ptyxplore/bin/python
WORKDIR=${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}                  # override: WORKDIR=... sbatch ...
OUTDIR=${OUTDIR:-recon}                                          # override: OUTDIR=... sbatch ...

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "workdir: $WORKDIR"
echo "args: $*"

cd "$WORKDIR"
"$PYTHON" "$SCRIPT" --outdir "$OUTDIR" --tag "job${SLURM_JOB_ID:-local}" "$@" \
    | grep --line-buffered -v -E "^(Step done|Object Update|Pupil Update|Zernike Update|Function '.*' executed in)"

echo "done $(date)"
