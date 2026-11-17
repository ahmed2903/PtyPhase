#!/bin/bash
# Curved-Ewald-slice reconstruction of real data (recon_ewald.py) as one Slurm job on Maxwell.
# All arguments are passed straight to recon_ewald.py (see `python recon_ewald.py --help`);
# anything not given uses the script's defaults.
#
# Submit from the folder the relative paths refer to (outputs go there too):
#   cd /gpfs/cfel/user/mohahmed/BP_data_recons_Apr26/inp_nanowires
#   sbatch /gpfs/cfel/user/mohahmed/ptyphase/scripts/run_recon_ewald.sh \
#          --input newData/roi-A_scan-484_binned4.h5 --steps dnc:300 er:50 --z_extent_nm 300
#   OUTDIR=ewald_dm sbatch .../run_recon_ewald.sh --input ... --steps dm:200:0.9 er:50 \
#          --support_file sup.npy --shrinkwrap_steps 20 --sw_start 100
#
# One job per axis flip:
#   for f in xy x_neg_y neg_x_y neg_x_neg_y; do
#       sbatch .../run_recon_ewald.sh --input newData/roi-A_scan-484_binned4.h5 --steps dnc:300 er:50 \
#                                     --z_extent_nm 300 --axis_flip $f
#   done
#
#SBATCH --job-name=ewald_recon
#SBATCH --time=48:00:00
#SBATCH --partition=allcpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmed.mokhtar@desy.de
#SBATCH --output=/gpfs/cfel/user/mohahmed/BP_data_recons_Apr26/logs/recon_ewald_%j.out
#SBATCH --error=/gpfs/cfel/user/mohahmed/BP_data_recons_Apr26/logs/recon_ewald_%j.err

set -euo pipefail

SCRIPT=/gpfs/cfel/user/mohahmed/ptyphase/scripts/recon_ewald.py
PYTHON=/gpfs/cfel/user/mohahmed/envs/ptyxplore/bin/python
WORKDIR=${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}                  # override: WORKDIR=... sbatch ...
OUTDIR=${OUTDIR:-recon_ewald}                                    # override: OUTDIR=... sbatch ...

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "workdir: $WORKDIR"
echo "args: $*"

cd "$WORKDIR"
"$PYTHON" "$SCRIPT" --outdir "$OUTDIR" --tag "job${SLURM_JOB_ID:-local}" "$@" \
    | grep --line-buffered -v -E "^(Step done|Object Update|Pupil Update|Function '.*' executed in)"

echo "done $(date)"
