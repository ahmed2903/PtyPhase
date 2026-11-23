#!/bin/bash
# Gradient-based 3D reconstruction of real data (recon_opt.py, ptyphase.optimisation) as one
# Slurm job on a Maxwell GPU node. All arguments are passed straight to recon_opt.py
# (see `python recon_opt.py --help`); anything not given uses the script's defaults.
#
# Submit from the folder the relative paths refer to (outputs go there too):
#   cd /gpfs/cfel/user/mohahmed/BP_data_recons_Apr26/inp_nanowires
#   sbatch /gpfs/cfel/user/mohahmed/ptyphase/scripts/run_recon_opt.sh \
#          --input newData/roi-A_scan-484_binned4.h5 --epochs 2000 --lr_spectrum 3e-2 --pupil_start 200
#
# Polynomial pupil started from speckle tracking:
#   OUTDIR=recon_opt_poly sbatch .../run_recon_opt.sh --input ... --pupil_model poly --meta_pupil st_phase.npy
#
# One job per axis flip:
#   for f in xy x_neg_y neg_x_y neg_x_neg_y; do
#       sbatch .../run_recon_opt.sh --input newData/roi-A_scan-484_binned4.h5 --epochs 2000 --axis_flip $f
#   done
#
#SBATCH --job-name=fp_recon_opt
#SBATCH --time=24:00:00
#SBATCH --partition=allgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahmed.mokhtar@desy.de
#SBATCH --output=/gpfs/cfel/user/mohahmed/BP_data_recons_Apr26/logs/recon_opt_%j.out
#SBATCH --error=/gpfs/cfel/user/mohahmed/BP_data_recons_Apr26/logs/recon_opt_%j.err

set -euo pipefail

SCRIPT=/gpfs/cfel/user/mohahmed/ptyphase/scripts/recon_opt.py
PYTHON=/gpfs/cfel/user/mohahmed/envs/ptyxplore/bin/python
WORKDIR=${WORKDIR:-${SLURM_SUBMIT_DIR:-$PWD}}                  # override: WORKDIR=... sbatch ...
OUTDIR=${OUTDIR:-recon_opt}                                      # override: OUTDIR=... sbatch ...

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

echo "job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "workdir: $WORKDIR"
echo "args: $*"
nvidia-smi -L || true

cd "$WORKDIR"
"$PYTHON" "$SCRIPT" --outdir "$OUTDIR" --tag "job${SLURM_JOB_ID:-local}" "$@" \
    | grep --line-buffered -v -E "^(Function '.*' executed in)"

echo "done $(date)"
