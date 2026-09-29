#!/bin/bash
#SBATCH --job-name=psmareg_nnunet
#SBATCH --output=/home/iml/fryderyk.koegl/logs/%x_%j.out
#SBATCH --error=/home/iml/fryderyk.koegl/logs/%x_%j.err
#SBATCH -t 1-00:00:00
#SBATCH -p gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --mem=80G
#SBATCH --nice=10000
#SBATCH --gres=gpu:h100:1  # needs ~26 GB; plain gpu:1 can land on 20 GB A100 MIG slices
#SBATCH --cpus-per-task=16

# Train one fold of the autoPET3-style nnU-Net on PSMAReg.
# Usage: sbatch psmareg/train_fold.sbatch <fold>  (or psmareg/submit_folds.sh for chained 24h jobs)
# Resubmitting the same fold resumes from checkpoint_latest.pth; a finished fold exits immediately.
set -euo pipefail

FOLD=${1:?usage: sbatch train_fold.sbatch <fold 0-4>}
source /home/iml/fryderyk.koegl/code/autopet-3-submission/psmareg/env.sh
export nnUNet_n_proc_DA=12  # training is GPU-bound, more workers did not help

TRAINER=autoPET3_Trainer
PLANS=nnUNetResEncUNetLPlansMultiTalent
FOLD_DIR=$nnUNet_results/Dataset901_PSMAReg_lesions/${TRAINER}__${PLANS}__3d_fullres/fold_$FOLD

if [ -f "$FOLD_DIR/checkpoint_final.pth" ]; then
    echo "fold $FOLD already finished"
    exit 0
elif [ -f "$FOLD_DIR/checkpoint_latest.pth" ]; then
    INIT=(--c)
else
    INIT=(-pretrained_weights "$PRETRAINED")
fi

echo "fold $FOLD on $(hostname), init: ${INIT[*]}"
nvidia-smi
nnUNetv2_train 901 3d_fullres "$FOLD" -tr $TRAINER -p $PLANS "${INIT[@]}"
