#!/bin/bash
#SBATCH --job-name=psmareg_nnunet
#SBATCH --output=/home/iml/fryderyk.koegl/logs/%x_%j.out
#SBATCH --error=/home/iml/fryderyk.koegl/logs/%x_%j.err
#SBATCH -t 4-00:00:00
#SBATCH -p gpu_p
#SBATCH --mem=80G
#SBATCH --qos=gpu_reservation
#SBATCH --nice=10000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16

# Train one fold of the autoPET3-style nnU-Net on PSMAReg.
# Usage: sbatch psmareg/train_fold.sbatch <fold>
# Resubmitting the same fold resumes from checkpoint_latest.pth (e.g. after hitting the time limit).
set -euo pipefail

FOLD=${1:?usage: sbatch train_fold.sbatch <fold 0-4>}
source /home/iml/fryderyk.koegl/code/autopet-3-submission/psmareg/env.sh
export nnUNet_n_proc_DA=12  # training is GPU-bound, more workers did not help

TRAINER=autoPET3_Trainer
PLANS=nnUNetResEncUNetLPlansMultiTalent
FOLD_DIR=$nnUNet_results/Dataset901_PSMAReg_lesions/${TRAINER}__${PLANS}__3d_fullres/fold_$FOLD

if [ -f "$FOLD_DIR/checkpoint_latest.pth" ] || [ -f "$FOLD_DIR/checkpoint_final.pth" ]; then
    INIT=(--c)
else
    INIT=(-pretrained_weights "$PRETRAINED")
fi

echo "fold $FOLD on $(hostname), init: ${INIT[*]}"
nvidia-smi
nnUNetv2_train 901 3d_fullres "$FOLD" -tr $TRAINER -p $PLANS "${INIT[@]}"
