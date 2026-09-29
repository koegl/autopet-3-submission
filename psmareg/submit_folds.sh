#!/bin/bash
# Submit a chain of 24h jobs per fold; each job resumes the previous one (afterany dependency).
# Usage: psmareg/submit_folds.sh <n_jobs_per_fold> <fold> [<fold> ...]    e.g. psmareg/submit_folds.sh 6 1 2 3 4
set -euo pipefail
N=$1; shift
SBATCH_FILE=/home/iml/fryderyk.koegl/code/autopet-3-submission/psmareg/train_fold.sbatch
for FOLD in "$@"; do
    JID=$(sbatch --parsable --job-name=psmareg_nnunet_f$FOLD "$SBATCH_FILE" "$FOLD")
    CHAIN=$JID
    for _ in $(seq 2 "$N"); do
        JID=$(sbatch --parsable --job-name=psmareg_nnunet_f$FOLD --dependency=afterany:$JID "$SBATCH_FILE" "$FOLD")
        CHAIN="$CHAIN $JID"
    done
    echo "fold $FOLD: $CHAIN"
done
