#!/usr/bin/env bash
set -euo pipefail

: "${UNIBRAIN_ROOT:?Set the deployment root}"
: "${UNIBRAIN_DATA_ROOT:?Set the prepared fold root}"
: "${UNIBRAIN_RUN_ROOT:?Set the experiment output root}"
dataset=${1:?Specify abide or adhd}
mode=${2:-full}
case "$dataset" in abide|adhd) ;; *) exit 2 ;; esac
case "$mode" in smoke|full) ;; *) exit 2 ;; esac
export UNIBRAIN_SOURCES="$UNIBRAIN_ROOT/sources"
export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
python="$UNIBRAIN_ROOT/env/bin/python"
cd "$UNIBRAIN_ROOT/code"
output="$UNIBRAIN_RUN_ROOT/${dataset}_${mode}"
data="$UNIBRAIN_DATA_ROOT/${dataset}_round1"
options=(--epochs 100 --patience 10 --evaluate-test --save-predictions)
if [[ "$mode" == smoke ]]; then
    data="$UNIBRAIN_DATA_ROOT/${dataset}_smoke"
    options=(--epochs 2 --max-steps 2 --eval-limit 2)
fi
mkdir -p "$UNIBRAIN_RUN_ROOT"
date -Is
hostname
nvidia-smi --query-gpu=index,uuid,name,memory.free --format=csv
command=("$python" scripts/train_replacement.py
    --source "$UNIBRAIN_SOURCES/reg_unibrain" --data "$data" --output "$output"
    --method synthstrip --setting joint --initialization random
    --adapter "$UNIBRAIN_ROOT/adapters/synthstrip.py"
    --dataset "$dataset" --fold 1 --seed 17 --gpu 0
    --threads "$OMP_NUM_THREADS" --connect-paths --verify-gradients
    --checkpoint-mode resume --selection-metric classification_loss)
if [[ -f "$output/last.pt" ]]; then
    command+=(--resume "$output/last.pt")
fi
printf 'COMMAND: '; printf '%q ' "${command[@]}" "${options[@]}"; printf '\n'
"${command[@]}" "${options[@]}"
date -Is
