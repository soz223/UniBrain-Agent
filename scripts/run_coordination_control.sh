#!/usr/bin/env bash
set -euo pipefail

dataset=$1
control=$2
fold=${FOLD:-1}
root=${UNIBRAIN_SOURCES:?}
data="${FOLD_ROOT:?}/${dataset}_round${fold}"
out="${RUN_ROOT:?}/${dataset}_${control}"
cd "$(dirname "$(realpath "${BASH_SOURCE[0]}")")/.."
mkdir -p "$out"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
if [[ $control == volume ]]; then
    exec "${TRAIN_PYTHON:?}" scripts/volume_baseline.py "$data" "$out/validation.json"
fi
args=(--source "$root/reg_unibrain" --data "$data" --output "$out" --dataset "$dataset"
      --method original --setting original --paper-original --fold "$fold" --seed 17
      --epochs 1000 --patience 50 --selection-metric classification_loss
      --template-normalization minmax --train-probe-size 64 --verify-gradients
      --checkpoint-mode resume --threads 2 --gpu 0 --wall-seconds "${WALL_SECONDS:-82500}")
case "$control" in
    equal) args+=(--coordination none --paper-registration-weight 1) ;;
    pcgrad) args+=(--coordination pcgrad) ;;
    taskforce) args+=(--coordination taskforce --taskforce-profile paper --taskforce-gradient-normalization l2 --coordination-reward same_subject) ;;
    *) echo "Unknown control: $control" >&2; exit 2 ;;
esac
if [[ -e "$out/last.pt" ]]; then
    args+=(--resume "$out/last.pt")
fi
exec "${TRAIN_PYTHON:?}" scripts/train_replacement.py "${args[@]}"
