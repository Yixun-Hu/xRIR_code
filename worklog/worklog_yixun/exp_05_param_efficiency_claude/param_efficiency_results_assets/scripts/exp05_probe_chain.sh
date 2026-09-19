#!/bin/bash
# usage: exp05_probe_chain.sh <gpu> <reviewed_commit> ; four sequential exp_05 fit probes (S then L, simple then cylindrical)
S=/tmp/claude-1013/-home-yixunhu-codespace-xRIR-code/279323f7-0bed-4b41-8631-99735eadc6f1/scratchpad
GPU=$1; SHA=$2
for arm in "S simple" "S cylindrical" "L simple" "L cylindrical"; do
  tier=${arm% *}; bb=${arm#* }
  $S/exp05_probe_run.sh "$GPU" "$SHA" "$tier" "$bb"
done
echo "EXP05_PROBE_CHAIN_DONE $(date -Is)"
