#!/usr/bin/env bash
# Run a pairwise round-robin tournament between 4 transformation families,
# then a championship match between the top 2.
#
# Usage:
#   bash run_tournament.sh [OUTPUT_DIR] [N_EPOCHS_ROUND_ROBIN] [N_EPOCHS_CHAMPIONSHIP]
#
# Defaults:
#   OUTPUT_DIR=./results_tournament
#   N_EPOCHS_ROUND_ROBIN=500000
#   N_EPOCHS_CHAMPIONSHIP=1000000

set -euo pipefail

OUTPUT_DIR="${1:-./results_tournament}"
N_EPOCHS_RR="${2:-500000}"
N_EPOCHS_H2H="${3:-1000000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/train_learned_policy.py"
ANALYZE_SCRIPT="${SCRIPT_DIR}/analyze_tournament.py"

RR_DIR="${OUTPUT_DIR}/round_robin"
mkdir -p "${RR_DIR}"

# Transform IDs: 0=translation, 1=scaling, 2=quadratic, 3=random
TRANSFORMS=(translation scaling quadratic random)
IDS=(0 1 2 3)

echo "============================================"
echo " Transform Tournament"
echo "============================================"
echo "Round-robin epochs per match: ${N_EPOCHS_RR}"
echo "Championship epochs: ${N_EPOCHS_H2H}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Round-robin: all 6 pairwise matches
for i in 0 1 2 3; do
  for j in $(seq $((i+1)) 3); do
    NAME_A="${TRANSFORMS[$i]}"
    NAME_B="${TRANSFORMS[$j]}"
    MATCH_DIR="${RR_DIR}/${NAME_A}_vs_${NAME_B}"
    echo "--- Match: ${NAME_A} vs ${NAME_B} ---"

    python3 "${TRAIN_SCRIPT}" \
      --enabled_transforms="${IDS[$i]},${IDS[$j]}" \
      --policy_guided_calibration \
      --n_epochs="${N_EPOCHS_RR}" \
      --calibration_window_size=500 \
      --eval_calibration_batches=100 \
      --save_dir="${MATCH_DIR}" \
      2>&1 | tee "${MATCH_DIR}/train.log"
    echo ""
  done
done

# Analyze round-robin
echo "--- Analyzing round-robin results ---"
python3 "${ANALYZE_SCRIPT}" --results_dir="${RR_DIR}" --write_json
echo ""

# Extract top 2 finalists from summary.json
FINALIST_A=$(python3 -c "import json; s=json.load(open('${RR_DIR}/summary.json')); print(s['finalists'][0])")
FINALIST_B=$(python3 -c "import json; s=json.load(open('${RR_DIR}/summary.json')); print(s['finalists'][1])")

# Map names back to IDs
name_to_id() {
  case "$1" in
    translation) echo 0 ;;
    scaling) echo 1 ;;
    quadratic) echo 2 ;;
    random) echo 3 ;;
  esac
}

ID_A=$(name_to_id "${FINALIST_A}")
ID_B=$(name_to_id "${FINALIST_B}")

echo "============================================"
echo " Championship: ${FINALIST_A} vs ${FINALIST_B}"
echo "============================================"

H2H_DIR="${OUTPUT_DIR}/championship_${FINALIST_A}_vs_${FINALIST_B}"
mkdir -p "${H2H_DIR}"

python3 "${TRAIN_SCRIPT}" \
  --enabled_transforms="${ID_A},${ID_B}" \
  --policy_guided_calibration \
  --n_epochs="${N_EPOCHS_H2H}" \
  --calibration_window_size=500 \
  --eval_calibration_batches=100 \
  --save_dir="${H2H_DIR}" \
  2>&1 | tee "${H2H_DIR}/train.log"

echo ""
echo "============================================"
echo " Tournament complete!"
echo " Round-robin results: ${RR_DIR}/summary.json"
echo " Championship results: ${H2H_DIR}/"
echo "============================================"
