#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

CONFIG_FILE="${CONFIG_FILE:-configs/transreid_nfc_scfg_w10k29_to_full12.yml}"
GPUS="${1:-${GPUS:-0}}"
OUTPUT_DIR="${2:-${OUTPUT_DIR:-outputs/w10k29_to_full12_scfg_$(date +%Y%m%d_%H%M%S)}}"
EPOCHS="${3:-${EPOCHS:-60}}"
BATCH_SIZE="${4:-${BATCH_SIZE:-128}}"
BASE_LR="${BASE_LR:-0.004}"
MASTER_PORT="${MASTER_PORT:-29531}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-data/WildlifeReID-10K}"
PRETRAIN_PATH="${PRETRAIN_PATH:-pretrained/jx_vit_base_p16_224-80ecf9dd.pth}"

NUM_GPUS="$("${PYTHON_BIN}" - "$GPUS" <<'PY'
import sys
print(len([x for x in sys.argv[1].split(",") if x.strip()]))
PY
)"

export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"

mkdir -p "$OUTPUT_DIR"

cat > "${OUTPUT_DIR}/launch_command.txt" <<EOF
CONFIG_FILE=${CONFIG_FILE}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
OUTPUT_DIR=${OUTPUT_DIR}
DATA_ROOT=${DATA_ROOT}
PRETRAIN_PATH=${PRETRAIN_PATH}
EPOCHS=${EPOCHS}
BATCH_SIZE=${BATCH_SIZE}
BASE_LR=${BASE_LR}
MASTER_PORT=${MASTER_PORT}
EOF

echo "Training MetaN: TransReID + SCFG + NFC"
echo "Config:      ${CONFIG_FILE}"
echo "Output:      ${OUTPUT_DIR}"
echo "Data root:   ${DATA_ROOT}"
echo "Pretrain:    ${PRETRAIN_PATH}"
echo "GPUs:        ${CUDA_VISIBLE_DEVICES}"
echo "Num GPUs:    ${NUM_GPUS}"
echo "Epochs:      ${EPOCHS}"
echo "Batch size:  ${BATCH_SIZE}"
echo "Base LR:     ${BASE_LR}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --nproc_per_node="${NUM_GPUS}" \
  --master_port="${MASTER_PORT}" \
  train.py \
  --config_file "${CONFIG_FILE}" \
  OUTPUT_DIR "${OUTPUT_DIR}" \
  DATASETS.TRAIN_ROOTS "['${DATA_ROOT}']" \
  MODEL.PRETRAIN_PATH "${PRETRAIN_PATH}" \
  SOLVER.MAX_EPOCHS "${EPOCHS}" \
  SOLVER.IMS_PER_BATCH "${BATCH_SIZE}" \
  SOLVER.BASE_LR "${BASE_LR}"
