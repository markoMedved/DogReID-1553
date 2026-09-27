#!/bin/bash
# Convenient runner script for OpenAnimals on DogReID-1553
# Usage: ./run_dogreid.sh [bot|agw|sbs|mgn|arbase] [NUM_GPUS]

set -e

MODEL_CHOICE=${1:-"mgn"}
GPUS=${2:-"1"}

case "$MODEL_CHOICE" in
    bot)
        CONFIG="configs/DogReID/bot.yml"
        ;;
    agw)
        CONFIG="configs/DogReID/agw.yml"
        ;;
    sbs)
        CONFIG="configs/DogReID/sbs.yml"
        ;;
    mgn)
        CONFIG="configs/DogReID/mgn.yml"
        ;;
    arbase|ar_base|ARBase)
        CONFIG="configs/DogReID/arbase.yml"
        ;;
    *)
        if [ -f "$MODEL_CHOICE" ]; then
            CONFIG="$MODEL_CHOICE"
        else
            echo "Unknown model choice: $MODEL_CHOICE"
            echo "Options: bot, agw, sbs, mgn, arbase (or a direct path to a YAML config)"
            exit 1
        fi
        ;;
esac

echo "=========================================================="
echo "Running OpenAnimals Model: $MODEL_CHOICE"
echo "Config: $CONFIG"
echo "Number of GPUs: $GPUS"
echo "=========================================================="

echo ">>> Phase 1: Training..."
python tools/train_net.py --config-file "$CONFIG" --num-gpus "$GPUS"

echo ">>> Phase 2: Evaluation on DogReID-1553 Benchmark..."
python tools/eval_dogreid.py --config-file "$CONFIG" --eval-open --bootstrap-iter 100

echo ">>> Run complete for $MODEL_CHOICE!"
