#!/bin/bash
# ============================================================================
# SpecForge DFlash training launcher — Qwen3-8B on Ascend NPU, HF backend.
#
# Defaults match the user a00652497's environment:
#   target model : /share/canada_group_folder/ckpt/Qwen3-8B
#   training set : /share/canada_group_folder/dataset/perfectblend_train_regen.jsonl
#   draft config : ./configs/qwen3-8b-dflash.json (SpecForge built-in)
#
# Override anything via env vars without editing this file. Examples:
#
#   # ---- Sanity test (1 NPU, 1 step, tiny batch — verifies the code path) ----
#   NUM_NPUS=1 ASCEND_RT_VISIBLE_DEVICES=0 \
#       BATCH_SIZE=1 MAX_LENGTH=512 NUM_ANCHORS=16 NUM_EPOCHS=1 \
#       LOG_INTERVAL=1 SAVE_INTERVAL=999999 \
#       OUTPUT_DIR=./outputs/sanity-test \
#       bash docs/ascend_npu/run_qwen3_8b_dflash_npu.sh
#
#   # ---- Full 8-NPU training (defaults) ----
#   bash docs/ascend_npu/run_qwen3_8b_dflash_npu.sh
#
# Prereqs (do once, see installation_zh.md):
#   - Source CANN (sets ASCEND_HOME_PATH)
#   - conda activate ./conda/specforge_npu
#   - All deps installed via Steps 0..6
# ============================================================================

set -euo pipefail

# ---- Paths ----
TARGET_MODEL=${TARGET_MODEL:-/share/canada_group_folder/ckpt/Qwen3-8B}
TRAIN_DATA=${TRAIN_DATA:-/share/canada_group_folder/dataset/perfectblend_train_regen.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/qwen3-8b-dflash-npu}

# ---- Devices ----
NUM_NPUS=${NUM_NPUS:-8}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- Hyperparams ----
# Memory budget on a 32 GB Atlas (Qwen3-8B target replicated per-NPU under HF
# backend, max_len=3072): target ~16 GB + draft FSDP ~3.5 GB + activations
# ~6–10 GB @ batch=1 = ~28 GB. batch=2 pushes activations to ~12–20 GB and
# OOMs. Keep BATCH_SIZE=1; raise effective batch via ACCUMULATION_STEPS.
BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
MAX_LENGTH=${MAX_LENGTH:-3072}
NUM_EPOCHS=${NUM_EPOCHS:-6}
LR=${LR:-6e-4}
NUM_ANCHORS=${NUM_ANCHORS:-512}
BLOCK_SIZE=${BLOCK_SIZE:-16}
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.04}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ---- Logging / checkpointing ----
LOG_INTERVAL=${LOG_INTERVAL:-50}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
REPORT_TO=${REPORT_TO:-tensorboard}

# ---- NPU runtime env (from CANN best-practice; see other ascend_npu docs) ----
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-100000}
export NPU_ASD_ENABLE=${NPU_ASD_ENABLE:-0}
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}

# ---- Auto-locate SpecForge repo root (this script lives at docs/ascend_npu/) ----
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/qwen3-8b-dflash.json}

# ---- Sanity-print config ----
cat <<EOF
=================== SpecForge DFlash training (NPU) ===================
ROOT_DIR                  : $ROOT_DIR
TARGET_MODEL              : $TARGET_MODEL
DRAFT_CONFIG              : $DRAFT_CONFIG
TRAIN_DATA                : $TRAIN_DATA
OUTPUT_DIR                : $OUTPUT_DIR
NUM_NPUS                  : $NUM_NPUS
ASCEND_RT_VISIBLE_DEVICES : $ASCEND_RT_VISIBLE_DEVICES
BATCH_SIZE                : $BATCH_SIZE
ACCUMULATION_STEPS        : $ACCUMULATION_STEPS
MAX_LENGTH                : $MAX_LENGTH
NUM_EPOCHS                : $NUM_EPOCHS
LEARNING_RATE             : $LR
NUM_ANCHORS               : $NUM_ANCHORS
BLOCK_SIZE                : $BLOCK_SIZE
LOSS_DECAY_GAMMA          : $LOSS_DECAY_GAMMA
=======================================================================
EOF

# ---- Verify required files exist before torchrun ----
[[ -f "$DRAFT_CONFIG" ]] || { echo "ERROR: DRAFT_CONFIG not found: $DRAFT_CONFIG" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: TARGET_MODEL dir not found: $TARGET_MODEL" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]   || { echo "ERROR: TRAIN_DATA not found: $TRAIN_DATA" >&2; exit 1; }

mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"

# Use explicit 127.0.0.1 instead of --standalone. --standalone resolves this
# machine's hostname for rendezvous, which fails on hosts whose hostname is
# a non-routable external DNS name (e.g. ISP-assigned reverse DNS).
# 127.0.0.1 is always reachable.
MASTER_ADDR_LOCAL=${MASTER_ADDR_LOCAL:-127.0.0.1}
MASTER_PORT_LOCAL=${MASTER_PORT_LOCAL:-29533}

ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES \
torchrun \
    --nproc_per_node "$NUM_NPUS" \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr "$MASTER_ADDR_LOCAL" \
    --master_port "$MASTER_PORT_LOCAL" \
    scripts/train_dflash.py \
    --target-model-path "$TARGET_MODEL" \
    --draft-config-path "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --target-model-backend hf \
    --attention-backend sdpa \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --accumulation-steps "$ACCUMULATION_STEPS" \
    --max-length "$MAX_LENGTH" \
    --learning-rate "$LR" \
    --num-anchors "$NUM_ANCHORS" \
    --block-size "$BLOCK_SIZE" \
    --loss-decay-gamma "$LOSS_DECAY_GAMMA" \
    --warmup-ratio "$WARMUP_RATIO" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --chat-template qwen \
    --report-to "$REPORT_TO" \
    --log-interval "$LOG_INTERVAL" \
    --save-interval "$SAVE_INTERVAL"
