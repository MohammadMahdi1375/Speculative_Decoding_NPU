#!/bin/bash
# ============================================================================
# SpecForge DFlash training launcher — Qwen3.5-9B on Ascend NPU, SGLANG backend.
#
# Adapted from run_qwen3_8b_dflash_npu_sglang.sh for the Qwen3.5-9B target.
#
# KEY DIFFERENCES FROM Qwen3-8B:
#   - Qwen3.5-9B uses Qwen3_5ForConditionalGeneration (VL-style wrapper),
#     NOT Qwen3ForCausalLM. The sglang model registry routes to
#     sglang.srt.models.qwen3_5 which does NOT natively expose
#     set_eagle3_layers_to_capture / aux_hidden_states.
#   - We apply qwen3_5_patch.py BEFORE training starts to monkey-patch
#     the hidden-state capture contract into the qwen3_5 model classes.
#   - Draft config must match Qwen3.5-9B's hidden_size (3584) and
#     num_hidden_layers (36). If you don't have a dedicated config yet,
#     see the NOTE below on creating one.
#
# Usage:
#   # Full 8-NPU run
#   bash docs/ascend_npu/run_qwen3_5_9b_dflash_npu_sglang.sh
#
#   # Sanity test (1 NPU)
#   NUM_NPUS=1 TP_SIZE=1 ASCEND_RT_VISIBLE_DEVICES=0 \
#       BATCH_SIZE=1 MAX_LENGTH=512 NUM_ANCHORS=16 NUM_EPOCHS=1 \
#       LOG_INTERVAL=1 SAVE_INTERVAL=999999 \
#       OUTPUT_DIR=./outputs/sanity-test-qwen3.5-9b-sglang \
#       bash docs/ascend_npu/run_qwen3_5_9b_dflash_npu_sglang.sh
#
# ============================================================================

set -euo pipefail

# ---- Paths ----
TARGET_MODEL=${TARGET_MODEL:-/home/n84449292/m84379596/Huggingface/Qwen3.5-9B/}
TRAIN_DATA=${TRAIN_DATA:-/home/n84449292/m84379596/DFlash/Specforge_NPU/SpecForge/cache/dataset/sharegpt_train_filtered.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/qwen3.5-9b-dflash-npu-sglang}

# ---- Devices ----
NUM_NPUS=${NUM_NPUS:-8}
TP_SIZE=${TP_SIZE:-$NUM_NPUS}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- DFlash hyperparams ----
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-3072}
NUM_EPOCHS=${NUM_EPOCHS:-6}
LR=${LR:-6e-4}
NUM_ANCHORS=${NUM_ANCHORS:-512}
BLOCK_SIZE=${BLOCK_SIZE:-16}
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.04}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ---- SGLang backend hyperparams ----
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-ascend}
# Qwen3.5-9B is slightly larger than Qwen3-8B (3584 hidden, 36 layers vs
# 4096 hidden, 32 layers — but 9B total params are close). 0.4 should still
# work; lower to 0.35 if you OOM on the target side.
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.05}

# ---- Logging / checkpointing ----
LOG_INTERVAL=${LOG_INTERVAL:-50}
SAVE_INTERVAL=${SAVE_INTERVAL:-1000}
REPORT_TO=${REPORT_TO:-tensorboard}

# ---- NPU runtime env ----
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-100000}
export NPU_ASD_ENABLE=${NPU_ASD_ENABLE:-0}
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}

# ---- Auto-locate SpecForge root ----
ROOT_DIR=/home/n84449292/m84379596/DFlash/Specforge_NPU/SpecForge


# ---- Draft config ----
# NOTE: You need a draft config that matches Qwen3.5-9B's architecture.
# Qwen3.5-9B has hidden_size=3584, num_hidden_layers=36.
# If you already have configs/qwen3.5-9b-dflash.json, great.
# If not, copy configs/qwen3-8b-dflash.json and update:
#   - "hidden_size": 3584
#   - "intermediate_size": 18944  (check model's config.json)
#   - "num_hidden_layers": 36
#   - "num_attention_heads": 28
#   - "num_key_value_heads": 4
#   - Any other architecture fields that differ
# Or point DRAFT_CONFIG at a pre-existing one.
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/qwen3.5-9b-dflash.json}

# ---- Apply Qwen3.5 sglang patch ----
# The patch monkey-patches sglang.srt.models.qwen3_5 to expose
# set_eagle3_layers_to_capture + aux_hidden_states, which SpecForge's
# sglang DFlash backend requires. Without it you'll hit:
#   "mat1 and mat2 shapes cannot be multiplied (N x 3584 and 17920 x 3584)"
# We set QWEN3_5_DFLASH_PATCH so train_dflash.py can import & apply it
# before instantiating the sglang engine.
QWEN3_5_PATCH=${QWEN3_5_PATCH:-$(dirname "$SCRIPT_DIR")/patches/qwen3_5_patch.py}
if [[ -f "$QWEN3_5_PATCH" ]]; then
    echo "INFO: Qwen3.5 DFlash patch found at $QWEN3_5_PATCH"
    # Make patch importable — add its parent dir to PYTHONPATH
    PATCH_DIR=$(dirname "$QWEN3_5_PATCH")
    export PYTHONPATH="${PATCH_DIR}${PYTHONPATH:+:$PYTHONPATH}"
    # Signal train_dflash.py to apply the patch
    export SPECFORGE_QWEN3_5_PATCH=1
else
    echo "WARNING: Qwen3.5 patch not found at $QWEN3_5_PATCH"
    echo "         If sglang's qwen3_5 model doesn't natively support"
    echo "         set_eagle3_layers_to_capture, training will fail."
    echo "         Set QWEN3_5_PATCH=/path/to/qwen3_5_patch.py to fix."
fi

cat <<EOF
================ SpecForge DFlash training (NPU, SGLANG, Qwen3.5-9B) ================
ROOT_DIR                       : $ROOT_DIR
TARGET_MODEL                   : $TARGET_MODEL
DRAFT_CONFIG                   : $DRAFT_CONFIG
TRAIN_DATA                     : $TRAIN_DATA
OUTPUT_DIR                     : $OUTPUT_DIR
NUM_NPUS / TP_SIZE             : $NUM_NPUS / $TP_SIZE
ASCEND_RT_VISIBLE_DEVICES      : $ASCEND_RT_VISIBLE_DEVICES
BATCH_SIZE / MAX_LENGTH        : $BATCH_SIZE / $MAX_LENGTH
SGLANG_ATTENTION_BACKEND       : $SGLANG_ATTENTION_BACKEND
SGLANG_MEM_FRACTION_STATIC     : $SGLANG_MEM_FRACTION_STATIC
QWEN3_5_PATCH                  : ${QWEN3_5_PATCH:-"(not set)"}
==================================================================================
EOF

[[ -f "$DRAFT_CONFIG" ]] || { echo "ERROR: DRAFT_CONFIG not found: $DRAFT_CONFIG" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: TARGET_MODEL dir not found: $TARGET_MODEL" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]   || { echo "ERROR: TRAIN_DATA not found: $TRAIN_DATA" >&2; exit 1; }

if (( NUM_NPUS % TP_SIZE != 0 )); then
    echo "ERROR: NUM_NPUS ($NUM_NPUS) must be divisible by TP_SIZE ($TP_SIZE)" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"

MASTER_ADDR_LOCAL=${MASTER_ADDR_LOCAL:-127.0.0.1}
MASTER_PORT_LOCAL=${MASTER_PORT_LOCAL:-30060}

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
    --target-model-backend sglang \
    --tp-size "$TP_SIZE" \
    --attention-backend sdpa \
    --sglang-attention-backend "$SGLANG_ATTENTION_BACKEND" \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
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
    --save-interval "$SAVE_INTERVAL" \
    --embedding-key model.language_model.embed_tokens.weight \
    --trust-remote-code