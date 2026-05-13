#!/bin/bash
# ============================================================================
# SpecForge DFlash training launcher — Qwen3-8B on Ascend NPU, SGLANG backend.
#
# Same defaults as run_qwen3_8b_dflash_npu.sh, but the target model is served
# via sglang in-process (with tensor parallel across NPUs) instead of plain
# HuggingFace forward. SGLang backend gives lower target-side latency once
# tuned, but requires the full sglang+sgl_kernel_npu+triton-ascend stack
# (which you have, from installation_zh.md Steps 4-5).
#
# This is the "let's see what happens" path. Expect first-run hiccups —
# sglang's NPU backend is younger than its CUDA path; some attention
# backends or runtime flags may not work the first time. The script
# defaults to conservative choices; fallbacks are documented inline.
#
# Override anything via env vars, e.g.:
#
#   # Sanity test (1 NPU, tiny everything)
#   NUM_NPUS=1 TP_SIZE=1 ASCEND_RT_VISIBLE_DEVICES=0 \
#       BATCH_SIZE=1 MAX_LENGTH=512 NUM_ANCHORS=16 NUM_EPOCHS=1 \
#       LOG_INTERVAL=1 SAVE_INTERVAL=999999 \
#       OUTPUT_DIR=./outputs/sanity-test-sglang \
#       bash docs/ascend_npu/run_qwen3_8b_dflash_npu_sglang.sh
#
#   # Full 8-NPU run with target sharded across all 8 (tp=8)
#   bash docs/ascend_npu/run_qwen3_8b_dflash_npu_sglang.sh
#
# ============================================================================

set -euo pipefail

# ---- Paths ----
TARGET_MODEL=${TARGET_MODEL:-/share/canada_group_folder/ckpt/Qwen3-8B}
TRAIN_DATA=${TRAIN_DATA:-/share/canada_group_folder/dataset/perfectblend_train_regen.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/qwen3-8b-dflash-npu-sglang}

# ---- Devices ----
NUM_NPUS=${NUM_NPUS:-8}
TP_SIZE=${TP_SIZE:-$NUM_NPUS}     # target sharded across all NPUs by default
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- DFlash hyperparams ----
BATCH_SIZE=${BATCH_SIZE:-2}
MAX_LENGTH=${MAX_LENGTH:-3072}
NUM_EPOCHS=${NUM_EPOCHS:-6}
LR=${LR:-6e-4}
NUM_ANCHORS=${NUM_ANCHORS:-512}
BLOCK_SIZE=${BLOCK_SIZE:-16}
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.04}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

# ---- SGLang backend hyperparams ----
# Attention backend on NPU. Try `triton` first (uses triton-ascend installed
# in Step 5). If sglang complains it's not registered, try in order:
#   - "ascend"      — if upstream sglang has the NPU-specific kernel path
#   - "torch_native"— pure-pytorch fallback, slowest but always works
# Do NOT use the default `flashinfer` or `fa3` — they are CUDA-only.
SGLANG_ATTENTION_BACKEND=${SGLANG_ATTENTION_BACKEND:-triton}
# Fraction of NPU HBM that sglang takes for static state (model weights + KV
# pool). 0.4 leaves ~60% for our FSDP draft + activations + grads. Lower if
# you OOM on the target side; raise if you have headroom.
SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.4}

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
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/qwen3-8b-dflash.json}

cat <<EOF
=================== SpecForge DFlash training (NPU, SGLANG) ===================
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
==============================================================================
EOF

[[ -f "$DRAFT_CONFIG" ]] || { echo "ERROR: DRAFT_CONFIG not found: $DRAFT_CONFIG" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: TARGET_MODEL dir not found: $TARGET_MODEL" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]   || { echo "ERROR: TRAIN_DATA not found: $TRAIN_DATA" >&2; exit 1; }

# Sanity: TP_SIZE must divide NUM_NPUS evenly
if (( NUM_NPUS % TP_SIZE != 0 )); then
    echo "ERROR: NUM_NPUS ($NUM_NPUS) must be divisible by TP_SIZE ($TP_SIZE)" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

cd "$ROOT_DIR"

# See note in run_qwen3_8b_dflash_npu.sh about why we avoid --standalone.
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
    --trust-remote-code
