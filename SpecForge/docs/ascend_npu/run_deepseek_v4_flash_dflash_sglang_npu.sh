#!/bin/bash
# ============================================================================
# SpecForge DFlash training launcher — DeepSeek-V4-Flash, SGLang backend on
# 8x Ascend NPU.
#
# Prereq: run prepare_dsv4_stub_target.py WITH --save-stub-weights first.
#   python scripts/prepare_dsv4_stub_target.py \
#       --output-dir ./stubs/DeepSeek-V4-Flash-Stub-SGLang \
#       --save-stub-weights
#
# This produces a directory containing real (random-init) safetensors that
# SGLang can load. SGLang has no random-init path of its own.
#
# For the REAL run, point TARGET_MODEL at the actual DeepSeek-V4-Flash weights
# directory and remove the smoke-test overrides at the bottom of this file.
# ============================================================================

set -euo pipefail

# ---- Paths ----------------------------------------------------------------
TARGET_MODEL=${TARGET_MODEL:-./stubs/DeepSeek-V4-Flash-Stub-SGLang}
TRAIN_DATA=${TRAIN_DATA:-/home/n84449292/m84379596/DFlash/Specforge_NPU/SpecForge/cache/dataset/sharegpt_train_filtered.jsonl}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/dsv4-flash-dflash-sglang-npu}

# ---- Devices --------------------------------------------------------------
NUM_NPUS=${NUM_NPUS:-8}
ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}

# ---- SpecForge-level parallelism -----------------------------------------
# TP_SIZE carves the world: TP=N for the SGLang target, DP=8/N for the draft.
# For DeepSeek-V4-Flash on 8 NPUs, TP=8 keeps the target on one parallel group
# and leaves the draft replicated (which is fine — the draft is tiny).
TP_SIZE=${TP_SIZE:-8}

# ---- SGLang-level kwargs (consumed by the NPU-aware patch in
#       dflash_target_model.py via environment variables) ------------------

##### @Moh_7596 — Attention backend, single source of truth.
# "dsv4"   → DeepseekV4AscendAttnBackend (sgl-kernel-npu V4 kernels). Use for V4 models.
# "ascend" → AscendAttnBackend (generic NPU attention). Use for non-V4 models.
# Override at the command line:   SPECFORGE_ATTENTION_BACKEND=ascend bash run_...sh
export SPECFORGE_ATTENTION_BACKEND=${SPECFORGE_ATTENTION_BACKEND:-dsv4}
################

# Leave plenty of room for the draft model + FSDP states. The default 0.88
# in stock SGLang is wrong for this co-residency setup.
export SPECFORGE_SGLANG_MEM_FRACTION_STATIC=${SPECFORGE_SGLANG_MEM_FRACTION_STATIC:-0.55}
# Expert parallelism size — typically == TP_SIZE for MoE.
export SPECFORGE_SGLANG_EP_SIZE=${SPECFORGE_SGLANG_EP_SIZE:-$TP_SIZE}
# Radix cache off for training-data prep (no benefit from prefix reuse).
export SPECFORGE_SGLANG_DISABLE_RADIX_CACHE=${SPECFORGE_SGLANG_DISABLE_RADIX_CACHE:-1}

# ---- DeepSeek-V4 Ascend fused-kernel env vars ------------------------------
# Required per SGLang issue #23598 to enable the V4-specific kernels on NPU.
# Comment out individual lines if your SGLang/CANN build doesn't yet ship a
# given kernel — SGLang will fall back to slower generic paths.
export USE_FUSED_COMPRESSOR=${USE_FUSED_COMPRESSOR:-1}
export LI_KV_DTYPE_INT8=${LI_KV_DTYPE_INT8:-1}
export USE_PA_DECODE=${USE_PA_DECODE:-1}
export USE_PA_PREFILL=${USE_PA_PREFILL:-1}
export USE_FUSED_HC_POST_ASCENDC=${USE_FUSED_HC_POST_ASCENDC:-1}
export USE_FUSED_HC_PRE_ASCENDC=${USE_FUSED_HC_PRE_ASCENDC:-1}
export USE_NPU_MOE_GATING_TOP_K=${USE_NPU_MOE_GATING_TOP_K:-1}
export USE_FUSED_TRANSPOSE_BATCHMATMUL=${USE_FUSED_TRANSPOSE_BATCHMATMUL:-1}
export USE_ROPE_PARTIAL_IN_PLACE_ASCENDC=${USE_ROPE_PARTIAL_IN_PLACE_ASCENDC:-1}

# ---- NPU runtime env ------------------------------------------------------
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-2}
export ACLNN_CACHE_LIMIT=${ACLNN_CACHE_LIMIT:-100000}
export NPU_ASD_ENABLE=${NPU_ASD_ENABLE:-0}
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}

# ---- Hyperparams ---------------------------------------------------------
# Sanity defaults — small, fast. Override for real runs.
BATCH_SIZE=${BATCH_SIZE:-1}
ACCUMULATION_STEPS=${ACCUMULATION_STEPS:-1}
MAX_LENGTH=${MAX_LENGTH:-512}
NUM_EPOCHS=${NUM_EPOCHS:-1}
LR=${LR:-6e-4}
NUM_ANCHORS=${NUM_ANCHORS:-16}
BLOCK_SIZE=${BLOCK_SIZE:-16}
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7.0}
WARMUP_RATIO=${WARMUP_RATIO:-0.04}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}

LOG_INTERVAL=${LOG_INTERVAL:-1}
SAVE_INTERVAL=${SAVE_INTERVAL:-999999}
REPORT_TO=${REPORT_TO:-tensorboard}

# ---- Auto-locate SpecForge repo root --------------------------------------
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &> /dev/null && pwd)
ROOT_DIR=$(dirname "$(dirname "$SCRIPT_DIR")")
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/deepseek-v4-flash-dflash.json}

cat <<EOF
============ SpecForge DFlash on SGLang (DeepSeek-V4-Flash, NPU) ============
ROOT_DIR                       : $ROOT_DIR
TARGET_MODEL                   : $TARGET_MODEL
DRAFT_CONFIG                   : $DRAFT_CONFIG
TRAIN_DATA                     : $TRAIN_DATA
OUTPUT_DIR                     : $OUTPUT_DIR
NUM_NPUS / TP_SIZE / EP_SIZE   : $NUM_NPUS / $TP_SIZE / $SPECFORGE_SGLANG_EP_SIZE
SPECFORGE_ATTENTION_BACKEND    : $SPECFORGE_ATTENTION_BACKEND
mem_fraction_static            : $SPECFORGE_SGLANG_MEM_FRACTION_STATIC
ASCEND_RT_VISIBLE_DEVICES      : $ASCEND_RT_VISIBLE_DEVICES
=============================================================================
EOF

# ---- Pre-flight checks ----------------------------------------------------
[[ -f "$DRAFT_CONFIG" ]] || { echo "ERROR: DRAFT_CONFIG not found: $DRAFT_CONFIG" >&2; exit 1; }
[[ -d "$TARGET_MODEL" ]] || { echo "ERROR: TARGET_MODEL dir not found: $TARGET_MODEL" >&2; exit 1; }
[[ -f "$TRAIN_DATA" ]]   || { echo "ERROR: TRAIN_DATA not found: $TRAIN_DATA" >&2; exit 1; }

# SGLang needs real weight files — verify they're there.
if ! compgen -G "$TARGET_MODEL/*.safetensors" > /dev/null; then
    echo "ERROR: $TARGET_MODEL contains no *.safetensors files." >&2
    echo "       SGLang cannot load this directory. Either:" >&2
    echo "         a) re-run prepare_dsv4_stub_target.py WITH --save-stub-weights, or" >&2
    echo "         b) point TARGET_MODEL at a real weight directory." >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"
cd "$ROOT_DIR"

MASTER_ADDR_LOCAL=${MASTER_ADDR_LOCAL:-127.0.0.1}
MASTER_PORT_LOCAL=${MASTER_PORT_LOCAL:-29534}

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
    --trust-remote-code \
    --tp-size "$TP_SIZE" \
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