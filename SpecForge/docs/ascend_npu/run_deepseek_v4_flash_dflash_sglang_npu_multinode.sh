#!/usr/bin/env bash
# =============================================================================
# DFlash training — DeepSeek-V4-Flash target, SGLang backend, multinode, Ascend NPU
# Based on the working Qwen3 multinode script, adapted for V4-Flash + dsv4 NPU port.
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd)"
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ---- Multinode rendezvous (export by caller) --------------------------------
: "${MASTER_ADDR:?MASTER_ADDR must be set (IP of NODE_RANK=0)}"
: "${MASTER_PORT:=29502}"
: "${NNODES:=2}"
: "${NODE_RANK:=0}"
: "${HCCL_SOCKET_IFNAME:?HCCL_SOCKET_IFNAME must be set to the inter-node NIC}"
export MASTER_ADDR MASTER_PORT NNODES NODE_RANK HCCL_SOCKET_IFNAME

# Also needed for c10d store / Gloo rendezvous (caller may set; default to HCCL one)
export TP_SOCKET_IFNAME=${TP_SOCKET_IFNAME:-$HCCL_SOCKET_IFNAME}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-$HCCL_SOCKET_IFNAME}

# ---- Device + SGLang knobs --------------------------------------------------
export SPECFORGE_DEVICE=${SPECFORGE_DEVICE:-npu}
export SPECFORGE_DIST_BACKEND=${SPECFORGE_DIST_BACKEND:-hccl}

NUM_NPUS=${NUM_NPUS:-8}
TP_SIZE=${TP_SIZE:-8}
SEED=${SEED:-42}
BATCH_SIZE=${BATCH_SIZE:-1}
MAX_LENGTH=${MAX_LENGTH:-2048}
ATTENTION_BACKEND=${ATTENTION_BACKEND:-sdpa}

export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((NUM_NPUS-1)))}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}
# V4-Flash is huge; allocate more of HBM to SGLang model + KV cache
export SGLANG_MEM_FRACTION_STATIC=${SGLANG_MEM_FRACTION_STATIC:-0.55}
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=${SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN:-1}

# ---- V4 NPU port required env vars ------------------------------------------
export SPECFORGE_ATTENTION_BACKEND=${SPECFORGE_ATTENTION_BACKEND:-dsv4}
export SGLANG_OPT_USE_TILELANG_MHC_PRE=${SGLANG_OPT_USE_TILELANG_MHC_PRE:-False}
export SGLANG_OPT_USE_TILELANG_MHC_POST=${SGLANG_OPT_USE_TILELANG_MHC_POST:-False}
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-False}
export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=${SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT:-True}
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=${SGLANG_OPT_BF16_FP32_GEMM_ALGO:-torch}
export SPECFORGE_SGLANG_EP_SIZE=${SPECFORGE_SGLANG_EP_SIZE:-16}

# ---- HCCL inter-node tuning -------------------------------------------------
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1200}
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_OP_BASE_FFTS_MODE_ENABLE=${HCCL_OP_BASE_FFTS_MODE_ENABLE:-TRUE}

# ---- Paths ------------------------------------------------------------------
TARGET_MODEL=${TARGET_MODEL:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-bf16}
TRAIN_DATA=${TRAIN_DATA:?TRAIN_DATA must be set}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/dsv4-flash-dflash-sglang-multinode}
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/deepseek-v4-flash-dflash.json}

# ---- Hyperparams ------------------------------------------------------------
NUM_EPOCHS=${NUM_EPOCHS:-2}
LEARNING_RATE=${LEARNING_RATE:-3e-4}
WARMUP_RATIO=${WARMUP_RATIO:-0.04}
MAX_GRAD_NORM=${MAX_GRAD_NORM:-1.0}
NUM_ANCHORS=${NUM_ANCHORS:-256}
LOSS_DECAY_GAMMA=${LOSS_DECAY_GAMMA:-7.0}
LOG_INTERVAL=${LOG_INTERVAL:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
CHAT_TEMPLATE=${CHAT_TEMPLATE:-deepseek}

# ---- HF cache ---------------------------------------------------------------
export HF_HOME=${HF_HOME:-/share/canada_group_folder/hf_cache/}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HUB_CACHE}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}
export HF_HUB_DISABLE_TELEMETRY=1

# ---- WandB ------------------------------------------------------------------
# export WANDB_MODE=${WANDB_MODE:-offline}
# export WANDB_PROJECT=${WANDB_PROJECT:-specforge-dsv4-flash-dflash}
# export WANDB_NAME=${WANDB_NAME:-dsv4-flash-dflash-sglang-2node}
export TENSORBOARD_LOG_DIR=${TENSORBOARD_LOG_DIR:-$OUTPUT_DIR/tensorboard}

# ---- Banner -----------------------------------------------------------------
WORLD_SIZE=$((NNODES * NUM_NPUS))
cat <<EOF
================================================================================
[INFO] SpecForge DFlash MULTINODE — DeepSeek-V4-Flash (NPU, SGLang backend)
[INFO] NNODES=$NNODES NODE_RANK=$NODE_RANK NUM_NPUS=$NUM_NPUS WORLD_SIZE=$WORLD_SIZE
[INFO] MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT
[INFO] HCCL_SOCKET_IFNAME=$HCCL_SOCKET_IFNAME
[INFO] ASCEND_RT_VISIBLE_DEVICES=$ASCEND_RT_VISIBLE_DEVICES
[INFO] TP_SIZE=$TP_SIZE  EP_SIZE=$SPECFORGE_SGLANG_EP_SIZE
[INFO] SGLANG_MEM_FRACTION_STATIC=$SGLANG_MEM_FRACTION_STATIC
[INFO] SPECFORGE_ATTENTION_BACKEND=$SPECFORGE_ATTENTION_BACKEND
[INFO] TARGET_MODEL=$TARGET_MODEL
[INFO] TRAIN_DATA=$TRAIN_DATA
[INFO] DRAFT_CONFIG=$DRAFT_CONFIG
[INFO] OUTPUT_DIR=$OUTPUT_DIR
[INFO] BATCH_SIZE=$BATCH_SIZE MAX_LENGTH=$MAX_LENGTH CHAT_TEMPLATE=$CHAT_TEMPLATE
================================================================================
EOF

mkdir -p "$OUTPUT_DIR"

# ---- Launch -----------------------------------------------------------------
torchrun \
  --nnodes="$NNODES" \
  --node_rank="$NODE_RANK" \
  --nproc_per_node="$NUM_NPUS" \
  --master_addr="$MASTER_ADDR" \
  --master_port="$MASTER_PORT" \
  "$ROOT_DIR/scripts/train_dflash.py" \
    --target-model-path "$TARGET_MODEL" \
    --target-model-backend sglang \
    --trust-remote-code \
    --tp-size "$TP_SIZE" \
    --sglang-ep-size "${SPECFORGE_SGLANG_EP_SIZE:-1}" \
    --seed "$SEED" \
    --sglang-mem-fraction-static "$SGLANG_MEM_FRACTION_STATIC" \
    --embedding-key "${EMBEDDING_KEY:-model.embed_tokens.weight}" \
    --lm-head-key "${LM_HEAD_KEY:-lm_head.weight}" \
    --draft-config-path "$DRAFT_CONFIG" \
    --train-data-path "$TRAIN_DATA" \
    --output-dir "$OUTPUT_DIR" \
    --num-epochs "$NUM_EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --warmup-ratio "$WARMUP_RATIO" \
    --max-grad-norm "$MAX_GRAD_NORM" \
    --max-length "$MAX_LENGTH" \
    --chat-template "$CHAT_TEMPLATE" \
    --attention-backend "$ATTENTION_BACKEND" \
    --num-anchors "$NUM_ANCHORS" \
    --loss-decay-gamma "$LOSS_DECAY_GAMMA" \
    --log-interval "$LOG_INTERVAL" \
    --save-interval "$SAVE_INTERVAL" \
    --resume \
    --report-to tensorboard
    # --report-to wandb \
    # --wandb-project "$WANDB_PROJECT" \
    # --wandb-name "$WANDB_NAME"
