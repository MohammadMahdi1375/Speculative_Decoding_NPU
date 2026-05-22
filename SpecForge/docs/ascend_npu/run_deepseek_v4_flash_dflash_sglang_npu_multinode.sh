#!/usr/bin/env bash
# =============================================================================
# SpecForge DFlash MULTINODE launcher for Ascend NPU.
# Works for the STUB V4 model AND the full V4-Flash BF16 model.
#
# Run on EACH node simultaneously (open two terminals):
#   On node 0 (master, e.g. 80.5.5.108):
#     NODE_RANK=0 MASTER_ADDR=80.5.5.108 HCCL_SOCKET_IFNAME=<nic> \
#       TARGET_MODEL=<path> TRAIN_DATA=<path> bash <this_script>
#
#   On node 1 (worker, e.g. 80.5.5.109):
#     NODE_RANK=1 MASTER_ADDR=80.5.5.108 HCCL_SOCKET_IFNAME=<nic> \
#       TARGET_MODEL=<path> TRAIN_DATA=<path> bash <this_script>
#
# Find HCCL_SOCKET_IFNAME with:
#   ip -br addr show
# Pick the NIC whose IP matches MASTER_ADDR's subnet (e.g. eth0, enp*).
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT_DIR="$(cd -- "$SCRIPT_DIR/../.." &>/dev/null && pwd)"   # → SpecForge/
cd "$ROOT_DIR"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

# ───── Multinode topology ─────────────────────────────────────────────────────
: "${NODE_RANK:?Need NODE_RANK (0 for master, 1+ for workers)}"
: "${MASTER_ADDR:?Need MASTER_ADDR (IP of NODE_RANK=0)}"
: "${HCCL_SOCKET_IFNAME:?Need HCCL_SOCKET_IFNAME (NIC name; check 'ip -br addr show')}"
NNODES=${NNODES:-2}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_PORT=${MASTER_PORT:-29500}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
export MASTER_ADDR MASTER_PORT NNODES NODE_RANK HCCL_SOCKET_IFNAME

# ───── Paths ──────────────────────────────────────────────────────────────────
: "${TARGET_MODEL:?Need TARGET_MODEL (stub or full BF16 model dir)}"
: "${TRAIN_DATA:?Need TRAIN_DATA (sharegpt jsonl path; must be reachable on ALL nodes)}"
DRAFT_CONFIG=${DRAFT_CONFIG:-$ROOT_DIR/configs/deepseek-v4-flash-dflash.json}
OUTPUT_DIR=${OUTPUT_DIR:-./outputs/dsv4-multinode-$(date +%Y%m%d-%H%M%S)}

# ───── Parallelism ────────────────────────────────────────────────────────────
TP_SIZE=${TP_SIZE:-1}
EP_SIZE=${EP_SIZE:-1}
export SPECFORGE_SGLANG_EP_SIZE=$EP_SIZE

# ───── NPU runtime ────────────────────────────────────────────────────────────
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-$(seq -s, 0 $((NPROC_PER_NODE-1)))}
export ASCEND_LAUNCH_BLOCKING=${ASCEND_LAUNCH_BLOCKING:-0}
export TASK_QUEUE_ENABLE=${TASK_QUEUE_ENABLE:-1}

# ───── HCCL inter-node tuning ─────────────────────────────────────────────────
export HCCL_BUFFSIZE=${HCCL_BUFFSIZE:-200}                # MB per stream
export HCCL_CONNECT_TIMEOUT=${HCCL_CONNECT_TIMEOUT:-1200} # seconds
export HCCL_EXEC_TIMEOUT=${HCCL_EXEC_TIMEOUT:-3600}
export HCCL_OP_BASE_FFTS_MODE_ENABLE=${HCCL_OP_BASE_FFTS_MODE_ENABLE:-TRUE}
export HCCL_DETERMINISTIC=${HCCL_DETERMINISTIC:-false}
# If your cluster uses RDMA, also consider:
#   export HCCL_RDMA_TC=160; HCCL_RDMA_SL=4

# ───── V4 NPU port required env ───────────────────────────────────────────────
export SPECFORGE_ATTENTION_BACKEND=${SPECFORGE_ATTENTION_BACKEND:-dsv4}
export SGLANG_OPT_USE_TILELANG_MHC_PRE=${SGLANG_OPT_USE_TILELANG_MHC_PRE:-False}
export SGLANG_OPT_USE_TILELANG_MHC_POST=${SGLANG_OPT_USE_TILELANG_MHC_POST:-False}
export SGLANG_OPT_DEEPGEMM_HC_PRENORM=${SGLANG_OPT_DEEPGEMM_HC_PRENORM:-False}
export SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT=${SGLANG_NPU_DISABLE_ACL_FORMAT_WEIGHT:-True}
export SGLANG_OPT_BF16_FP32_GEMM_ALGO=${SGLANG_OPT_BF16_FP32_GEMM_ALGO:-torch}

# ───── Banner ─────────────────────────────────────────────────────────────────
cat <<EOF
================================================================================
SpecForge DFlash MULTINODE — DeepSeek-V4-Flash (NPU)
================================================================================
ROOT_DIR                    : $ROOT_DIR
NODE_RANK / NNODES          : $NODE_RANK / $NNODES
MASTER_ADDR : MASTER_PORT   : $MASTER_ADDR : $MASTER_PORT
NPROC_PER_NODE / WORLD_SIZE : $NPROC_PER_NODE / $WORLD_SIZE
HCCL_SOCKET_IFNAME          : $HCCL_SOCKET_IFNAME
ASCEND_RT_VISIBLE_DEVICES   : $ASCEND_RT_VISIBLE_DEVICES
TARGET_MODEL                : $TARGET_MODEL
DRAFT_CONFIG                : $DRAFT_CONFIG
TRAIN_DATA                  : $TRAIN_DATA
OUTPUT_DIR                  : $OUTPUT_DIR
TP_SIZE / EP_SIZE           : $TP_SIZE / $EP_SIZE
SPECFORGE_ATTENTION_BACKEND : $SPECFORGE_ATTENTION_BACKEND
================================================================================
EOF

mkdir -p "$OUTPUT_DIR"

# ───── Launch ─────────────────────────────────────────────────────────────────
torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    --node-rank=$NODE_RANK \
    --master-addr=$MASTER_ADDR \
    --master-port=$MASTER_PORT \
    "$ROOT_DIR/scripts/train_dflash.py" \
        --target-model-path "$TARGET_MODEL" \
        --draft-config "$DRAFT_CONFIG" \
        --train-data "$TRAIN_DATA" \
        --output-dir "$OUTPUT_DIR" \
        --tp-size "$TP_SIZE" \
    2>&1 | tee "$OUTPUT_DIR/train_node${NODE_RANK}.log"