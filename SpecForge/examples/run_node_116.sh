cd /home/a00652497/m84379596/DFlash/SpecForge
mkdir -p logs

# ---- CANN + ATB setup FIRST (before any exports it might wipe) ----
export CANN_HOME=/home/canada_group_account/CANN/8.5.0.B030
set +u
source "$CANN_HOME/ascend-toolkit/set_env.sh"
[ -f "$CANN_HOME/nnal/asdsip/set_env.sh" ] && source "$CANN_HOME/nnal/asdsip/set_env.sh"
set -u
ABI=$(python -c "import torch; print(1 if torch.compiled_with_cxx11_abi() else 0)")
export LD_LIBRARY_PATH="$CANN_HOME/nnal/atb/8.5.0/atb/cxx_abi_${ABI}/lib:${LD_LIBRARY_PATH:-}"
python -c "import ctypes; ctypes.CDLL('libatb.so'); print('libatb OK')" || { echo "ATB load failed"; }

# ---- All exports AFTER CANN setup ----
# export SGLANG_OPT_USE_TILELANG_MHC_PRE=false
# export SGLANG_OPT_USE_TILELANG_MHC_POST=false
# export SGLANG_OPT_DEEPGEMM_HC_PRENORM=false
# export SGLANG_OPT_USE_OVERLAP_STORE_CACHE=false

export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=80.5.5.112
export MASTER_PORT=29500
export NNODES=2
export NODE_RANK=1
export HCCL_SOCKET_IFNAME=enp189s0f0
export GLOO_SOCKET_IFNAME=enp189s0f0
export HCCL_CONNECT_TIMEOUT=7200
export HCCL_EXEC_TIMEOUT=7200
export TRAIN_DATA=/share/canada_group_folder/dataset/perfectblend_train_10ksubset.jsonl
export TARGET_MODEL=/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-config-only
export HF_HOME=/home/a00652497/m84379596/Huggingface

# ---- Launch ----
TASK_QUEUE_ENABLE=1 \
NUM_NPUS=8 \
TP_SIZE=1 \
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
BATCH_SIZE=1 \
MAX_LENGTH=512 \
SGLANG_MEM_FRACTION_STATIC=0.10 \
LOAD_FORMAT=dummy \
SGLANG_JSON_MODEL_OVERRIDE='{"num_hidden_layers": 2, "n_routed_experts": 16}' \
NUM_EPOCHS=1 \
LOG_INTERVAL=10 \
ATTENTION_BACKEND=eager \
CHAT_TEMPLATE=deepseek-v3 \
OUTPUT_DIR=./outputs/dsv4flash-dflash-sanity-2layer-tp8-2nodes \
bash docs/ascend_npu/run_dsv4flash_dflash_npu_sglang_multinode.sh \
  2>&1 | tee logs/dsv4flash_2node_rank1_$(date +%Y%m%d_%H%M%S).log