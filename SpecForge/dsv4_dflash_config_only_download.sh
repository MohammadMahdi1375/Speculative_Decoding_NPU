#!/usr/bin/env bash
# =============================================================================
# Download ONLY config.json + tokenizer from DeepSeek-V4-Flash
# No model weights — SGLang's --load-format dummy handles the rest
# =============================================================================
set -euo pipefail

MODEL_ID=${1:-deepseek-ai/DeepSeek-V4-Flash}
OUTPUT_DIR=${2:-/share/canada_group_folder/ckpt/DeepSeek-V4-Flash-config-only}

export HF_HOME=${HF_HOME:-/home/a00652497/m84379596/Huggingface}

echo "[INFO] Downloading config + tokenizer ONLY from ${MODEL_ID}"
echo "[INFO] Output: ${OUTPUT_DIR}"
echo "[INFO] This should be a few MB, NOT 150GB"

# huggingface-cli download with --include to grab only what SGLang needs:
#   config.json           — model architecture (required for --load-format dummy)
#   tokenizer*            — tokenizer files
#   generation_config*    — generation defaults
#   *.py                  — any custom modeling code (trust_remote_code)
#   encoding_dsv4/        — DSV4's custom encoding module

huggingface-cli download "${MODEL_ID}" \
    --include "config.json" \
              "tokenizer*" \
              "generation_config*" \
              "*.py" \
              "encoding_dsv4/*" \
    --local-dir "${OUTPUT_DIR}" \
    --local-dir-use-symlinks False

echo ""
echo "============================================"
echo "[DONE] Config-only checkpoint at: ${OUTPUT_DIR}"
echo ""
echo "Contents:"
find "${OUTPUT_DIR}" -type f | head -30
echo ""
echo "Total size:"
du -sh "${OUTPUT_DIR}"
echo ""
echo "[NEXT] Use this as TARGET_MODEL with --load-format dummy"
echo "============================================"