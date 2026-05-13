#!/bin/bash
# ============================================================================
# Wrapper for sgl-kernel-npu's build.sh that bypasses its hard-coded read of
# /etc/Ascend/ascend_cann_install.info.
#
# *** STATUS: legacy / fallback. ***
# The recommended install path now uses the fork branch
# `Sawyer117/sgl-kernel-npu#npu-install-stable`, which has the same fix
# baked into build.sh source. This wrapper is kept only for the case where
# you must build from upstream sgl-project/sgl-kernel-npu directly while
# PR #460 is still open. See docs/ascend_npu/installation.md Step 5.
#
# Background:
#   Upstream sgl-kernel-npu/build.sh:97-99 reads /etc/Ascend/ascend_cann_install.info
#   to find the CANN toolkit and unconditionally sources its set_env.sh — even
#   when the user has already sourced a different CANN install. On multi-user
#   hosts where /etc/Ascend/ascend_cann_install.info was written by another
#   user's root install, this picks the wrong (often inaccessible) CANN.
#
# This wrapper:
#   1. Derives the correct Toolkit_InstallPath from your already-sourced
#      ASCEND_HOME_PATH.
#   2. PATH-prepends a fake `cat` that intercepts only the read of
#      /etc/Ascend/ascend_cann_install.info and returns YOUR path.
#   3. Invokes the real build.sh.
#
# Source code is NOT modified. The shim only exists for the duration of this
# script (cleaned up via trap on EXIT).
#
# Usage:
#   # 0. Ensure CANN is sourced (sets ASCEND_HOME_PATH)
#   source /path/to/your/CANN/.../set_env.sh
#
#   # 1. cd to your sgl-kernel-npu clone
#   cd /path/to/sgl-kernel-npu
#
#   # 2. Run this wrapper, passing whatever build.sh args you want
#   bash /path/to/SpecForge/docs/ascend_npu/build_sgl_kernel_npu.sh -a kernels
#
# Override:
#   If auto-detection picks the wrong toolkit, set CANN_TOOLKIT_PATH explicitly:
#   CANN_TOOLKIT_PATH=/abs/path/to/toolkit/dir bash build_sgl_kernel_npu.sh -a kernels
#   (the dir must contain a set_env.sh)
# ============================================================================

set -euo pipefail

if [[ -z "${ASCEND_HOME_PATH:-}" ]]; then
    echo "Error: ASCEND_HOME_PATH is not set." >&2
    echo "Source your CANN's set_env.sh first, then re-run this script." >&2
    exit 1
fi

# Derive the toolkit-install root (the dir whose set_env.sh exists).
# ASCEND_HOME_PATH is typically:
#   /<...>/CANN/<ver>/ascend-toolkit/latest
# and set_env.sh is typically at one of:
#   /<...>/CANN/<ver>/ascend-toolkit/set_env.sh
#   /<...>/CANN/<ver>/set_env.sh
#   /<...>/CANN/<ver>/ascend-toolkit/latest/set_env.sh

TOOLKIT_INSTALL_PATH="${CANN_TOOLKIT_PATH:-}"
if [[ -z "$TOOLKIT_INSTALL_PATH" ]]; then
    candidates=(
        "${ASCEND_HOME_PATH%/latest}"                  # .../ascend-toolkit
        "${ASCEND_HOME_PATH%/ascend-toolkit/latest}"   # .../<ver>
        "$ASCEND_HOME_PATH"                            # .../latest
    )
    for c in "${candidates[@]}"; do
        if [[ -f "$c/set_env.sh" ]]; then
            TOOLKIT_INSTALL_PATH="$c"
            break
        fi
    done
fi

if [[ -z "$TOOLKIT_INSTALL_PATH" ]] || [[ ! -f "$TOOLKIT_INSTALL_PATH/set_env.sh" ]]; then
    echo "Error: could not locate a directory containing set_env.sh near" >&2
    echo "       ASCEND_HOME_PATH=$ASCEND_HOME_PATH" >&2
    echo "Try setting CANN_TOOLKIT_PATH explicitly:" >&2
    echo "  CANN_TOOLKIT_PATH=/abs/path/to/toolkit/dir bash $0 $*" >&2
    exit 1
fi

echo "[shim] Will tell build.sh: Toolkit_InstallPath=$TOOLKIT_INSTALL_PATH"

# Build the cat shim in a private temp dir
SHIM_DIR=$(mktemp -d -t sgl_kernel_npu_shim.XXXXXX)
trap 'rm -rf "$SHIM_DIR"' EXIT

cat > "$SHIM_DIR/cat" <<EOF
#!/bin/bash
# Intercepts ONLY the build.sh read of /etc/Ascend/ascend_cann_install.info.
# All other 'cat' invocations are passed through to /bin/cat.
if [[ "\${1:-}" == "/etc/Ascend/ascend_cann_install.info" ]]; then
    echo "Toolkit_InstallPath=$TOOLKIT_INSTALL_PATH"
else
    exec /bin/cat "\$@"
fi
EOF
chmod +x "$SHIM_DIR/cat"

# Hand off to the real build.sh with the shim's cat at the front of PATH.
echo "[shim] Invoking: bash build.sh $*"
PATH="$SHIM_DIR:$PATH" bash build.sh "$@"
