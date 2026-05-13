#!/bin/bash
# ============================================================================
# Multi-node pre-flight diagnostic. Run this on EACH NODE before launching
# multi-node training. Prints everything needed to fill in the launcher
# env vars (MASTER_ADDR, HCCL_SOCKET_IFNAME) and verifies the host is sane.
#
# Usage:
#   bash docs/ascend_npu/multinode_preflight.sh
#
# Paste the output back to discuss the right MASTER_ADDR / NIC choice.
# ============================================================================

set -uo pipefail

c_g='\e[1;32m'; c_y='\e[1;33m'; c_r='\e[1;31m'; c_e='\e[0m'
ok()   { echo -e "${c_g}[ok]${c_e}   $*"; }
warn() { echo -e "${c_y}[warn]${c_e} $*"; }
err()  { echo -e "${c_r}[err]${c_e}  $*"; }
sec()  { echo; echo -e "${c_y}=== $* ===${c_e}"; }

sec "0. Host identity"
echo "  hostname              : $(hostname)"
echo "  hostname resolves to  : $(getent hosts "$(hostname)" 2>/dev/null || echo '<not in /etc/hosts>')"
echo "  whoami                : $(whoami)"
echo "  date / kernel         : $(date -Is) / $(uname -r)"

sec "1. Routable IPv4 addresses (candidates for MASTER_ADDR)"
ip -o -4 addr show scope global up | awk '{print "  " $2 "  " $4}' \
    | sed 's|/[0-9]*||'

sec "2. UP NICs with link state (candidates for HCCL_SOCKET_IFNAME)"
ip -o link show up | awk -F': ' '$2 != "lo" {split($2,a,"@"); print "  " a[1]}' \
    | while read -r nic; do
        ipv4=$(ip -4 addr show "$nic" | awk '/inet /{print $2}' | head -1)
        echo "  $nic  ${ipv4:-<no ipv4>}"
    done

sec "3. NPU availability"
if command -v npu-smi >/dev/null 2>&1; then
    npu-smi info | head -40
else
    err "npu-smi not on PATH — source CANN env first?"
fi

sec "4. Conda env / Python / SpecForge state"
echo "  CONDA_PREFIX                      : ${CONDA_PREFIX:-<unset>}"
echo "  python                            : $(which python 2>/dev/null || echo '<not on PATH>')"
echo "  ASCEND_HOME_PATH                  : ${ASCEND_HOME_PATH:-<unset; source CANN!>}"
SPECFORGE_DIR=${SPECFORGE_DIR:-/home/$(whoami)/2026/SpecForge}
if [[ -d "$SPECFORGE_DIR/.git" ]]; then
    echo "  SpecForge dir                     : $SPECFORGE_DIR"
    echo "  SpecForge branch / commit         : $(git -C "$SPECFORGE_DIR" branch --show-current) / $(git -C "$SPECFORGE_DIR" log -1 --format='%h %s')"
else
    err "SpecForge not at $SPECFORGE_DIR — set SPECFORGE_DIR=<path> and re-run"
fi

sec "5. Quick sanity: can torch + torch_npu + specforge import?"
if [[ -n "${CONDA_PREFIX:-}" ]]; then
    python -c "
import sys
try:
    import torch, torch_npu
    from yunchang.globals import HAS_NPU
    import specforge
    print('  torch                  :', torch.__version__)
    print('  torch_npu              :', torch_npu.__version__)
    print('  yunchang.HAS_NPU       :', HAS_NPU)
    print('  torch.npu.is_available :', torch.npu.is_available())
    print('  torch.npu.device_count :', torch.npu.device_count())
    print('  specforge import       : OK')
except Exception as e:
    print('  IMPORT FAILED:', type(e).__name__, e)
    sys.exit(1)
" || err "Import failed — check Step 6 of installation_zh.md"
else
    warn "skipped — activate conda env first"
fi

sec "6. Suggestions"
echo "  - Pick ONE address from Section 1 as MASTER_ADDR (the IP of node-0)."
echo "  - Pick the NIC from Section 2 whose IP matches; that becomes HCCL_SOCKET_IFNAME."
echo "  - Both nodes must use the SAME HCCL_SOCKET_IFNAME name."
echo "  - From the OTHER node, run:"
echo "        ping -c 3 <chosen MASTER_ADDR>"
echo "        nc -zv <chosen MASTER_ADDR> 29533"
echo "    Both must succeed before launching multi-node training."
