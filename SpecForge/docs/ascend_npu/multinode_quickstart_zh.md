# 多机训练 Quick Start（双机 16 NPU）

> 你已经验证过两台机器都能单机跑通。这份文档把双机启动的命令按"复制粘贴顺序"整理好，
> 配套 `multinode_preflight.sh`（pre-flight 诊断）+ `run_qwen3_8b_dflash_npu_multinode.sh`
> （正式 launcher）。

---

## Phase 1: 在 **每一台机器** 上分别跑诊断

```bash
cd /home/a00652497/2026/SpecForge
git pull
bash docs/ascend_npu/multinode_preflight.sh
```

把两台机器的输出都贴回来。重点看：
- Section 1 列的 IPv4 地址（找出每台机器的真实可达 IP）
- Section 2 列的 NIC 名（确认两台同名 NIC，通常是 `enp*` 或 `eth*`）
- Section 3 NPU 是否干净
- Section 5 specforge import 是否 OK

---

## Phase 2: 验证两机互通

挑出 **节点 0** 的 IP（`<NODE0_IP>`）和共同 NIC 名（`<NIC>`），然后**在节点 1 上**跑：

```bash
# 替换 <NODE0_IP> 为节点 0 的真实 IP
ping -c 3 <NODE0_IP>

# 测试默认 rendezvous 端口 29533 能不能通
# 节点 0 上先开监听：
#     nc -l 29533
# 节点 1 上 nc 测连接：
nc -zv <NODE0_IP> 29533
```

**两条都通**才能进入 Phase 3。不通见文档末尾"端口被防火墙挡"的处置。

---

## Phase 3: 启动训练

### 节点 0 上

```bash
cd /home/a00652497/2026/SpecForge

MASTER_ADDR=<NODE0_IP> \
NNODES=2 \
NODE_RANK=0 \
HCCL_SOCKET_IFNAME=<NIC> \
bash docs/ascend_npu/run_qwen3_8b_dflash_npu_multinode.sh
```

启动后会卡在 `init_process_group` 等节点 1 来 rendezvous——**此时立刻去节点 1**。

### 节点 1 上（紧接着启动）

```bash
cd /home/a00652497/2026/SpecForge

MASTER_ADDR=<NODE0_IP> \
NNODES=2 \
NODE_RANK=1 \
HCCL_SOCKET_IFNAME=<NIC> \
bash docs/ascend_npu/run_qwen3_8b_dflash_npu_multinode.sh
```

注意：**`MASTER_ADDR` 在两个节点上都填节点 0 的 IP**（不是各自的本机 IP）。

---

## 期望成功的 log 关键行

两台机器各打印自己脚本头部的 banner，然后约 30 秒后看到（节点 0 视角）：

```
device mesh: DeviceMesh((dp=16, tp=1), device: 'npu', stride: ...)
```

`dp=16`（不是 8）证明跨节点 16 张卡进了同一个 DP 组，HCCL 联通。

接着进入训练循环，第一个 step 比单机慢一些（首次跨节点 ReduceScatter / AllGather）：

```
Train - Step 1 [...], Loss: <数字>, Acc: <数字>
```

---

## 卡点排查

### 节点 0 永远等 rendezvous（卡 30+ 秒不动）

99% 是 NIC 名错或两节点不一致。Phase 1 的诊断输出贴出来，我帮你确认。

### 端口被防火墙挡

Phase 2 的 `nc -zv` 不通但 `ping` 通——典型防火墙问题。两条路：

1. **换端口**：`MASTER_PORT=12345 ...` 找一个开放的（咨询管理员或自己试）。
2. **临时开端口**（root 才行）：
   ```bash
   sudo iptables -I INPUT -p tcp --dport 29533 -j ACCEPT
   ```

### `HCCL connection failed` 但 TCPStore 已连上

第一次跨节点 HCCL 建链慢，default timeout 可能不够。脚本默认已经把
`HCCL_CONNECT_TIMEOUT=3600`，应该足够。如果还是 timeout，把
`HCCL_CONNECT_TIMEOUT=7200` 试试。

### `RuntimeError: NPU out of memory`（其中某节点 OOM）

跟单机时同样的问题——某台机器的某张卡被别人占了。
回到节点上 `npu-smi info` 看占用，`pkill` 僵尸或换 `ASCEND_RT_VISIBLE_DEVICES` 避开被占卡。

---

## 你的反馈节奏

按这三阶段做，**每一阶段做完贴一次输出回来**：
1. Phase 1 双机诊断 → 我确认 MASTER_ADDR / NIC 选择
2. Phase 2 互通测试 → 确认网络通
3. Phase 3 启动后第一个 step 的 log（前 50 行 + step 1 的 loss）→ 确认 dp=16 + 训练正确

任一阶段报错 paste 报错原文，下一步定位。
