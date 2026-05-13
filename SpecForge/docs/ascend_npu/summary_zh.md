# SpecForge —— DFlash 与 Ascend NPU 支持情况速览

> 范围：本文档总结 (1) SpecForge 当前对 DFlash 训练 pipeline 的支持情况、
> (2) DFlash 路径在 Ascend NPU 上的支持现状、
> (3) 在 Ascend NPU 上部署 DFlash 训练所需的环境、样例启动脚本、主函数入口与调用栈。

---

## 1. SpecForge 概览

SpecForge 是 SGLang 团队的「**投机解码 (speculative decoding) 草稿模型训练框架**」，
训出来的 draft 模型可以直接接到 SGLang 推理框架里。仓库内已经实现两条训练 pipeline：

| Pipeline | 入口脚本 | 核心 wrapper | 草稿模型 | Target 后端 |
|---|---|---|---|---|
| EAGLE3 | `scripts/train_eagle3.py` | `specforge/core/eagle3.py` | `specforge/modeling/draft/llama3_eagle.py` | `eagle3_target_model.py` (HF / SGLang) |
| **DFlash** | `scripts/train_dflash.py` | `specforge/core/dflash.py` | `specforge/modeling/draft/dflash.py` | `dflash_target_model.py` (HF / SGLang) |

DFlash 仅支持 online 训练（target 隐状态由 target 模型现场前向得到）。
没有 offline 路径；只有 EAGLE3 才有 offline 分支。

---

## 2. DFlash 支持度

DFlash 在仓库中已是一等公民，所有要素齐备：

- **训练脚本**：`scripts/train_dflash.py`（入口 `main()` 在第 561 行）。
- **Online 训练 wrapper**：`specforge/core/dflash.py` 中的 `OnlineDFlashModel`
  （包含 anchor 采样、noise embedding 构造、FlexAttention/SDPA block mask、
  以及块级 CE loss 与可选的 `loss_decay_gamma` 加权）。
- **草稿模型**：`specforge/modeling/draft/dflash.py` 中的 `DFlashDraftModel`，
  基于 Qwen3 组件 (`Qwen3DFlashAttention`, `Qwen3DFlashDecoderLayer`) 构建。
- **Target backend**：`specforge/modeling/target/dflash_target_model.py` 提供
  `HFDFlashTargetModel` 与 `SGLangDFlashTargetModel`，由
  `get_dflash_target_model(backend="hf" | "sglang")` 分发。
- **注意力后端**：`--attention-backend {eager, sdpa, flex_attention}` 三选一，
  默认 `flex_attention`。
- **参考 draft 配置**：`configs/qwen3-8b-dflash.json`、
  `configs/qwen3.5-35b-a3b-dflash.json`、`configs/longcat-flash-dflash.json`。
- **端到端示例脚本**：`examples/run_qwen3_8b_dflash_online.sh`、
  `examples/run_qwen3.5_35b_a3b_dflash_online.sh`、
  `examples/run_longcat_flash_dflash_online.sh`。
- **单元测试**：`tests/test_utils/test_dflash_mask.py`。
- **文档**：`docs/` 下目前**没有**专门讲 DFlash 的文档，是当前的一个空白点。

---

## 3. Ascend NPU 支持度（针对 DFlash 场景）

**结论：SpecForge 自身目前不支持 Ascend NPU，DFlash 在 NPU 上无法开箱即用。** 证据：

1. `pyproject.toml` 钉的是 `torch==2.9.1` + `sglang==0.5.9`，没有任何
   `torch_npu` / CANN 相关依赖；只有 `requirements-rocm.txt` 提供了非 CUDA 的路线。
2. `specforge/distributed.py:75` 硬编码 `dist.init_process_group(backend="nccl")`、
   `torch.cuda.set_device(...)`，并以 `device_type="cuda"` 创建 device mesh。
3. `scripts/train_dflash.py` 中 `.cuda()` / `device="cuda"` 共有 7 处硬编码
   （第 163、194、426、499–505 行）。
4. 全仓 grep `torch_npu | torch.npu | ascend` 共 **0** 条命中。
5. **唯一一处涉及 NPU 的代码**位于
   `specforge/modeling/target/sglang_backend/model_runner.py:78–79`：
   ```python
   elif self.device == "npu":
       backend = "hccl"
   ```
   这条分支继承自 SGLang 的 `ModelRunner`，SpecForge 仅覆写了
   `init_torch_distributed`。它只在 `--target-model-backend sglang` 时被走到，
   并不代表整个训练流程能跑通 NPU——draft 模型、FSDP、优化器、张量搬运
   依然全部走 `torch.cuda.*`。

也就是说：**在 Ascend NPU 上部署 DFlash 需要做实际的适配工作**。

---

## 4. 在 Ascend NPU 上部署 DFlash

### 4.1 环境

- **硬件 / 驱动**：Atlas A2 / A3 训练卡 + 配套 CANN（建议 ≥7.0）+ HCCL。
- **Python**：3.11（`pyproject.toml` 强制要求）。
- **PyTorch**：装**昇腾官方 torch + torch_npu**（建议先用 2.4 / 2.5 系列，更稳）。
  注意这与 `pyproject.toml` 钉的 `torch==2.9.1` 冲突——要么放宽版本约束，
  要么使用社区最新 wheel。
- **纯 Python 依赖**：`transformers==4.57.1`、`accelerate`、`datasets`、`wandb`、
  `tqdm`、`tensorboard` 直接 pip 装即可。
- **暂时不要装 `sglang==0.5.9`**：其 NPU 支持成熟度有限，且 SpecForge 的
  `sglang_backend/model_runner.py` 与该版本强绑定。先用 `--target-model-backend hf`
  绕过去。
- **注意力后端**：`flash-attn` 在 NPU 上不可用；FlexAttention 同样有风险。
  推荐用 `--attention-backend sdpa`（torch_npu 上 SDPA 有 fallback 实现）。
- **`yunchang`（序列并行）**：只在 EAGLE3 多机长上下文时需要。单机 DFlash 用不到，
  但 `specforge/distributed.py` 在 import 阶段就会触发 yunchang 的全局对象，
  必要时需要 stub 掉。

### 4.2 最小代码改动

1. `specforge/distributed.py:66–120`
   - `nccl` → `hccl`
   - `torch.cuda.*` → `torch.npu.*`
   - `init_device_mesh("cuda", ...)` → `("npu", ...)`
   - `DeviceMesh.from_group(..., device_type="cuda")` → `"npu"`
   - 建议加一个全局常量
     `DEVICE = "npu" if torch_npu_available else "cuda"` 在各处统一引用。

2. `scripts/train_dflash.py:163, 194, 426, 499–505`
   - 把 `.cuda()` / `device="cuda"` 改成 `.to(device)` / `device=device`。

3. `specforge/modeling/target/dflash_target_model.py:101`
   - `torch.cuda.current_device()` → `torch.npu.current_device()`
     （仅在仍坚持使用 SGLang backend 时才需要）。

4. 在程序入口加 `import torch_npu`，让 PyTorch 注册 NPU 后端。

只动这四处即可把 `--target-model-backend hf` 路径跑起来。

### 4.3 样例启动脚本（适配后）

```bash
# 假设 §4.2 中的改动已经打入仓库
export ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HCCL_TIMEOUT=1800
export TORCHINDUCTOR_CACHE_DIR=$ROOT_DIR/cache/compiled_kernels
export SPECFORGE_DATA_NUM_PROC=32

torchrun --standalone --nproc_per_node 8 \
    $ROOT_DIR/scripts/train_dflash.py \
    --target-model-path Qwen/Qwen3-8B \
    --draft-config-path $ROOT_DIR/configs/qwen3-8b-dflash.json \
    --train-data-path $ROOT_DIR/cache/dataset/perfectblend_qwen3-8b_regen.jsonl \
    --output-dir $ROOT_DIR/outputs/qwen3-8b-dflash-npu \
    --target-model-backend hf \
    --attention-backend sdpa \
    --num-epochs 6 --batch-size 2 --max-length 3072 \
    --learning-rate 6e-4 --warmup-ratio 0.04 --max-grad-norm 1.0 \
    --block-size 16 --num-anchors 512 --loss-decay-gamma 7.0 \
    --chat-template qwen --report-to tensorboard
```

---

## 5. 主函数入口与调用栈（DFlash online）

**入口**：`scripts/train_dflash.py:561` 的 `if __name__ == "__main__": main()`

```
main()                                                        # train_dflash.py:342
├── parse_args()                                              # train_dflash.py:39
├── set_seed(seed)
├── init_distributed(timeout, tp_size)                        # specforge/distributed.py:66
│     dist.init_process_group("nccl", ...)                    【NPU: 改为 "hccl"】
│     torch.cuda.set_device(...)                              【NPU: 改为 torch.npu】
│     init_device_mesh("cuda", (dp, tp))                      【NPU: 改为 "npu"】
│     set_seq_parallel_pg(...)                                # yunchang
├── build_models(args)                                        # train_dflash.py:149
│     ├── get_dflash_target_model(backend=hf|sglang)          # dflash_target_model.py:290
│     │     ├── HFDFlashTargetModel.from_pretrained()         # AutoModelForCausalLM
│     │     └── SGLangDFlashTargetModel.from_pretrained()     # SGLangRunner (sglang 0.5.9)
│     │           └── init_torch_distributed()                # 唯一带 npu/hccl 分支的位置
│     ├── DFlashDraftModel(draft_config).cuda()               # modeling/draft/dflash.py:212
│     └── target_model.set_capture_layers(target_layer_ids)
├── （可选）加载 checkpoint
├── AutoTokenizer.from_pretrained(...)
├── build_dataloader(args, tokenizer)                         # train_dflash.py:210
│     └── build_eagle3_dataset() + prepare_dp_dataloaders()
├── TargetEmbeddingsAndHead.from_pretrained(...)              # modeling/target/target_utils.py
├── OnlineDFlashModel(draft, lm_head, embed, ...)             # core/dflash.py:96
├── FSDP(dflash_model,
│       mixed_precision=bf16,
│       sharding=SHARD_GRAD_OP,
│       use_orig_params=True)
├── BF16Optimizer(draft_model, ...)                           # specforge/optimizer.py:7
├── create_tracker(args, output_dir)                          # specforge/tracker.py
└── 训练主循环 (epoch × step)：
        target_model.generate_dflash_data(ids, mask, lm)      # 拿到 hidden_states
        ├── HF backend: target_model.forward(output_hidden_states=True)
        └── SGLang backend: SGLangDFlashTargetModel._extend(reqs)
              └── ScheduleBatch + ForwardBatch + model_runner.forward()
        loss, acc = OnlineDFlashModel(input_ids, hidden_states, loss_mask)
        ├── _sample_anchor_positions()                        # 随机选 anchor
        ├── _create_noise_embed()                             # mask token 构 noise
        ├── create_dflash_block_mask()                        # FlexAttention / SDPA 掩码
        ├── DFlashDraftModel.forward(noise_emb, target_hidden, ...)
        │     └── Qwen3DFlashDecoderLayer × N → norm
        ├── lm_head + 块级 CE loss（可选 loss_decay_gamma 加权）
        └── 返回 (loss, acc)
        loss.backward(); optimizer.step()
        save_checkpoint() / record_metrics()
└── destroy_distributed()
```

---

## 6. 一句话总结

- **DFlash** 在 SpecForge 中已经是完整、可训练的训练 pipeline——入口脚本、wrapper、
  草稿模型、target backend、配置文件与示例启动脚本一应俱全。
- **Ascend NPU** 目前**并未**得到官方支持。`sglang_backend/model_runner.py` 中
  那条 `npu/hccl` 分支只是从 SGLang 继承下来的，没有覆盖整个训练栈。
- 想在 Ascend NPU 上跑 DFlash，按 §4.2 做一次轻量的 device 抽象 patch，
  并优先选择 `--target-model-backend hf` + `--attention-backend sdpa` 这条最薄的路径，
  再用 §4.3 的样例脚本启动训练。
