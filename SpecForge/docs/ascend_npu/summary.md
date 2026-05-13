# SpecForge — DFlash & Ascend NPU Support Summary

> Scope: this note summarises (1) how SpecForge supports the DFlash training pipeline today,
> (2) the current state of Ascend NPU support for the DFlash path, and
> (3) what is required to deploy DFlash training on Ascend NPU — environment, sample
> launch script, main entry point, and call stack.

---

## 1. What SpecForge Provides

SpecForge is the SGLang team's framework for training **speculative-decoding draft models**
that can be served directly inside SGLang. Two training pipelines are implemented:

| Pipeline | Entry script | Core wrapper | Draft model | Target backend |
|---|---|---|---|---|
| EAGLE3 | `scripts/train_eagle3.py` | `specforge/core/eagle3.py` | `specforge/modeling/draft/llama3_eagle.py` | `eagle3_target_model.py` (HF / SGLang) |
| **DFlash** | `scripts/train_dflash.py` | `specforge/core/dflash.py` | `specforge/modeling/draft/dflash.py` | `dflash_target_model.py` (HF / SGLang) |

DFlash is online-only (target hidden states are produced on-the-fly by the target model).
There is no offline DFlash path; only EAGLE3 has an offline branch.

---

## 2. DFlash Support Status

DFlash is a first-class pipeline in the repository. Everything you need is already in tree:

- **Training script** — `scripts/train_dflash.py` (entry: `main()` at line 561).
- **Online wrapper** — `OnlineDFlashModel` in `specforge/core/dflash.py` (anchor sampling,
  noise-embedding construction, FlexAttention / SDPA block mask, block-wise CE loss with
  optional `loss_decay_gamma`).
- **Draft model** — `DFlashDraftModel` in `specforge/modeling/draft/dflash.py`, built on
  Qwen3 components (`Qwen3DFlashAttention`, `Qwen3DFlashDecoderLayer`).
- **Target backend** — `specforge/modeling/target/dflash_target_model.py` provides both
  `HFDFlashTargetModel` and `SGLangDFlashTargetModel`, dispatched by
  `get_dflash_target_model(backend="hf" | "sglang")`.
- **Attention backends** — selectable via `--attention-backend {eager, sdpa, flex_attention}`;
  default is `flex_attention`.
- **Reference draft configs** — `configs/qwen3-8b-dflash.json`,
  `configs/qwen3.5-35b-a3b-dflash.json`, `configs/longcat-flash-dflash.json`.
- **End-to-end examples** — `examples/run_qwen3_8b_dflash_online.sh`,
  `examples/run_qwen3.5_35b_a3b_dflash_online.sh`,
  `examples/run_longcat_flash_dflash_online.sh`.
- **Unit test** — `tests/test_utils/test_dflash_mask.py`.
- **Docs** — no DFlash-specific document under `docs/` yet (a gap worth noting).

---

## 3. Ascend NPU Support Status (for DFlash)

**Bottom line: SpecForge does not officially support Ascend NPU today.**
DFlash on NPU is not runnable out of the box. Evidence:

1. `pyproject.toml` pins `torch==2.9.1` + `sglang==0.5.9`; there is no `torch_npu` /
   CANN dependency. Only `requirements-rocm.txt` provides a non-CUDA path.
2. `specforge/distributed.py:75` hard-codes
   `dist.init_process_group(backend="nccl")`, `torch.cuda.set_device(...)`, and creates
   device meshes with `device_type="cuda"`.
3. `scripts/train_dflash.py` hard-codes `.cuda()` / `device="cuda"` in seven places
   (lines 163, 194, 426, 499–505).
4. A repo-wide grep for `torch_npu | torch.npu | ascend` returns **0 hits**.
5. The **only** NPU-aware code is in
   `specforge/modeling/target/sglang_backend/model_runner.py:78–79`:
   ```python
   elif self.device == "npu":
       backend = "hccl"
   ```
   This branch is inherited from SGLang's own `ModelRunner`; SpecForge merely overrides
   `init_torch_distributed`. It is reached only when `--target-model-backend sglang`
   and does not, by itself, mean the training loop works on NPU — the draft model,
   FSDP wrapping, optimizer, and tensor placement still go through `torch.cuda.*`.

In short: deploying DFlash on Ascend NPU requires real porting work.

---

## 4. Deploying DFlash on Ascend NPU

### 4.1 Environment

- **Hardware / driver** — Atlas A2/A3 training cards + matching CANN (≥7.0 recommended)
  + HCCL.
- **Python** — 3.11 (required by `pyproject.toml`).
- **PyTorch** — install **Ascend's torch + torch_npu** (the 2.4 / 2.5 line is the safer
  bet today). Note this conflicts with the pinned `torch==2.9.1` in `pyproject.toml`;
  loosen the pin or use the latest community wheel.
- **Pure-Python deps** — `transformers==4.57.1`, `accelerate`, `datasets`, `wandb`,
  `tqdm`, `tensorboard` install normally.
- **Avoid `sglang==0.5.9`** initially — its NPU support is immature, and SpecForge's
  `sglang_backend/model_runner.py` is tightly coupled to that version. Start with
  `--target-model-backend hf` to bypass it.
- **Attention backend** — `flash-attn` is not available on NPU; FlexAttention is also
  risky. Use `--attention-backend sdpa` (torch_npu provides a fallback SDPA
  implementation).
- **`yunchang`** (sequence parallel) — only required for long-context EAGLE3 multi-host
  training. Single-host DFlash does not need it, but `specforge/distributed.py`
  imports it eagerly; either stub it out or keep it on CPU paths only.

### 4.2 Code patches required (minimum)

1. `specforge/distributed.py:66–120`
   - `nccl` → `hccl`
   - `torch.cuda.*` → `torch.npu.*`
   - `init_device_mesh("cuda", ...)` → `("npu", ...)`
   - `DeviceMesh.from_group(..., device_type="cuda")` → `"npu"`
   - Recommendation: add a top-level constant
     `DEVICE = "npu" if torch_npu_available else "cuda"` and use it everywhere.

2. `scripts/train_dflash.py:163, 194, 426, 499–505`
   - Replace `.cuda()` / `device="cuda"` with `.to(device)` / `device=device`.

3. `specforge/modeling/target/dflash_target_model.py:101`
   - `torch.cuda.current_device()` → `torch.npu.current_device()` (only if you keep
     using the SGLang backend).

4. Add `import torch_npu` at the program entry so PyTorch registers the NPU backend.

These four patches are enough to bring up the `--target-model-backend hf` path.

### 4.3 Sample launch script (after patching)

```bash
# Assumes the patches in §4.2 are applied
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

## 5. Main Entry & Call Stack (DFlash online)

**Entry point:** `scripts/train_dflash.py:561` → `if __name__ == "__main__": main()`

```
main()                                                        # train_dflash.py:342
├── parse_args()                                              # train_dflash.py:39
├── set_seed(seed)
├── init_distributed(timeout, tp_size)                        # specforge/distributed.py:66
│     dist.init_process_group("nccl", ...)                    [NPU: switch to "hccl"]
│     torch.cuda.set_device(...)                              [NPU: torch.npu]
│     init_device_mesh("cuda", (dp, tp))                      [NPU: "npu"]
│     set_seq_parallel_pg(...)                                # yunchang
├── build_models(args)                                        # train_dflash.py:149
│     ├── get_dflash_target_model(backend=hf|sglang)          # dflash_target_model.py:290
│     │     ├── HFDFlashTargetModel.from_pretrained()         # AutoModelForCausalLM
│     │     └── SGLangDFlashTargetModel.from_pretrained()     # SGLangRunner (sglang 0.5.9)
│     │           └── init_torch_distributed()                # has the only npu/hccl branch
│     ├── DFlashDraftModel(draft_config).cuda()               # modeling/draft/dflash.py:212
│     └── target_model.set_capture_layers(target_layer_ids)
├── (optional) load checkpoint
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
└── training loop (epoch × step):
        target_model.generate_dflash_data(ids, mask, lm)      # produces hidden_states
        ├── HF backend: target_model.forward(output_hidden_states=True)
        └── SGLang backend: SGLangDFlashTargetModel._extend(reqs)
              └── ScheduleBatch + ForwardBatch + model_runner.forward()
        loss, acc = OnlineDFlashModel(input_ids, hidden_states, loss_mask)
        ├── _sample_anchor_positions()                        # random anchors
        ├── _create_noise_embed()                             # mask-token noise
        ├── create_dflash_block_mask()                        # FlexAttention BlockMask / SDPA
        ├── DFlashDraftModel.forward(noise_emb, target_hidden, ...)
        │     └── Qwen3DFlashDecoderLayer x N → norm
        ├── lm_head + block-wise CE loss (optional decay weight)
        └── returns (loss, acc)
        loss.backward(); optimizer.step()
        save_checkpoint() / record_metrics()
└── destroy_distributed()
```

---

## 6. TL;DR

- **DFlash** is a complete, production-style training pipeline in SpecForge — entry,
  wrapper, draft model, target backends, configs, and example launch scripts are all
  there.
- **Ascend NPU** is **not** officially supported. The single `npu/hccl` branch present
  in `sglang_backend/model_runner.py` is inherited from SGLang and does not cover the
  rest of the training stack.
- To run DFlash on Ascend NPU, apply the small device-abstraction patch listed in §4.2,
  start with `--target-model-backend hf` + `--attention-backend sdpa`, and use the
  sample launch script in §4.3.
