from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixCache
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardBatch
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.server_args import ServerArgs
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm
from sglang.srt.utils import require_mlp_sync, require_mlp_tp_gather
from transformers import AutoModelForCausalLM

from specforge.distributed import get_tp_group

from .sglang_backend import SGLangRunner


# Detect torch_npu once at import time. We use this both to flip SGLang's
# device/attention defaults and to choose the right current_device() call.
try:
    import torch_npu  # noqa: F401
    _TORCH_NPU_AVAILABLE = True
except ImportError:
    _TORCH_NPU_AVAILABLE = False


def _current_device_id() -> int:
    """Return the current accelerator device id. On NPU we call torch.npu
    directly rather than relying on torch_npu's torch.cuda aliasing, which
    is reliable for most ops but not guaranteed for current_device()."""
    if _TORCH_NPU_AVAILABLE and torch.npu.is_available():
        return torch.npu.current_device()
    return torch.cuda.current_device()


@dataclass
class DFlashTargetOutput:
    hidden_states: torch.Tensor  # [batch, seq_len, hidden_size]
    input_ids: torch.Tensor  # [batch, seq_len]
    attention_mask: torch.Tensor  # [batch, seq_len]
    loss_mask: torch.Tensor  # [batch, seq_len]


class DFlashTargetModel(ABC):
    """Abstract base class for DFlash target model backend."""

    def __init__(self):
        self.capture_layer_ids = None

    @classmethod
    @abstractmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "DFlashTargetModel":
        """Initialize the target model backend."""

    @abstractmethod
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        """Generate context hidden states for DFlash training."""

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        """Set which layers' hidden states to capture."""
        self.capture_layer_ids = layer_ids


class SGLangDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model_runner: SGLangRunner):
        super().__init__()
        self.model_runner = model_runner

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> "SGLangDFlashTargetModel":
        tp_size = dist.get_world_size(get_tp_group())

        # ------------------------------------------------------------------
        # NPU-aware defaults. SpecForge's SGLangBackendArgs may not expose
        # the NPU-specific flags, so we set sensible defaults here when
        # torch_npu is detected. Anything already in kwargs (i.e. explicitly
        # passed by the user) wins.
        #
        # Environment-variable overrides (read here as a fallback channel
        # when the schema doesn't expose them):
        #   SPECFORGE_SGLANG_MEM_FRACTION_STATIC
        #   SPECFORGE_SGLANG_EP_SIZE
        #   SPECFORGE_SGLANG_DISABLE_RADIX_CACHE  (0/1)
        # ------------------------------------------------------------------
        if _TORCH_NPU_AVAILABLE:
            kwargs.setdefault("device", "npu")
            ##### @Moh_7596 — attention backend from env var (single source of truth)
            # Set SPECFORGE_ATTENTION_BACKEND in the launch script.
            # See SpecForge/docs/ascend_npu/run_deepseek_v4_flash_dflash_sglang_npu.sh
            _env_backend = os.environ.get("SPECFORGE_ATTENTION_BACKEND")
            if _env_backend:
                kwargs["attention_backend"] = _env_backend
                print(
                    f"[@Moh_7596] attention_backend={_env_backend!r} "
                    f"(from SPECFORGE_ATTENTION_BACKEND)",
                    flush=True,
                )
            else:
                kwargs.setdefault("attention_backend", "ascend")
            ################
            # SGLang NPU recipes typically run without the radix cache for
            # the kind of one-shot forwards we do here (training data prep).
            kwargs.setdefault("disable_radix_cache", True)

        # mem_fraction_static: leave headroom for the draft + FSDP states
        # that live in the same process. Default 0.88 is too aggressive.
        if "mem_fraction_static" not in kwargs:
            env_mf = os.environ.get("SPECFORGE_SGLANG_MEM_FRACTION_STATIC")
            kwargs["mem_fraction_static"] = (
                float(env_mf) if env_mf is not None else 0.55
            )

        # ep_size: MoE expert parallelism. For DeepSeek-V4 you almost
        # always want ep_size == tp_size.
        if "ep_size" not in kwargs:
            env_ep = os.environ.get("SPECFORGE_SGLANG_EP_SIZE")
            kwargs["ep_size"] = int(env_ep) if env_ep is not None else tp_size

        # Optional radix-cache override via env var.
        env_dr = os.environ.get("SPECFORGE_SGLANG_DISABLE_RADIX_CACHE")
        if env_dr is not None:
            kwargs["disable_radix_cache"] = env_dr not in ("0", "false", "False")

        # Build the SGLang ServerArgs. Note disable_cuda_graph=True covers
        # both CUDA-graph and (on most builds) NPU-graph paths — graph
        # capture is brittle inside a training process that's also running
        # FSDP backward on the draft.
        server_args = ServerArgs(
            model_path=pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
            dtype=torch_dtype,
            enable_return_hidden_states=True,  # Critical for DFlash
            disable_cuda_graph=True,
            tp_size=tp_size,
            pp_size=1,
            **kwargs,
        )

        # Debug print on rank 0 so we can verify the args got through.
        if dist.get_rank() == 0:
            print(
                f"[SGLangDFlashTargetModel] ServerArgs: "
                f"device={server_args.device}, "
                f"attention_backend={getattr(server_args, 'attention_backend', None)}, "
                f"tp_size={server_args.tp_size}, "
                f"ep_size={server_args.ep_size}, "
                f"mem_fraction_static={server_args.mem_fraction_static}, "
                f"disable_radix_cache={getattr(server_args, 'disable_radix_cache', None)}, "
                f"disable_cuda_graph={server_args.disable_cuda_graph}",
                flush=True,
            )

        tp_rank = dist.get_rank(get_tp_group())
        moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
        model_config = ModelConfig.from_server_args(server_args)

        model_runner = SGLangRunner(
            model_config=model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            gpu_id=_current_device_id(),  # NPU-safe
            tp_rank=tp_rank,
            tp_size=server_args.tp_size,
            moe_ep_rank=moe_ep_rank,
            moe_ep_size=server_args.ep_size,
            pp_rank=0,
            pp_size=1,
            server_args=server_args,
            nccl_port=None,
        )
        return cls(model_runner)

    def set_capture_layers(self, layer_ids: List[int]) -> None:
        super().set_capture_layers(layer_ids)
        if hasattr(self.model_runner.model, "set_eagle3_layers_to_capture"):
            self.model_runner.model.set_eagle3_layers_to_capture(layer_ids)
            print(self.model_runner.model.model.layers_to_capture)

    @torch.no_grad
    def _extend(self, reqs):
        cache_params = CacheInitParams(
            disable=False,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            page_size=self.model_runner.server_args.page_size,
        )
        tree_cache = RadixCache(cache_params)

        batch = ScheduleBatch.init_new(
            reqs=reqs,
            req_to_token_pool=self.model_runner.req_to_token_pool,
            token_to_kv_pool_allocator=self.model_runner.token_to_kv_pool_allocator,
            tree_cache=tree_cache,
            model_config=self.model_runner.model_config,
            enable_overlap=False,
            spec_algorithm=SpeculativeAlgorithm.NONE,
        )
        batch.prepare_for_extend()

        if require_mlp_sync(self.model_runner.server_args):
            Scheduler.prepare_mlp_sync_batch_raw(
                batch,
                dp_size=self.model_runner.server_args.dp_size,
                attn_tp_size=1,
                tp_group=self.model_runner.tp_group,
                get_idle_batch=None,
                disable_cuda_graph=self.model_runner.server_args.disable_cuda_graph,
                spec_algorithm=SpeculativeAlgorithm.NONE,
                speculative_num_draft_tokens=None,
                require_mlp_tp_gather=require_mlp_tp_gather(
                    self.model_runner.server_args
                ),
                disable_overlap_schedule=self.model_runner.server_args.disable_overlap_schedule,
                offload_tags=set(),
            )

        model_worker_batch = batch.get_model_worker_batch()
        forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
        forward_batch.capture_hidden_mode = CaptureHiddenMode.FULL

        output = self.model_runner.forward(forward_batch)
        if hasattr(output, "logits_output"):
            output = output.logits_output

        input_lens = [len(req.origin_input_ids) for req in reqs]
        if (
            hasattr(output, "aux_hidden_states")
            and output.aux_hidden_states is not None
        ):
            hidden_states_list = torch.split(
                output.aux_hidden_states, input_lens, dim=0
            )
        elif hasattr(output, "hidden_states") and output.hidden_states is not None:
            hidden_states_list = torch.split(output.hidden_states, input_lens, dim=0)
        else:
            raise ValueError("SGLang output does not contain hidden states.")

        ##### @Moh_7596 — V4 pre_hc_head has shape [T, num_mtp * hidden_size]
        # because each input token produces hidden states for `num_mtp` future
        # positions (Multi-Token Prediction). The draft model wants only the
        # capture_layer_ids-indexed MTP segments. Slice them out here.
        hidden_size = self.model_runner.model_config.hidden_size
        if (
            self.capture_layer_ids is not None
            and hidden_states_list
            and hidden_states_list[0].shape[-1] > hidden_size
            and hidden_states_list[0].shape[-1] % hidden_size == 0
        ):
            num_segments = hidden_states_list[0].shape[-1] // hidden_size
            if all(0 <= i < num_segments for i in self.capture_layer_ids):
                if not getattr(self, "_v4_mtp_slice_warned", False):
                    import logging as _log
                    _log.getLogger(__name__).warning(
                        f"[@Moh_7596] V4 hidden_states has "
                        f"{num_segments} MTP segments \u00d7 {hidden_size} dim "
                        f"= {hidden_states_list[0].shape[-1]} features. "
                        f"Slicing to capture_layer_ids={self.capture_layer_ids}."
                    )
                    self._v4_mtp_slice_warned = True
                sliced = []
                for hs in hidden_states_list:
                    chunks = [
                        hs[..., i * hidden_size : (i + 1) * hidden_size]
                        for i in self.capture_layer_ids
                    ]
                    sliced.append(torch.cat(chunks, dim=-1))
                hidden_states_list = sliced
        ################
        else:
            raise ValueError("SGLang output does not contain hidden states.")

        self.model_runner.req_to_token_pool.clear()
        self.model_runner.token_to_kv_pool_allocator.clear()

        return hidden_states_list

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        sampling_params = SamplingParams(temperature=0, max_new_tokens=1)
        reqs, data_cache = [], []

        if isinstance(input_ids, torch.Tensor):
            input_ids_list = torch.split(input_ids, 1, dim=0)
            attn_mask_list = torch.split(attention_mask, 1, dim=0)
            loss_mask_list = torch.split(loss_mask, 1, dim=0)

        for idx, (curr_ids, curr_attn, curr_loss) in enumerate(
            zip(input_ids_list, attn_mask_list, loss_mask_list)
        ):
            req = Req(
                rid=str(idx),
                origin_input_text="",
                origin_input_ids=curr_ids.view(-1).tolist(),
                sampling_params=sampling_params,
            )
            req.fill_ids = req.origin_input_ids
            req.extend_input_len = len(req.fill_ids) - len(req.prefix_indices)
            data_cache.append((curr_ids, curr_attn, curr_loss))
            reqs.append(req)

        hidden_states_list = self._extend(reqs)
        # @Moh_7596 — NaN probe
        if hidden_states_list:
            _h0 = hidden_states_list[0]
            _has_nan = bool(_h0.isnan().any().item())
            _has_inf = bool(_h0.isinf().any().item())
            if _has_nan or _has_inf or not getattr(self, "_nan_logged", False):
                import logging as _log
                _log.getLogger(__name__).warning(
                    f"[@Moh_7596 NaN probe] target hidden_states[0]: "
                    f"shape={tuple(_h0.shape)}, dtype={_h0.dtype}, "
                    f"has_nan={_has_nan}, has_inf={_has_inf}, "
                    f"min={_h0.min().item() if not _has_nan else 'nan'}, "
                    f"max={_h0.max().item() if not _has_nan else 'nan'}, "
                    f"mean={_h0.float().mean().item() if not _has_nan else 'nan'}"
                )
                self._nan_logged = True

        # Stack back to batch
        hidden_states = torch.cat([h.unsqueeze(0) for h in hidden_states_list], dim=0)
        input_ids = torch.cat([d[0] for d in data_cache], dim=0)
        attention_mask = torch.cat([d[1] for d in data_cache], dim=0)
        loss_mask = torch.cat([d[2] for d in data_cache], dim=0)

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


class HFDFlashTargetModel(DFlashTargetModel):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: torch.dtype = None,
        device: str = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "HFDFlashTargetModel":

        # Check if actual weights exist, otherwise use from_config for random init
        import glob, os as _os
        _weight_files = glob.glob(_os.path.join(pretrained_model_name_or_path, "*.safetensors")) + \
                        glob.glob(_os.path.join(pretrained_model_name_or_path, "pytorch_model*.bin"))
        if _weight_files:
            target_model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path,
                torch_dtype=torch_dtype,
                cache_dir=cache_dir,
                output_hidden_states=True,
                trust_remote_code=trust_remote_code,
                **kwargs,
            ).eval()
        else:
            from transformers import AutoConfig
            print(f"[SANITY CHECK] No weights found — using from_config() with random init")
            _config = AutoConfig.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust_remote_code
            )
            _config.output_hidden_states = True
            target_model = AutoModelForCausalLM.from_config(
                _config, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
            ).eval()

        if device:
            target_model = target_model.to(device)

        return cls(target_model)

    @torch.no_grad()
    def generate_dflash_data(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        loss_mask: torch.Tensor,
    ) -> DFlashTargetOutput:
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
        )

        # hidden_states[0] = embedding output; hidden_states[i+1] = layer i output
        offset = 1
        selected = []
        if self.capture_layer_ids is not None:
            for idx in self.capture_layer_ids:
                selected.append(outputs.hidden_states[idx + offset])
            hidden_states = torch.cat(selected, dim=-1)
        else:
            hidden_states = outputs.hidden_states[-1]

        return DFlashTargetOutput(
            hidden_states=hidden_states,
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
        )


def get_dflash_target_model(
    pretrained_model_name_or_path: str,
    backend: str = "sglang",
    torch_dtype: torch.dtype = None,
    device: str = None,
    cache_dir: Optional[str] = None,
    **kwargs,
) -> DFlashTargetModel:
    if backend == "sglang":
        return SGLangDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    elif backend == "hf":
        return HFDFlashTargetModel.from_pretrained(
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            torch_dtype=torch_dtype,
            device=device,
            cache_dir=cache_dir,
            **kwargs,
        )
    else:
        raise ValueError(f"Invalid backend: {backend}")