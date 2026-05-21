"""
qwen3_5_patch.py
================

Monkey-patches sglang's Qwen3.5 model classes to support EAGLE3/DFlash-style
auxiliary hidden state capture, which SpecForge's sglang DFlash training needs.

Why this is needed
------------------
SpecForge's sglang backend assumes the underlying sglang model exposes:

  * ``set_eagle3_layers_to_capture(layer_ids)`` on the top-level wrapper
  * ``model.model.layers_to_capture`` (a list of int) on the inner model
  * ``aux_hidden_states`` on the model output (a list of per-target-layer tensors)

This contract is implemented natively in ``sglang.srt.models.qwen3`` (the 8B
model), but NOT in ``sglang.srt.models.qwen3_5`` as of the version pinned by
SpecForge. Without it the SGLang DFlash target backend falls back to returning
the last hidden state only, causing a shape mismatch in the DFlash draft
model (``mat1 and mat2 shapes cannot be multiplied (N x 4096 and 20480 x 4096)``).

What this patch does
--------------------
1. Patches ``Qwen3_5ForCausalLM.forward`` (the class that iterates decoder
   layers in qwen3_5.py) so that, when ``self.layers_to_capture`` is set, it
   collects each target layer's output into an ``aux_hidden_states`` list and
   returns ``(hidden_states, aux_hidden_states)``.

2. Patches ``Qwen3_5ForConditionalGeneration`` (and the MoE variant) to:
     - Initialize ``self.capture_aux_hidden_states = False``
     - Provide a ``set_eagle3_layers_to_capture(layer_ids)`` method that
       mirrors qwen3.py (stores ``[val + 1 for val in layer_ids]``)
     - Override ``forward`` to unpack the inner-model tuple and pass
       ``aux_hidden_states`` to the logits processor (which SpecForge has
       already wrapped to concatenate the captured layers).

Caveats
-------
- The wrapper ``forward`` is overridden with a text-only path. Fine for
  DFlash training; do NOT apply to a process that also serves VL inference.
- Idempotent; safe to call multiple times.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

_PATCH_APPLIED = False


def apply_qwen3_5_dflash_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        logger.info("Qwen3.5 DFlash patch already applied; skipping")
        return

    try:
        from sglang.srt.models import qwen3_5
    except ImportError as e:
        raise ImportError(
            "Cannot import sglang.srt.models.qwen3_5 — your sglang install "
            "doesn't include Qwen3.5 support."
        ) from e

    _patch_inner_model(qwen3_5.Qwen3_5ForCausalLM)
    if hasattr(qwen3_5, "Qwen3_5MoeForCausalLM"):
        _patch_inner_model(qwen3_5.Qwen3_5MoeForCausalLM)

    _patch_wrapper(qwen3_5.Qwen3_5ForConditionalGeneration)
    if hasattr(qwen3_5, "Qwen3_5MoeForConditionalGeneration"):
        _patch_wrapper(qwen3_5.Qwen3_5MoeForConditionalGeneration)

    _PATCH_APPLIED = True
    logger.info(
        "Applied Qwen3.5 DFlash hidden-state capture patch to sglang model classes"
    )


def _patch_inner_model(cls: type) -> None:
    if getattr(cls, "_dflash_inner_patched", False):
        return

    @torch.no_grad()
    def patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors=None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ):
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        layers_to_capture = getattr(self, "layers_to_capture", None)
        aux_hidden_states: Optional[List[torch.Tensor]] = (
            [] if layers_to_capture is not None else None
        )

        from sglang.srt.eplb.expert_distribution import (
            get_global_expert_distribution_recorder,
        )

        for layer_idx in range(len(self.layers)):
            layer = self.layers[layer_idx]
            with get_global_expert_distribution_recorder().with_current_layer(
                layer_idx
            ):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                )

            if (
                input_deepstack_embeds is not None
                and input_deepstack_embeds.numel() > 0
                and layer_idx < 3
            ):
                sep = self.hidden_size * layer_idx
                hidden_states.add_(
                    input_deepstack_embeds[:, sep : sep + self.hidden_size]
                )

            if (
                aux_hidden_states is not None
                and (layer_idx + 1) in layers_to_capture
            ):
                if residual is not None:
                    captured = hidden_states + residual
                else:
                    captured = hidden_states
                aux_hidden_states.append(captured)

        if not self.pp_group.is_last_rank:
            from sglang.srt.distributed.parallel_state import PPProxyTensors

            return PPProxyTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if aux_hidden_states is not None:
            return hidden_states, aux_hidden_states
        return hidden_states

    cls.forward = patched_forward
    cls._dflash_inner_patched = True
    logger.info(f"Patched inner model: {cls.__module__}.{cls.__name__}.forward")


def _patch_wrapper(cls: type) -> None:
    if getattr(cls, "_dflash_wrapper_patched", False):
        return

    original_init = cls.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.capture_aux_hidden_states = False

    def set_eagle3_layers_to_capture(
        self, layer_ids: Optional[List[int]] = None
    ) -> None:
        if hasattr(self, "pp_group") and not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True

        if layer_ids is None:
            inner_cfg = getattr(self.config, "text_config", self.config)
            num_layers = inner_cfg.num_hidden_layers
            self.model.layers_to_capture = [2, num_layers // 2, num_layers - 3]
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

        logger.info(
            f"{cls.__name__}.set_eagle3_layers_to_capture: "
            f"layers_to_capture = {self.model.layers_to_capture}"
        )

    @torch.no_grad()
    def patched_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors=None,
        **kwargs,
    ):
        out = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds=input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        aux_hidden_states: Optional[List[torch.Tensor]] = None
        if self.capture_aux_hidden_states and isinstance(out, tuple):
            hidden_states, aux_hidden_states = out
        else:
            hidden_states = out

        if hasattr(self, "pp_group") and not self.pp_group.is_last_rank:
            return hidden_states

        if get_embedding:
            if hasattr(self, "pooler"):
                return self.pooler(hidden_states, forward_batch)
            return hidden_states

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
            aux_hidden_states,
        )

    cls.__init__ = patched_init
    cls.set_eagle3_layers_to_capture = set_eagle3_layers_to_capture
    cls.forward = patched_forward
    cls._dflash_wrapper_patched = True
    logger.info(
        f"Patched wrapper: {cls.__module__}.{cls.__name__} "
        "(added set_eagle3_layers_to_capture, overrode forward)"
    )
