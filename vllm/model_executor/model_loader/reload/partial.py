# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Resolution of checkpoint weight names to model modules for partial reloads."""

from collections.abc import Iterable

import torch

from vllm.logger import init_logger

__all__ = ["resolve_touched_modules"]

logger = init_logger(__name__)


def _deepest_module(
    named_modules: dict[str, torch.nn.Module], weight_name: str
) -> torch.nn.Module | None:
    """Find the deepest module whose qualname is a prefix of ``weight_name``."""
    parts = weight_name.split(".")
    for depth in range(len(parts) - 1, 0, -1):
        module = named_modules.get(".".join(parts[:depth]))
        if module is not None:
            return module
    return None


def resolve_touched_modules(
    model: torch.nn.Module,
    weight_names: Iterable[str],
) -> set[torch.nn.Module]:
    """Resolve checkpoint weight names to the modules they may load into.

    Names are first mapped through the model's ``hf_to_vllm_mapper`` (when
    defined), so stacked renames (e.g. ``q_proj`` -> ``qkv_proj``) resolve to
    the exact target module. Each mapped name is then matched to its deepest
    enclosing module, and that module's whole subtree is included, which
    bounds every parameter the model's ``load_weights`` may write for the
    name. Names the mapper drops are excluded, since ``load_weights`` will
    not consume them.

    Args:
        model: model the weights will be loaded into
        weight_names: checkpoint-format weight names of the partial update

    Returns:
        the set of modules that may receive weights from the update

    Raises:
        ValueError: if a name cannot be matched to any module. The model is
            untouched when this is raised.
    """
    mapper = getattr(model, "hf_to_vllm_mapper", None)
    named_modules = dict(model.named_modules())
    touched_roots: set[torch.nn.Module] = set()
    unresolved: list[str] = []

    for name in weight_names:
        mapped_name = name
        if mapper is not None:
            mapped_name = mapper._map_name(name)
            if mapped_name is None:
                logger.debug("%s: dropped by weights mapper", name)
                continue

        module = _deepest_module(named_modules, mapped_name)
        if module is None:
            unresolved.append(name)
        else:
            touched_roots.add(module)

    if unresolved:
        raise ValueError(
            "Cannot resolve the modules the following weights belong to, so "
            f"they cannot be partially reloaded: {unresolved}. Use a full "
            "reload instead."
        )

    touched: set[torch.nn.Module] = set()
    for root in touched_roots:
        touched.update(root.modules())
    return touched
