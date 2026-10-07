# Alpaccaroo - minimal target-device PyTorch/Triton ROCm smoke kernel.
# MIT License. See LICENSE.
"""Imported only after the strict Qwen35 ROCm environment checks pass."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _increment(source, target, size: tl.constexpr):
    offset = tl.arange(0, size)
    target_value = tl.load(source + offset) + 1.0
    tl.store(target + offset, target_value)


def run_probe(device_index: int) -> None:
    """Prove one allocation and one JIT launch on the selected HIP device."""

    device = f"cuda:{device_index}"
    with torch.cuda.device(device_index):
        source = torch.zeros((1,), device=device, dtype=torch.float32)
        target = torch.empty_like(source)
        _increment[(1,)](source, target, size=1)
        torch.cuda.synchronize(device_index)
        value = float(target.item())
    if value != 1.0:
        raise RuntimeError(f"Triton ROCm smoke returned {value}, expected 1.0")


__all__ = ["run_probe"]
