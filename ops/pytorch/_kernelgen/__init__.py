# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Input and contract helpers for pinned FlagGems kernelgen-dev recipes."""

from dataclasses import dataclass, field
from copy import deepcopy
from functools import lru_cache
import hashlib
import random
import json
from pathlib import Path

import torch


@dataclass
class BenchmarkCasePlan:
    shape: object = None
    params: dict = field(default_factory=dict)
    builder_args: tuple = ()


def generate_tensor_input(shape, dtype, device):
    """FlagGems benchmark/utils.py FLOAT/INT/BOOL/COMPLEX construction."""
    if dtype in (torch.float16, torch.float32, torch.bfloat16, torch.complex64):
        return torch.randn(shape, dtype=dtype, device=device)
    if dtype == torch.bool:
        return torch.randint(
            0, 2, shape, device="meta" if device == "meta" else "cpu", dtype=dtype
        ).to(device)
    if dtype not in (torch.int16, torch.int32):
        raise ValueError(f"Upstream generate_tensor_input does not support {dtype}")
    info = torch.iinfo(dtype)
    return torch.randint(
        info.min,
        info.max,
        shape,
        device="meta" if device == "meta" else "cpu",
        dtype=dtype,
    ).to(device)


def repeat_interleave_repeats(shape, dtype, device):
    """Replay workload-defining repeats independently of process/device RNG.

    These two recipes have one variant per shape. Generate on CPU with a local
    generator so CPU and CUDA reconstruction agree without changing global RNG.
    Preserve the source's int64 repeats and uniform [0, 31) distribution.
    """
    if torch.device(device).type == "meta":
        return torch.empty((shape[0],), dtype=torch.int64, device=device)
    identity = json.dumps(["repeat_interleave_repeats_v1", list(shape), str(dtype)])
    seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randint(
        0, 31, (shape[0],), dtype=torch.int64, generator=generator
    ).to(device)


def unpack_inputs(value):
    """Upstream unpack_to_args_kwargs: the final dict contains keyword args."""
    if not isinstance(value, (tuple, list)):
        value = (value,)
    if value and isinstance(value[-1], dict):
        return tuple(value[:-1]), value[-1]
    return tuple(value), {}


def standard_inputs(kind, shape, dtype, device):
    x = generate_tensor_input(shape, dtype, device)
    if kind == "unary":
        yield (x,)
    elif kind == "unary_out":
        yield x, {"out": torch.empty_like(x)}
    elif kind == "binary":
        yield x, generate_tensor_input(shape, dtype, device)
    elif kind == "reduction":
        yield (x, 1) if x.ndim > 1 else (x,)
    else:
        raise ValueError(f"Unknown input family: {kind}")


def case_inputs(case_fn, builder, shape, dtype, device):
    for plan in case_fn(shape, dtype):
        yield builder(plan, dtype, device)


@lru_cache(maxsize=1)
def _case_catalog():
    return json.loads((Path(__file__).parent / "cases.json").read_text())


def export_contract(name, dtypes, native_resolver, input_iter):
    """Expose the existing ops contract without importing FlagGems or pytest.

    Descriptor/variant metadata was enumerated from upstream builders on meta
    tensors at generation time. Real inputs are reconstructed on the requested
    device; no tensors are cached across latency/NCU processes.
    """
    records = _case_catalog()[name]
    multiple_variants = any(row["variant"] for row in records)

    def native():
        try:
            op = native_resolver()
            return op if callable(op) else None
        except (AttributeError, ImportError, RuntimeError):
            return None

    def grid():
        return [
            {"shape": deepcopy(row["shape"]), "variant": row["variant"]}
            for row in records
        ]

    def key_shape(binding):
        shape = binding["shape"]
        if all(isinstance(d, int) for d in shape):
            if not multiple_variants:
                return list(shape)
            return "x".join(map(str, shape)) + f"_case{binding['variant']}"
        # Nested descriptors remain in config. A compact stable key avoids
        # exceeding NCU report filename limits when the collector names reports.
        encoded = json.dumps(shape, separators=(",", ":")).encode()
        return f"descriptor_{hashlib.sha256(encoded).hexdigest()[:16]}_case{binding['variant']}"

    def build_inputs(binding, dtype, device):
        def construct():
            iterator = iter(input_iter(tuple(binding["shape"]), dtype, device))
            for _ in range(binding["variant"] + 1):
                value = next(iterator)
            return unpack_inputs(value)

        if name not in {
            "constant_pad_nd",
            "count_nonzero",
            "diag",
            "index_select_backward",
            "linspace",
            "pad",
            "select_scatter",
        }:
            return construct()
        # These upstream builders sample scalar dimensions/padding/steps. Keep
        # those parameters identical across config, latency and NCU replay.
        # Restore Python/CPU RNG state; CUDA tensor data remains independently
        # randomized as in the upstream benchmarks. Builders run sequentially
        # in the collector's isolated processes.
        seed_data = json.dumps([name, binding, str(dtype)], sort_keys=True).encode()
        seed = int.from_bytes(hashlib.sha256(seed_data).digest()[:8], "big")
        python_state = random.getstate()
        cpu_state = torch.random.get_rng_state()
        try:
            random.seed(seed)
            generator = torch.Generator(device="cpu").manual_seed(seed)
            torch.random.set_rng_state(generator.get_state())
            return construct()
        finally:
            random.setstate(python_state)
            torch.random.set_rng_state(cpu_state)

    def config(binding, dtype):
        row = next(
            r
            for r in records
            if r["shape"] == binding["shape"] and r["variant"] == binding["variant"]
        )
        result = deepcopy(row["config"][str(dtype)])
        result["variant"] = binding["variant"]
        return result

    return {
        "OP_NAME": name,
        "DTYPES": dtypes,
        "IS_INPLACE": False,
        "native": native,
        "grid": grid,
        "build_inputs": build_inputs,
        "key_shape": key_shape,
        "config": config,
    }
