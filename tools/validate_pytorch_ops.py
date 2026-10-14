#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Validate PyTorch baseline contracts on CUDA without changing production grids.

Each operator uses its smallest nonempty source descriptor, all call variants of
that descriptor, and every declared dtype. This is a bounded CUDA probe, not a
full latency/NCU collection. CPU factories and no-kernel native calls are reported
explicitly. Process isolation permits continuing after an operator's CUDA error.
"""

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent.parent


def tensor_elements(value):
    if isinstance(value, dict):
        if "shape" in value and "dtype" in value:
            return math.prod(value["shape"])
        return sum(tensor_elements(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(tensor_elements(v) for v in value)
    return 0


def descriptor_elements(value):
    if isinstance(value, (list, tuple)):
        if all(isinstance(v, int) and v >= 0 for v in value):
            return math.prod(value)
        return sum(descriptor_elements(v) for v in value)
    return 1


def select_bindings(module, max_elements):
    def cost(binding):
        if hasattr(module, "config"):
            actual = tensor_elements(module.config(binding, module.DTYPES[0])["inputs"])
            if actual:
                return actual
        return descriptor_elements(binding.get("shape", ()))

    options = [(cost(b), b) for b in module.grid()]
    nonempty = [(n, b) for n, b in options if n > 0]
    if not nonempty:
        raise ValueError("No nonempty source descriptor")
    size, first = min(nonempty, key=lambda row: row[0])
    if size > max_elements:
        raise ValueError("Smallest source descriptor exceeds the CUDA probe limit")
    # Preserve every native call variant for the selected descriptor.
    selected = [(n, b) for n, b in nonempty if b.get("shape") == first.get("shape")]
    if any(n > max_elements for n, b in selected):
        raise ValueError("A source call variant exceeds the CUDA probe limit")
    return [b for n, b in selected]


def iter_tensors(value, torch):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from iter_tensors(v, torch)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from iter_tensors(v, torch)


def input_metadata(value, torch):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, dict):
        return {key: input_metadata(child, torch) for key, child in value.items()}
    if isinstance(value, (tuple, list)):
        return [input_metadata(child, torch) for child in value]
    if isinstance(value, torch.dtype):
        return str(value)
    return value


def validate_operator(name, max_elements):
    import torch

    module = importlib.import_module(name)
    op = module.native()
    if not callable(op):
        raise ValueError("Native unavailable")
    bindings = select_bindings(module, max_elements)
    rows = []
    for binding in bindings:
        for dtype in module.DTYPES:
            row = {"binding": binding, "dtype": str(dtype)}
            try:
                torch.manual_seed(314159)
                args, kwargs = module.build_inputs(binding, dtype, "cuda")
                if hasattr(module, "config"):
                    actual = input_metadata({"args": args, "kwargs": kwargs}, torch)
                    expected = module.config(binding, dtype)["inputs"]
                    if actual != expected:
                        raise ValueError("Real CUDA input metadata differs from config")
                inputs = list(iter_tensors(args, torch))
                old_inputs = [x.clone() for x in inputs]
                torch.cuda.synchronize()
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ]
                ) as profile:
                    output = op(*args, **kwargs)
                    torch.cuda.synchronize()
                kernels = [
                    event.name
                    for event in profile.events()
                    if event.device_type == torch.autograd.DeviceType.CUDA
                    and not event.name.startswith(("Memcpy", "Memset"))
                ]
                if not kernels:
                    raise ValueError("Native emits no CUDA compute kernel")
                for actual, original in zip(inputs, old_inputs):
                    torch.testing.assert_close(
                        actual, original, rtol=0, atol=0, equal_nan=True
                    )
                # Repeated calls must remain valid with the same timed inputs.
                op(*args, **kwargs)
                torch.cuda.synchronize()
                row.update(status="passed", kernel_count=len(kernels))
                del args, kwargs, output, inputs, old_inputs
                torch.cuda.empty_cache()
            except Exception as error:
                row.update(status="failed", reason=f"{type(error).__name__}: {error}")
            rows.append(row)
            if any(
                message in row.get("reason", "")
                for message in ("illegal memory", "device-side assert")
            ):
                return {
                    "op": name,
                    "status": "failed",
                    "cases": rows,
                    "cuda_context_failed": True,
                }
    return {
        "op": name,
        "status": "passed" if all(r["status"] == "passed" for r in rows) else "partial",
        "cases": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", help="Comma-separated operator filenames; default all")
    parser.add_argument("--device", default="0")
    parser.add_argument("--output", default="pytorch_cuda_validation.json")
    parser.add_argument(
        "--max-input-elements",
        type=int,
        default=8_388_608,
        help="Probe limit only; production operator grids stay intact",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120,
        help="Timeout per isolated group or retried operator",
    )
    parser.add_argument("--_worker", help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    # Load Torch (and its stdlib dependencies) before adding operator filenames.
    importlib.import_module("torch")

    ops_dir = ROOT / "ops/pytorch"
    sys.path.insert(0, str(ops_dir))
    if args._worker:
        names = json.loads(Path(args._worker).read_text())
        for name in names:
            try:
                result = validate_operator(name, args.max_input_elements)
            except Exception as error:
                result = {
                    "op": name,
                    "status": "failed",
                    "reason": f"{type(error).__name__}: {error}",
                }
            print(json.dumps(result), flush=True)
            if result.get("cuda_context_failed"):
                return 2
        return 0

    names = (
        args.ops.split(",")
        if args.ops
        else sorted(p.stem for p in ops_dir.glob("*.py") if p.stem != "__init__")
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="pytorch_probe_") as tmp:
        worker_input = Path(tmp) / "names.json"

        def run_group(group):
            worker_input.write_text(json.dumps(group))
            command = [
                sys.executable,
                __file__,
                "--_worker",
                str(worker_input),
                "--device",
                args.device,
                "--max-input-elements",
                str(args.max_input_elements),
            ]
            try:
                process = subprocess.run(
                    command, text=True, capture_output=True, timeout=args.timeout
                )
                rows = []
                for line in process.stdout.splitlines():
                    try:
                        record = json.loads(line)
                        if isinstance(record, dict) and "op" in record:
                            rows.append(record)
                    except ValueError:
                        pass
                return rows
            except subprocess.TimeoutExpired:
                return []

        for start in range(0, len(names), 12):
            group = names[start : start + 12]
            rows = run_group(group)
            seen = {r["op"] for r in rows}
            for name in group:
                if name not in seen:
                    retried = run_group([name])
                    rows.extend(
                        retried
                        or [
                            {
                                "op": name,
                                "status": "unverified",
                                "reason": "Isolated probe did not complete",
                            }
                        ]
                    )
            results.extend(rows)
            output.write_text(json.dumps(results, indent=2) + "\n")
            print(f"Validated {len(results)}/{len(names)} operators", flush=True)
    return 1 if any(r["status"] != "passed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
