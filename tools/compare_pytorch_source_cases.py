# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Execute pinned source constructors/input builders with NVIDIA capabilities.

Candidate backends and timing are stubbed; original source branch selection,
CORE shape selection, dtype lists and input iterators remain executable.
Compares multisets of input metadata, not tensor values or numerical results.
Run as a separate process: this tool redirects tensor factories to meta.
"""
import argparse
import ast
import collections
import importlib
import importlib.abc
import importlib.util
import inspect
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tarfile
import tempfile
import types
import warnings

import pytest
import torch

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(
    description="Compare catalog input structures with pinned NVIDIA CORE source cases (no timing). Requires torch and pytest."
)
parser.add_argument(
    "--source-repo",
    required=True,
    help="Local FlagGems git repository; exports the manifest commit without changing its checkout",
)
parser.add_argument("--output", required=True)
parser.add_argument("--ops", help="Comma-separated module names; default: full catalog")
args = parser.parse_args()
PR = Path(__file__).resolve().parents[1]
OUT = Path(args.output)
OUT.parent.mkdir(parents=True, exist_ok=True)
source_manifest = json.loads((PR / "ops/pytorch/_kernelgen/manifest.json").read_text())
source_dir = tempfile.TemporaryDirectory(prefix="pytorch_source_parity_")
SRC = Path(source_dir.name)
archive = subprocess.check_output(
    [
        "git",
        "-C",
        args.source_repo,
        "archive",
        source_manifest["source_commit"],
        "benchmark",
        "src/flag_gems/runtime/backend/_nvidia",
        "src/flag_gems/runtime/backend/backend_utils.py",
    ]
)
with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
    stream.extractall(SRC, filter="data")


# Replace only the candidate backend and benchmark timing/config integration.
# Source shape selection and input-building methods below are executed unchanged.
fg = types.ModuleType("flag_gems")
fg.__path__ = []
fg.device = "cuda"
fg.vendor_name = "nvidia"
fg.__version__ = "source-audit-stub"
fg.__getattr__ = lambda name: getattr(torch, name, None)
rt = types.ModuleType("flag_gems.runtime")
rt.__path__ = []


# Evaluate the pinned VendorDescriptor defaults and NVIDIA overrides rather
# than reproducing a hand-maintained capability list in the comparison tool.
def load_file_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


load_file_module(
    "backend_utils", SRC / "src/flag_gems/runtime/backend/backend_utils.py"
)
vendor = load_file_module(
    "source_nvidia", SRC / "src/flag_gems/runtime/backend/_nvidia/__init__.py"
).vendor_info
rt.device = types.SimpleNamespace(
    **{
        "support_" + kind: getattr(vendor, kind + "_enabled")
        for kind in ("bf16", "fp64", "int64", "fp8")
    },
    vendor_name=vendor.vendor_name,
)
rt.torch_backend_device = torch.backends.cuda
rt.torch_device_fn = torch.cuda
fg.runtime = rt
sys.modules["flag_gems"] = fg
sys.modules["flag_gems.runtime"] = rt
fu = types.ModuleType("flag_gems.utils")
fu.__path__ = []
fu.shape_utils = types.SimpleNamespace(
    size_in_bytes=lambda x: x.numel() * x.element_size()
)
sys.modules["flag_gems.utils"] = fu
sys.modules["flag_gems.utils.shape_utils"] = fu.shape_utils


class FakeLoader(importlib.abc.Loader):
    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__path__ = []
        module.__file__ = "<candidate-stub>"
        module.__getattr__ = lambda n: getattr(torch, n, lambda *a, **k: None)


class FakeFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("flag_gems.ops", "flag_gems.fused")):
            return importlib.util.spec_from_loader(
                fullname, FakeLoader(), is_package=True
            )


sys.meta_path.insert(0, FakeFinder())
conf = types.ModuleType("benchmark.conftest")
conf.Config = types.SimpleNamespace(
    mode=None,
    bench_level=None,
    query=False,
    user_desired_dtypes=None,
    user_desired_metrics=None,
    shape_file=str(SRC / "benchmark/core_shapes.yaml"),
    reference_only=False,
    current_nodeid="source_audit",
)
for n in ("emit_record_logger", "update_case_list", "update_result"):
    setattr(conf, n, lambda *a, **kw: None)
sys.modules["benchmark.conftest"] = conf
sys.path.insert(0, str(SRC))
from benchmark import base, consts

conf.Config.mode = consts.BenchMode.KERNEL
conf.Config.bench_level = consts.BenchLevel.CORE
base.Benchmark.device = "meta"

# Avoid allocating large real CPU/GPU tensors in a structural audit. Keep small
# CPU scalar draws real, since some builders use their values as call parameters.
for n in (
    "randn",
    "rand",
    "randint",
    "zeros",
    "ones",
    "empty",
    "full",
    "arange",
    "tensor",
    "randperm",
    "eye",
):
    original = getattr(torch, n)

    def wrapper(*a, _original=original, _name=n, **kw):
        device = str(kw.get("device", "cpu"))
        if device.startswith("cuda"):
            kw["device"] = "meta"
        if device == "cpu" and _name != "tensor":
            shape = kw.get("size")
            if shape is None:
                shape = next((x for x in a if isinstance(x, (list, tuple))), None)
            if (
                shape is not None
                and all(isinstance(x, int) for x in shape)
                and math.prod(shape) > 65536
            ):
                kw["device"] = "meta"
        return _original(*a, **kw)

    setattr(torch, n, wrapper)

captures = []
created = []
original_init = base.Benchmark.__init__


def tracking_init(self, *a, **kw):
    original_init(self, *a, **kw)
    created.append(self)


base.Benchmark.__init__ = tracking_init


def capture(self, *a, **kw):
    self.init_user_config()
    self.device = "meta"
    captures.append(self)


base.Benchmark.run = capture

manifest = json.loads((PR / "ops/pytorch/_kernelgen/manifest.json").read_text())[
    "operators"
]
catalog = json.loads((PR / "ops/pytorch/_kernelgen/cases.json").read_text())
manifest["silu"] = {
    "source_file": "benchmark/test_silu.py",
    "source_test": "test_silu",
    "module": "silu",
}
module_cache = {}
benches_cache = {}


def load_source(file):
    if file in module_cache:
        return module_cache[file]
    path = SRC / file
    name = "benchmark." + str(
        Path(file).relative_to("benchmark").with_suffix("")
    ).replace("/", ".")
    tree = ast.parse(path.read_text(), filename=str(path))
    decorators = {
        n.name: list(n.decorator_list)
        for n in tree.body
        if isinstance(n, ast.FunctionDef)
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.decorator_list = []
    mod = types.ModuleType(name)
    mod.__file__ = str(path)
    mod.__package__ = name.rsplit(".", 1)[0]
    sys.modules[name] = mod
    exec(compile(tree, str(path), "exec"), mod.__dict__)
    for value in list(mod.__dict__.values()):
        if isinstance(value, type) and issubclass(value, base.Benchmark):
            value.run = capture
    mod._audit_decorators = decorators
    module_cache[file] = mod
    return mod


def source_benches(info):
    key = (info["source_file"], info["source_test"])
    if key not in benches_cache:
        mod = load_source(key[0])
        captures.clear()
        created.clear()
        fn = getattr(mod, key[1])
        param_sets = [{}]
        for deco in mod._audit_decorators.get(key[1], []):
            if (
                isinstance(deco, ast.Call)
                and isinstance(deco.func, ast.Attribute)
                and deco.func.attr == "parametrize"
            ):
                keys = eval(
                    compile(ast.Expression(deco.args[0]), "<parametrize>", "eval"),
                    mod.__dict__,
                )
                keys = (
                    [x.strip() for x in keys.split(",")]
                    if isinstance(keys, str)
                    else keys
                )
                values = eval(
                    compile(ast.Expression(deco.args[1]), "<parametrize>", "eval"),
                    mod.__dict__,
                )
                rows = []
                for v in values:
                    if hasattr(v, "values"):
                        v = v.values
                    elif len(keys) == 1:
                        v = (v,)
                    rows.append(dict(zip(keys, v)))
                param_sets = [{**old, **new} for old in param_sets for new in rows]
        monkey = pytest.MonkeyPatch()
        try:
            for parameters in param_sets:
                if "monkeypatch" in inspect.signature(fn).parameters:
                    parameters["monkeypatch"] = monkey
                fn(**parameters)
        finally:
            monkey.undo()
        for instance in created:
            if instance not in captures:
                capture(instance)
        benches_cache[key] = list(captures)
    return benches_cache[key]


def describe(v):
    if isinstance(v, torch.Tensor):
        return {
            "shape": list(v.shape),
            "dtype": str(v.dtype),
            "stride": list(v.stride()),
            "requires_grad": v.requires_grad,
        }
    if isinstance(v, dict):
        return {
            k: ("cuda" if k == "device" and str(x) == "meta" else describe(x))
            for k, x in v.items()
        }
    if isinstance(v, (list, tuple)):
        return [describe(x) for x in v]
    if isinstance(v, torch.dtype):
        return str(v)
    if isinstance(v, torch.device):
        return str(v)
    return v


def metadata(value):
    value = value if isinstance(value, (list, tuple)) else (value,)
    a, k = (
        (tuple(value[:-1]), value[-1])
        if value and isinstance(value[-1], dict)
        else (tuple(value), {})
    )
    return describe({"args": a, "kwargs": k})


def normalize(v):
    # Random workload parameters were intentionally stabilized in the PR.
    return json.dumps(v, sort_keys=True, default=str)


sys.path.insert(0, str(PR / "ops/pytorch"))
result = {}
if args.ops:
    manifest = {name: manifest[name] for name in args.ops.split(",")}
for idx, (name, info) in enumerate(manifest.items()):
    row = {"source_file": info["source_file"], "source_test": info["source_test"]}
    try:
        all_benches = source_benches(info)
        if name == "sparse_semi_structured_linear":
            all_benches += source_benches(
                {**info, "source_test": "test_sparse_semi_structured_linear_with_bias"}
            )
        if name == "einsum":
            all_benches += source_benches({**info, "source_test": "test_einsum_bmm"})
        if name.startswith("to_copy_from_"):
            benches = [
                b
                for b in all_benches
                if str(getattr(b, "src_dtype", None))
                == "torch." + name.removeprefix("to_copy_from_")
            ]
        else:
            benches = [
                b for b in all_benches if b.op_name == info.get("source_op", name)
            ]
            if not benches and len(all_benches) == 1:
                benches = all_benches
        if not benches:
            raise ValueError("No matching source benchmark instance")
        module_name = info.get("module", name)
        s = importlib.util.spec_from_file_location(
            "parity_" + module_name, PR / "ops/pytorch" / f"{module_name}.py"
        )
        m = importlib.util.module_from_spec(s)
        s.loader.exec_module(m)
        sdtypes = sorted(set(str(d) for b in benches for d in b.dtypes))
        pdtypes = sorted(str(d) for d in m.DTYPES)
        row["source_dtypes"] = sdtypes
        row["pr_dtypes"] = pdtypes
        partition = info.get("source_dtype_partition")
        if partition:
            siblings = [
                v
                for v in source_manifest["operators"].values()
                if v.get("source_op") == info["source_op"]
            ]
            declared = [dt for v in siblings for dt in v["source_dtype_partition"]]
            assert (
                sorted(declared) == sdtypes
            ), "Source dtype partitions must cover the source exactly once"
            row["dtype_equal"] = sorted(partition) == pdtypes
            row["source_dtype_partition"] = partition
        else:
            row["dtype_equal"] = sdtypes == pdtypes
        row["source_descriptors"] = [describe(b.shapes) for b in benches]
        row["pr_descriptors"] = [x["shape"] for x in m.grid()]
        failures = []
        diffs = []
        checked = 0
        for dt in m.DTYPES:
            source = []
            try:
                for b in benches:
                    if dt not in b.dtypes:
                        continue
                    random.seed(104729)
                    torch.manual_seed(104729)
                    for value in b.get_input_iter(dt):
                        if name == "convert_weight_to_int4pack":
                            value = value[0]
                        meta = metadata(value)
                        if name == "einsum":
                            meta["args"].insert(
                                0, "bij,bjk->bik" if b.batched else "ij,jk->ik"
                            )
                        source.append(meta)
            except Exception as e:
                failures.append(
                    {"dtype": str(dt), "error": type(e).__name__ + ": " + str(e)[:300]}
                )
                continue
            target = []
            for binding in m.grid():
                if hasattr(m, "config"):
                    target.append(m.config(binding, dt)["inputs"])
                else:
                    a, k = m.build_inputs(binding, dt, "meta")
                    target.append(describe({"args": a, "kwargs": k}))
            sc = collections.Counter(map(normalize, source))
            pc = collections.Counter(map(normalize, target))
            checked += len(target)
            if sc != pc:
                diffs.append(
                    {
                        "dtype": str(dt),
                        "source_count": len(source),
                        "pr_count": len(target),
                        "source_only": [
                            json.loads(x) for x in list((sc - pc).elements())[:3]
                        ],
                        "pr_only": [
                            json.loads(x) for x in list((pc - sc).elements())[:3]
                        ],
                        "missing_count": sum((sc - pc).values()),
                        "extra_count": sum((pc - sc).values()),
                    }
                )
        row.update(
            checked_records=checked,
            metadata_differences=diffs,
            construction_errors=failures,
        )
        row["status"] = (
            "equal"
            if not diffs and not failures and row["dtype_equal"]
            else ("different" if diffs or not row["dtype_equal"] else "unresolved")
        )
    except Exception as e:
        row.update(status="unresolved", error=type(e).__name__ + ": " + str(e)[:500])
    result[name] = row
    if (idx + 1) % 50 == 0:
        OUT.write_text(json.dumps(result, indent=2, default=str))
        print(
            idx + 1,
            collections.Counter(r["status"] for r in result.values()),
            flush=True,
        )
OUT.write_text(json.dumps(result, indent=2, default=str))
print(
    "FINAL",
    len(result),
    collections.Counter(r["status"] for r in result.values()),
    flush=True,
)

source_dir.cleanup()
sys.exit(0 if all(r["status"] == "equal" for r in result.values()) else 1)
