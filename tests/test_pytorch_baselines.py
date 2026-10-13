# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Contract checks over every added recipe; no CUDA or FlagGems installation needed."""

import importlib.util
import json
import random
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops/pytorch"
sys.path.insert(0, str(OPS))
from _kernelgen import generate_tensor_input, unpack_inputs  # noqa: E402


def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, dict):
        return {
            key: "cuda" if key == "device" and child == "meta" else describe(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [describe(child) for child in value]
    if isinstance(value, torch.dtype):
        return str(value)
    return value


class PytorchCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((OPS / "_kernelgen/manifest.json").read_text())
        cls.cases = json.loads((OPS / "_kernelgen/cases.json").read_text())
        cls.modules = {}
        for name, info in cls.manifest["operators"].items():
            path = OPS / (info.get("module", name) + ".py")
            spec = importlib.util.spec_from_file_location("baseline_test_" + name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.modules[name] = module

    def test_inventory_and_source_provenance_match_files(self):
        names = set(self.modules)
        self.assertEqual(names, set(self.cases))
        expected = {
            info.get("module", name)
            for name, info in self.manifest["operators"].items()
        } | {"silu"}
        self.assertEqual(expected, {p.stem for p in OPS.glob("*.py")})
        coverage = json.loads((OPS / "coverage.json").read_text())
        self.assertEqual(coverage["summary"]["added_operators"], len(names))
        for name, module in self.modules.items():
            with self.subTest(op=name):
                info = self.manifest["operators"][name]
                self.assertEqual(module.OP_NAME, name)
                self.assertFalse(module.IS_INPLACE)
                self.assertEqual([str(d) for d in module.DTYPES], info["dtypes"])
                self.assertIn(self.manifest["source_commit"], module.__doc__)
                self.assertTrue(callable(module.native()))

    def test_keys_are_unique_and_ncu_filenames_fit(self):
        for name, module in self.modules.items():
            with self.subTest(op=name):
                bindings = module.grid()
                self.assertEqual(
                    len(bindings), self.manifest["operators"][name]["cases"]
                )
                keys = [str(module.key_shape(b)) for b in bindings]
                self.assertEqual(len(set(keys)), len(keys))
                for b in bindings:
                    key = module.key_shape(b)
                    tag = f"{name}_{'x'.join(map(str, key))}_complex128.ncu-rep"
                    self.assertLessEqual(len(tag.encode()), 255)

    def test_all_source_descriptors_dtypes_and_variants_reconstruct(self):
        for name, module in self.modules.items():
            for binding in module.grid():
                for dtype in module.DTYPES:
                    with self.subTest(op=name, binding=binding, dtype=dtype):
                        before = json.loads(json.dumps(binding))
                        args, kwargs = module.build_inputs(binding, dtype, "meta")
                        self.assertIsInstance(args, tuple)
                        self.assertIsInstance(kwargs, dict)
                        config = module.config(binding, dtype)
                        self.assertEqual(
                            describe({"args": args, "kwargs": kwargs}), config["inputs"]
                        )
                        self.assertEqual(
                            config["source_commit"], self.manifest["source_commit"]
                        )
                        self.assertEqual(binding, before)
                        json.dumps(config)

    def test_bindings_and_config_are_not_shared_mutable_state(self):
        module = self.modules["mm"]
        original = module.grid()[0]
        changed = module.grid()[0]
        changed["shape"][0] = -123
        self.assertEqual(module.grid()[0], original)
        config = module.config(original, module.DTYPES[0])
        config["inputs"]["args"][0]["shape"][0] = -123
        self.assertNotEqual(module.config(original, module.DTYPES[0]), config)

    def test_fixed_geometry_is_listed_once_per_source_case(self):
        for name, info in self.manifest["operators"].items():
            if not info["shape_source"].startswith("fixed"):
                continue
            module = self.modules[name]
            variants = [b["variant"] for b in module.grid()]
            self.assertEqual(len(variants), len(set(variants)))
        pool = self.modules["max_pool3d"]
        self.assertEqual(pool.grid()[0]["shape"], [4, 3, 16, 56, 56])
        self.assertEqual(len(pool.grid()), 4)

    def test_out_buffer_and_boolean_dtype_semantics(self):
        module = self.modules["exp_out"]
        args, kwargs = module.build_inputs(
            {"shape": [4, 8], "variant": 0}, torch.float32, "cpu"
        )
        output = module.native()(*args, **kwargs)
        self.assertIs(output, kwargs["out"])
        torch.testing.assert_close(output, torch.exp(args[0]))
        boolean = generate_tensor_input((32,), torch.bool, "cpu")
        self.assertEqual(boolean.dtype, torch.bool)
        self.assertEqual(boolean.shape, (32,))

    def test_unpack_matches_upstream_keyword_convention(self):
        value = torch.ones(2)
        args, kwargs = unpack_inputs((value, 1, {"keepdim": True}))
        self.assertEqual(args, (value, 1))
        self.assertEqual(kwargs, {"keepdim": True})
        self.assertEqual(unpack_inputs(({"size": (4, 8)},)), ((), {"size": (4, 8)}))

    def test_random_scalar_parameters_replay_without_polluting_host_rng(self):
        module = self.modules["pad"]
        binding = {"shape": [8, 8], "variant": 0}
        cpu_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        first = module.build_inputs(binding, torch.float32, "meta")[1]
        second = module.build_inputs(binding, torch.float32, "meta")[1]
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_state))


if __name__ == "__main__":
    unittest.main()
