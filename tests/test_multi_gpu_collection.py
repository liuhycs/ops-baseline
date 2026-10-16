# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise scheduling and recovery without CUDA or torch imports."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'parallel', Path(__file__).resolve().parents[1] / 'tools/collect_baseline_multi_gpu.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class CollectionTests(unittest.TestCase):
    def args(self, directory, **kwargs):
        values = dict(devices='0,1', repo='pytorch', output=str(Path(directory)/'data.json'),
                      ops='abs,add,mul', whitelist=None, blacklist=None, no_ncu=False,
                      op_timeout=5, reference_chip='H800', resume=False)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_dynamic_queue_isolation_merge_and_resume(self):
        calls = []
        active = set()
        lock = threading.Lock()
        simultaneous = threading.Event()
        def execute(command, environment, log, timeout, stop):
            name = command[command.index('--ops')+1]
            device = environment['CUDA_VISIBLE_DEVICES']
            with lock:
                self.assertNotIn(device, active)
                active.add(device)
                calls.append((name, device))
                if len(active) == 2:
                    simultaneous.set()
            self.assertTrue(simultaneous.wait(3))
            time.sleep(.03)
            out = command[command.index('--output')+1]
            m.atomic_json(out, {name: {'shapes': {'[1]': {'fp32': {'num_kernels': 1}}}},
                                '_environment': {'device': device},
                                '_scaling_factors': {'reference_chip': 'H800'}})
            with lock:
                active.remove(device)
            return 0
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = self.args(directory)
            self.assertEqual(m.collect(args, execute=execute), 0)
            self.assertEqual(sorted(n for n, _ in calls), ['abs', 'add', 'mul'])
            data = json.loads(Path(args.output).read_text())
            self.assertEqual(set(data['_environments']), {'0', '1'})
            self.assertEqual(len({t['result'] for t in data['_collection']['tasks'].values()}), 3)
            with self.assertRaisesRegex(ValueError, 'output exists'):
                m.collect(args, execute=execute)
            args.resume = True
            self.assertEqual(m.collect(args, execute=execute), 0)
            self.assertEqual(len(calls), 3)
            args.no_ncu = True
            with self.assertRaisesRegex(ValueError, 'configuration'):
                m.collect(args, execute=execute)

    def test_failures_and_missing_profiles_are_not_completed(self):
        def execute(command, environment, log, timeout, stop):
            name = command[command.index('--ops')+1]
            out = command[command.index('--output')+1]
            if name == 'mul':
                return 'timeout'
            m.atomic_json(out, {name: {'shapes': {'[1]': {'fp32': {'num_kernels': 0}}}}})
            return 1 if name == 'abs' else 0
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = self.args(directory)
            self.assertEqual(m.collect(args, execute=execute), 1)
            tasks = json.loads(Path(args.output).read_text())['_collection']['tasks']
            self.assertEqual(tasks['abs']['status'], 'failed')
            self.assertEqual(tasks['add']['status'], 'profiling_incomplete')
            self.assertEqual(tasks['mul']['returncode'], 'timeout')

    def test_device_and_operator_validation(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,5'}, clear=True):
            self.assertEqual(m.parse_devices('2,5'), ['2', '5'])
            for value in ['2,2', '2,', '0,1']:
                with self.assertRaises(ValueError):
                    m.parse_devices(value)
        with self.assertRaises(ValueError):
            m.select_operators('pytorch', {'does-not-exist'})

    def test_long_operator_selection_is_not_a_filename(self):
        names = ['operator_' + str(i) for i in range(100)]
        self.assertEqual(m.name_list(','.join(names)), set(names))

    def test_timeout_kills_descendants(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory)/'child.pid'
            code = ("import subprocess,sys,time; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                    f"Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)")
            result = m.run_process([sys.executable, '-c', code], dict(os.environ),
                                   Path(directory)/'log', 1, threading.Event())
            self.assertEqual(result, 'timeout')
            pid = int(pidfile.read_text())
            # A killed child may briefly remain as an OS zombie; it must not run.
            state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                                   capture_output=True, text=True).stdout.strip()
            self.assertTrue(not state or state.startswith('Z'), state)


if __name__ == '__main__':
    unittest.main()
