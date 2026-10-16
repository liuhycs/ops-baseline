#!/usr/bin/env python3
# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Dynamically schedule isolated operator collectors, one consumer per GPU."""
from __future__ import annotations

import argparse
import fcntl
import math
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def parse_devices(value):
    devices = [v.strip() for v in value.split(',')]
    if not all(devices) or len(devices) != len(set(devices)):
        raise ValueError('devices must be nonempty and unique')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is not None and not set(devices).issubset(visible.split(',')):
        raise ValueError('--devices must select tokens from inherited CUDA_VISIBLE_DEVICES')
    return devices


def name_list(value):
    if value is None:
        return None
    path = Path(value)
    content = path.read_text() if path.is_file() else value.replace(',', '\n')
    return {s for line in content.splitlines() if (s := line.split('#')[0].strip())}


def select_operators(repo, only=None, whitelist=None, blacklist=None):
    directory = ROOT / 'ops' / repo
    if not directory.is_dir() or directory.parent.resolve() != (ROOT / 'ops').resolve():
        raise ValueError(f'invalid operator repo: {repo}')
    names = sorted(p.stem for p in directory.glob('*.py') if p.stem != '__init__')
    for requested in (only, whitelist, blacklist):
        unknown = (requested or set()) - set(names)
        if unknown:
            raise ValueError(f'unknown operator modules: {sorted(unknown)}')
    return [n for n in names if (only is None or n in only)
            and (whitelist is None or n in whitelist) and n not in (blacklist or set())]


def source_digest(repo):
    digest = hashlib.sha256()
    paths = sorted((ROOT / 'ops' / repo).rglob('*.py'))
    paths += sorted((ROOT / 'ops' / repo).rglob('*.json'))
    paths += sorted((ROOT / 'tools').glob('*.py'))
    paths += [ROOT / 'tools' / 'hardware_specs.json']
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_process(command, environment, log_path, timeout, stop):
    """Terminate the entire group, including NCU and its Python child."""
    with log_path.open('w') as log:
        process = subprocess.Popen(command, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + timeout
        try:
            while process.poll() is None:
                if stop.wait(.2) or time.monotonic() >= deadline:
                    return 'interrupted' if stop.is_set() else 'timeout'
            return process.returncode
        finally:
            # NCU descendants can outlive their direct parent after a crash.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def collect(args, *, execute=run_process):
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with (output.with_suffix(output.suffix + '.lock')).open('a') as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('another collector owns this output') from exc
        return _collect(args, execute=execute)


def _collect(args, *, execute=run_process):
    if not math.isfinite(args.op_timeout) or args.op_timeout <= 0:
        raise ValueError("--op-timeout must be finite and positive")
    devices = parse_devices(args.devices)
    names = select_operators(args.repo, name_list(args.ops), name_list(args.whitelist),
                             name_list(args.blacklist))
    if not names:
        raise ValueError('no operators selected')
    output = Path(args.output).resolve()
    work = output.with_suffix(output.suffix + '.work')
    signature = {'repo': args.repo, 'operators': names, 'devices': devices,
                 'ncu_enabled': not args.no_ncu, 'op_timeout': args.op_timeout,
                 'reference_chip': args.reference_chip, 'source_sha256': source_digest(args.repo),
                 'report_dir': str(Path(args.report_dir).resolve()) if getattr(args, 'report_dir', None) else None}
    if output.exists() and not args.resume:
        raise ValueError('output exists; choose another path or use --resume')
    if args.resume:
        if not output.exists():
            raise ValueError('--resume requires an existing output')
        results = json.loads(output.read_text())
        if results.get('_collection', {}).get('configuration') != signature:
            raise ValueError('resume configuration/source differs from the existing run')
    else:
        results = {'_collection': {'configuration': signature,
                   'started_at_utc': datetime.now(timezone.utc).isoformat(), 'tasks': {}},
                   '_environments': {}}
    tasks = results['_collection']['tasks']
    pending = queue.Queue()
    for name in names:
        if tasks.get(name, {}).get('status') != 'completed':
            pending.put(name)
    work.mkdir(parents=True, exist_ok=True)
    atomic_json(output, results)
    stop = threading.Event()
    events = queue.Queue()

    def consume(device):
        while not stop.is_set():
            try:
                name = pending.get_nowait()
            except queue.Empty:
                return
            folder = work / name
            folder.mkdir(parents=True, exist_ok=True)
            part = folder / 'result.json'
            part.unlink(missing_ok=True)
            command = [sys.executable, str(ROOT / 'tools' / 'collect_baseline_nvidia.py'),
                       '--repo', args.repo, '--ops', name, '--output', str(part),
                       '--report-dir', str((Path(args.report_dir).resolve() / output.name / name)
                                           if getattr(args, 'report_dir', None) else folder / 'ncu_reports'),
                       '--op-timeout', str(args.op_timeout), '--reference-chip', args.reference_chip]
            if args.no_ncu:
                command.append('--no-ncu')
            environment = dict(os.environ, CUDA_VISIBLE_DEVICES=device, PYTHONUNBUFFERED='1')
            events.put((name, {'status': 'running', 'device': device}, None))
            try:
                code = execute(command, environment, folder / 'collector.log',
                               args.op_timeout + 120, stop)
                data = json.loads(part.read_text()) if part.exists() else {}
                entry = data.get(name)
                status = 'failed' if code != 0 else ('skipped' if entry is None else 'completed')
                if status == 'completed':
                    records = [r for shape in entry.get('shapes', {}).values() for r in shape.values()]
                    if not records:
                        status = 'skipped'
                    elif not args.no_ncu and any(not r.get('num_kernels') for r in records):
                        status = 'profiling_incomplete'
                record = {'status': status, 'device': device, 'returncode': code,
                          'log': str(folder / 'collector.log'), 'result': str(part)}
                events.put((name, record, data))
            except Exception as exc:
                events.put((name, {'status': 'failed', 'device': device,
                                  'error': f'{type(exc).__name__}: {exc}'}, {}))

    previous = {}
    def interrupted(signum, frame):
        stop.set()
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous[signum] = signal.signal(signum, interrupted)
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(consume, device) for device in devices]
            while not all(f.done() for f in futures) or not events.empty():
                try:
                    name, record, data = events.get(timeout=.2)
                except queue.Empty:
                    continue
                tasks[name] = record
                if data is not None:
                    if name in data:
                        results[name] = data[name]
                    else:
                        results.pop(name, None)
                    if '_environment' in data:
                        results['_environments'][record['device']] = data['_environment']
                        results.setdefault('_environment', data['_environment'])
                    if '_scaling_factors' in data:
                        results['_scaling_factors'] = data['_scaling_factors']
                    print(f"[{record['device']}] {name}: {record['status']}", flush=True)
                atomic_json(output, results)
            for future in futures:
                future.result()
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)
    results['_collection']['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
    results['_collection']['interrupted'] = stop.is_set()
    atomic_json(output, results)
    return int(stop.is_set() or any(tasks.get(n, {}).get('status') != 'completed' for n in names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--devices', required=True, help='GPU visibility tokens, e.g. 0,1,2,3')
    parser.add_argument('--repo', default='pytorch')
    parser.add_argument('--output', required=True)
    parser.add_argument('--report-dir', help='Optional report root (partitioned by output and operator)')
    parser.add_argument('--ops')
    parser.add_argument('--whitelist')
    parser.add_argument('--blacklist')
    parser.add_argument('--op-timeout', type=float, default=1800)
    parser.add_argument('--reference-chip', default='H800')
    parser.add_argument('--no-ncu', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.op_timeout <= 0:
        parser.error('--op-timeout must be positive')
    try:
        return collect(args)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    sys.exit(main())
