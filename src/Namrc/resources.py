"""Measured process memory, free RAM guards, and sequential child jobs."""

import csv
from pathlib import Path
import subprocess
import sys
import time

import psutil
from src.training.resources import temporary_disk_bytes

def capacity_plan(pixels, pixel_chunk_size, sort_chunk_size, *, probability_bytes=4):
    # Conservative scratch allowance for copies, int64 labels/indices, masks,
    # and cleanlab's two-class intermediate arrays. Does not include torch/CUDA.
    return {
        'pixel_chunk_size': pixel_chunk_size, 'sort_chunk_size': sort_chunk_size,
        'estimated_scratch_mib': max(pixel_chunk_size * 128, sort_chunk_size * 40) / 1024**2,
        'estimated_audit_disk_bytes': pixels * (probability_bytes + 4),
        'estimated_auto_disk_bytes': pixels * 4,
        'estimated_oof_disk_bytes': pixels * (4 * probability_bytes + 3),
        'temporary_extra_disk_bytes': pixels * (probability_bytes + 4),
        'selection': 'Exact floating-point radix selection; no sorting runs or merge buffers.',
        'note': 'OOF estimate includes worst-case overlap canonical probabilities. Model checkpoints, previews, Test predictions, file page cache, interpreter, torch/CUDA and model activations are additional; runtime RSS is monitored.',
    }


class MemoryMonitor:
    def __init__(self, directory, budget_mib=4096, min_free_mib=1024):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'memory_log.csv'
        self.budget_mib, self.min_free_mib = budget_mib, min_free_mib
        self.process = psutil.Process()
        self.maximum, self.last_tick = 0., 0.
        if self.path.exists():
            with self.path.open(encoding='utf-8', newline='') as stream:
                for row in csv.DictReader(stream):
                    self.maximum = max(self.maximum, float(row['observed_max_rss_sum_mib']))

    def record(self, stage, tiles=0, pixels=0, chunk=0, *, force=False):
        now = time.monotonic()
        if not force and now - self.last_tick < 1:
            return
        self.last_tick = now
        own = self.process.memory_info().rss
        child_rss, uss, pss, children = 0, 0, 0, self.process.children(recursive=True)
        full_available = True
        for process in [self.process, *children]:
            try:
                if process.pid != self.process.pid:
                    child_rss += process.memory_info().rss
                detail = process.memory_full_info()
                uss += getattr(detail, 'uss', 0)
                pss += getattr(detail, 'pss', 0)
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                full_available = False
        rss = (own + child_rss) / 1024**2
        free = psutil.virtual_memory().available / 1024**2
        self.maximum = max(self.maximum, rss)
        temporary = temporary_disk_bytes(self.directory)
        row = {'time_unix': time.time(), 'stage': stage, 'tiles': tiles, 'pixels': pixels,
               'chunk_pixels': chunk, 'parent_rss_mib': own / 1024**2, 'child_rss_mib': child_rss / 1024**2,
               'rss_sum_mib': rss, 'uss_sum_mib': uss / 1024**2 if full_available and uss else '',
               'pss_sum_mib': pss / 1024**2 if full_available and pss else '',
               'system_free_ram_mib': free, 'observed_max_rss_sum_mib': self.maximum,
               'temporary_disk_bytes': temporary, 'child_count': len(children),
               'rss_note': 'RSS sum may double count shared pages; USS/PSS may be unavailable.'}
        exists = self.path.exists()
        with self.path.open('a', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        if rss > self.budget_mib:
            raise MemoryError(f'{stage}: measured RSS sum {rss:.1f} MiB exceeds budget {self.budget_mib} MiB.')
        if free < self.min_free_mib:
            raise MemoryError(f'{stage}: system free RAM {free:.1f} MiB is below {self.min_free_mib} MiB.')

    def check_disk(self, needed):
        available = psutil.disk_usage(str(self.directory)).free
        if needed > available:
            raise OSError(f'Insufficient temporary disk: need {needed:,} bytes; available {available:,}.')


def run_command(command, monitor, stage):
    """One process tree at a time; stop its workers as well if a RAM guard fails."""
    monitor.record(stage, force=True)
    process = subprocess.Popen(command)
    try:
        while process.poll() is None:
            monitor.record(stage)
            time.sleep(.25)
        if process.returncode:
            raise RuntimeError(f'{stage} worker failed with exit code {process.returncode}; see its error.json/log.')
    except BaseException:
        if process.poll() is None:
            try:
                descendants = psutil.Process(process.pid).children(recursive=True)
            except psutil.NoSuchProcess:
                descendants = []
            for child in reversed(descendants):
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        raise
    else:
        monitor.record(stage + '_finished', force=True)


def run_job(job_path, monitor, stage):
    """Output is inherited and no tensors cross processes."""
    run_command([sys.executable, '-X', 'utf8', '-u', '-m', 'src.Namrc.worker', str(job_path)], monitor, stage)
