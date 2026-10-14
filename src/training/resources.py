"""Measured process memory, free RAM guards, and sequential child jobs."""

import csv
import math
from pathlib import Path
import subprocess
import sys
import time
import stat

import psutil


def temporary_disk_bytes(directory):
    """A writer may atomically rename a checkpoint after rglob yields its name."""
    total = 0
    for path in Path(directory).rglob('*.tmp'):
        try:
            info = path.stat()
        except FileNotFoundError:
            continue
        if stat.S_ISREG(info.st_mode):
            total += info.st_size
    return total

class MemoryMonitor:
    def __init__(self, directory, budget_mib=4096, min_free_mib=1024):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / 'memory_log.csv'
        self.budget_mib, self.min_free_mib = budget_mib, min_free_mib
        self.process = psutil.Process()
        self.maximum, self.last_tick = 0., 0.
        self.ignored_log_rows = 0
        if self.path.exists():
            with self.path.open(encoding='utf-8', errors='replace', newline='') as stream:
                for row in csv.DictReader(stream):
                    # An interrupted append can leave a partial or NUL-filled
                    # row. Preserve the file and recover only valid samples.
                    invalid = (None in row or any(value is None or '\x00' in value or '\ufffd' in value
                                                  for value in row.values()))
                    try:
                        maximum = float(row.get('observed_max_rss_sum_mib'))
                    except (TypeError, ValueError):
                        invalid = True
                        maximum = 0.
                    if invalid or not math.isfinite(maximum) or maximum < 0:
                        self.ignored_log_rows += 1
                        continue
                    self.maximum = max(self.maximum, maximum)
            if self.ignored_log_rows:
                print(f'Memory log recovery: ignored {self.ignored_log_rows} incomplete/invalid historical row(s) '
                      f'in {self.path}; valid measurements and original log retained.', file=sys.stderr, flush=True)

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
        # Separate a damaged, unterminated tail from the next measurement.
        # Append only: old measurements and evidence of the damaged row remain.
        with self.path.open('a+b') as stream:
            stream.seek(0, 2)
            exists = stream.tell() > 0
            if exists:
                stream.seek(-1, 2)
                if stream.read(1) not in (b'\r', b'\n'):
                    stream.write(b'\n')
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

