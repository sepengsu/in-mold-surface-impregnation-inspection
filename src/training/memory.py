"""Bound CUDA allocations and record measured VRAM without retaining tensors."""

import csv
from pathlib import Path
import time


class CudaMemoryGuard:
    def __init__(self, device, directory=None, budget_mib=0, min_free_mib=0):
        import torch
        self.device = device
        self.active = device.type == 'cuda'
        self.budget_mib, self.min_free_mib = float(budget_mib), float(min_free_mib)
        self.path = Path(directory) / 'vram_log.csv' if directory else None
        self.last_tick = 0.
        self.peak_reserved_mib = 0.
        self.limit_mib = None
        if min(self.budget_mib, self.min_free_mib) < 0:
            raise ValueError('VRAM limits must be nonnegative.')
        if self.active:
            free, total = torch.cuda.mem_get_info(device)
            if free / 1024**2 <= self.min_free_mib:
                raise MemoryError('Insufficient free VRAM before model allocation. Close other GPU applications.')
            if self.budget_mib:
                self.limit_mib = min(self.budget_mib, free / 1024**2 - self.min_free_mib)
                index = device.index if device.index is not None else torch.cuda.current_device()
                torch.cuda.set_per_process_memory_fraction(self.limit_mib * 1024**2 / total, index)
            self.check('initialize', force=True)

    def check(self, stage, *, force=False):
        if not self.active:
            return
        import torch
        free, total = torch.cuda.mem_get_info(self.device)
        allocated = torch.cuda.memory_allocated(self.device) / 1024**2
        reserved = torch.cuda.memory_reserved(self.device) / 1024**2
        self.peak_reserved_mib = max(self.peak_reserved_mib, torch.cuda.max_memory_reserved(self.device) / 1024**2)
        now = time.monotonic()
        if self.path and (force or now - self.last_tick >= 2):
            self.last_tick = now
            self.path.parent.mkdir(parents=True, exist_ok=True)
            row = dict(time_unix=time.time(), stage=stage, allocated_mib=allocated,
                       reserved_mib=reserved, peak_reserved_mib=self.peak_reserved_mib,
                       device_free_mib=free / 1024**2, device_total_mib=total / 1024**2,
                       allocator_limit_mib=self.limit_mib, min_free_mib=self.min_free_mib)
            exists = self.path.exists()
            with self.path.open('a', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(row))
                if not exists:
                    writer.writeheader()
                writer.writerow(row)
        if self.budget_mib and reserved > self.budget_mib + 1:
            raise MemoryError(f'{stage}: reserved VRAM exceeds {self.budget_mib:.0f} MiB.')
        if free / 1024**2 < self.min_free_mib:
            raise MemoryError(f'{stage}: free VRAM below {self.min_free_mib:.0f} MiB. Close other GPU applications.')
