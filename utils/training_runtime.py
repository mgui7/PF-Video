"""Synchronized clean stops for time-limited distributed training allocations."""
import time

import torch
import torch.distributed as dist


class TrainingDeadline:
    def __init__(self, seconds, started_at=None, check_every=100):
        start = time.monotonic() if started_at is None else started_at
        self.deadline = start + seconds if seconds > 0 else None
        self.check_every = check_every
        self.stopped = False

    def should_stop(self, global_step, device):
        if self.stopped:
            return True
        if self.deadline is None or global_step % self.check_every:
            return False
        distributed = dist.is_available() and dist.is_initialized()
        is_leader = not distributed or dist.get_rank() == 0
        expired = is_leader and time.monotonic() >= self.deadline
        decision = torch.tensor(int(expired), device=device)
        if distributed:
            dist.broadcast(decision, src=0)
        self.stopped = bool(decision.item())
        return self.stopped
