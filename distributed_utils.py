from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    enabled: bool
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def init_distributed() -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    enabled = world_size > 1

    if enabled and not dist.is_initialized():
        dist.init_process_group(backend="nccl")

    if torch.cuda.is_available():
        if enabled:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    return DistributedContext(
        enabled=enabled,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=device,
    )


def cleanup_distributed(ctx: DistributedContext) -> None:
    if ctx.enabled and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def rank_zero_print(ctx: DistributedContext, *args: Any, **kwargs: Any) -> None:
    if ctx.is_main:
        print(*args, **kwargs)


def split_count(total: int, rank: int, world_size: int) -> int:
    if world_size <= 1:
        return total
    if total <= 0:
        return 0
    base = total // world_size
    rem = total % world_size
    return base + (1 if rank < rem else 0)


def all_gather_objects(ctx: DistributedContext, obj: Any) -> list[Any]:
    if not ctx.enabled:
        return [obj]
    gathered: list[Any] = [None] * ctx.world_size
    dist.all_gather_object(gathered, obj)
    return gathered


def broadcast_object(ctx: DistributedContext, obj: Any, src: int = 0) -> Any:
    if not ctx.enabled:
        return obj
    payload = [obj]
    dist.broadcast_object_list(payload, src=src)
    return payload[0]
