# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Disk I/O backend for GPU<->NVMe block transfers via pinned staging buffers.

Uses separate IO threads for store and load so that loads (latency-critical)
never block behind stores (background work). Each thread owns its own pinned
staging buffers to avoid contention. Slot I/O uses preadv/pwritev, or
mooncake.uring batches when VLLM_KV_OFFLOAD_DISK_IO_BACKEND=io_uring.
"""

from __future__ import annotations

import contextlib
import itertools
import os
import queue
import threading
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.tiering.fs.io import load_uring_module
from vllm.v1.simple_kv_offload.cuda_mem_ops import (
    CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
    CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
    BatchMemcpyParams,
    build_params,
    copy_blocks,
    pin_tensor,
)

logger = init_logger(__name__)

O_DIRECT = getattr(os, "O_DIRECT", 0)
_ALIGNMENT = 4096


def _alloc_aligned_regions(
    region_bytes: list[int],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Carve O_DIRECT-aligned staging regions out of one CPU allocation.

    The CPU allocator only guarantees 64-byte alignment, so over-allocate by
    one alignment unit and align the base; every region is padded to the
    alignment so each region base is aligned too. One allocation lets the
    whole staging area be pinned and registered with io_uring once. The views
    keep the backing storage alive.
    """
    padded = [-(-nbytes // _ALIGNMENT) * _ALIGNMENT for nbytes in region_bytes]
    raw = torch.zeros(sum(padded) + _ALIGNMENT, dtype=torch.int8, device="cpu")
    base = -raw.data_ptr() % _ALIGNMENT
    arena = raw[base : base + sum(padded)]
    regions = [
        arena[offset : offset + nbytes]
        for nbytes, offset in zip(region_bytes, itertools.accumulate(padded, initial=0))
    ]
    return arena, regions


class DiskBackend:
    """Async disk offload backend with pipelined GPU DMA and interleaved IO.

    Architecture:
    - Separate coordinator threads for store and load (never block each other)
    - Interleaved pipeline: DMA slot N while preadv/pwritev slot N-1
    - O_DIRECT by default; page cache is opt-in via use_page_cache
    - Optional io_uring slot I/O through mooncake.uring, with the staging
      slots registered once as fixed buffers

    Same launch_copy interface as DmaCopyBackend so the worker can swap
    backends without changing calling code.
    """

    def __init__(self) -> None:
        self._store_params: BatchMemcpyParams | None = None
        self._load_params: BatchMemcpyParams | None = None
        self._load_stream: torch.cuda.Stream | None = None
        self._store_stream: torch.cuda.Stream | None = None
        self._store_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._load_queue: queue.SimpleQueue = queue.SimpleQueue()
        self._store_thread: threading.Thread | None = None
        self._load_thread: threading.Thread | None = None
        self._shutdown: bool = False
        self._fd: int = -1
        self._disk_path: str = ""
        self._total_block_bytes: int = 0
        self._store_buffer_caches: dict[str, torch.Tensor] = {}
        self._load_buffer_caches: dict[str, torch.Tensor] = {}
        self._store_slot_views: list[list[memoryview]] = []
        self._load_slot_views: list[list[memoryview]] = []
        self._per_tensor_bpb: list[int] = []
        self._tensor_names: list[str] = []
        self._staging_arena: torch.Tensor | None = None
        self._store_slot_ptrs: list[list[int]] = []
        self._load_slot_ptrs: list[list[int]] = []
        self._tensor_file_offsets: list[int] = []
        self._uring: Any = None
        self._uring_store_file: Any = None
        self._uring_load_file: Any = None

    def init(
        self,
        gpu_caches: dict[str, torch.Tensor],
        device: torch.device,
        load_stream: torch.cuda.Stream,
        store_stream: torch.cuda.Stream,
        disk_path: str,
        num_disk_slots: int,
        total_block_bytes: int,
        num_buffer_slots: int = 2,
        use_page_cache: bool = False,
    ) -> None:
        self._load_stream = load_stream
        self._store_stream = store_stream
        self._total_block_bytes = total_block_bytes
        self._num_buffer_slots = num_buffer_slots
        self._tensor_names = list(gpu_caches.keys())
        self._per_tensor_bpb = [
            t.stride(0) * t.element_size() for t in gpu_caches.values()
        ]

        assert total_block_bytes % _ALIGNMENT == 0, (
            f"total_block_bytes={total_block_bytes} not aligned to {_ALIGNMENT}"
        )

        # Separate buffer pools for store and load threads, carved from one
        # allocation so the staging area is pinned and registered once.
        num_tensors = len(self._tensor_names)
        self._staging_arena, regions = _alloc_aligned_regions(
            [num_buffer_slots * bpb for bpb in self._per_tensor_bpb] * 2
        )
        pin_tensor(self._staging_arena)
        self._store_buffer_caches = {
            name: region.view(num_buffer_slots, bpb)
            for name, bpb, region in zip(
                self._tensor_names, self._per_tensor_bpb, regions[:num_tensors]
            )
        }
        self._load_buffer_caches = {
            name: region.view(num_buffer_slots, bpb)
            for name, bpb, region in zip(
                self._tensor_names, self._per_tensor_bpb, regions[num_tensors:]
            )
        }

        # Pre-built iovec views per slot (avoid per-transfer .numpy() calls)
        self._store_slot_views = [
            [
                memoryview(self._store_buffer_caches[name][slot].numpy())
                for name in self._tensor_names
            ]
            for slot in range(num_buffer_slots)
        ]
        self._load_slot_views = [
            [
                memoryview(self._load_buffer_caches[name][slot].numpy())
                for name in self._tensor_names
            ]
            for slot in range(num_buffer_slots)
        ]
        # io_uring descriptors: one (ptr, len, file offset) per tensor per slot.
        self._store_slot_ptrs = [
            [self._store_buffer_caches[name][slot].data_ptr() for name in gpu_caches]
            for slot in range(num_buffer_slots)
        ]
        self._load_slot_ptrs = [
            [self._load_buffer_caches[name][slot].data_ptr() for name in gpu_caches]
            for slot in range(num_buffer_slots)
        ]
        self._tensor_file_offsets = list(
            itertools.accumulate(self._per_tensor_bpb[:-1], initial=0)
        )

        self._store_params = build_params(
            gpu_caches,
            self._store_buffer_caches,
            store_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_STREAM,
        )
        self._load_params = build_params(
            self._load_buffer_caches,
            gpu_caches,
            load_stream,
            src_access_order=CU_MEMCPY_SRC_ACCESS_ORDER_ANY,
        )

        os.makedirs(os.path.dirname(disk_path) or ".", exist_ok=True)
        # Slot contents never outlive the process, so unlink then O_EXCL rather
        # than reopening: a pre-existing file would otherwise keep its own
        # (possibly world-readable) mode, and blocks may encode user prompts.
        with contextlib.suppress(FileNotFoundError):
            os.unlink(disk_path)
        # O_DIRECT by default: page cache would consume the very host DRAM this
        # backend exists to conserve, and doubles the copy on the store path.
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
        if not use_page_cache:
            flags |= O_DIRECT
        self._fd = os.open(disk_path, flags, 0o600)
        self._disk_path = disk_path
        os.ftruncate(self._fd, num_disk_slots * total_block_bytes)
        self._init_uring(disk_path, use_page_cache)

        logger.info(
            "DiskBackend: path=%s, slots=%d, total=%.2f GB, buf=%dx%d bytes"
            " (page_cache=%s, io=%s)",
            disk_path,
            num_disk_slots,
            (num_disk_slots * total_block_bytes) / (1024**3),
            num_buffer_slots,
            total_block_bytes,
            use_page_cache,
            "io_uring" if self._uring_store_file is not None else "posix",
        )

        self._store_thread = threading.Thread(
            target=self._store_loop,
            args=(device, store_stream),
            daemon=True,
        )
        self._load_thread = threading.Thread(
            target=self._load_loop,
            args=(device, load_stream),
            daemon=True,
        )
        self._store_thread.start()
        self._load_thread.start()

    def launch_copy(
        self,
        src_blocks: list[int],
        dst_blocks: list[int],
        is_store: bool,
        event_idx: int,
        events_list: list[tuple[int, torch.Event]],
        wait_event: torch.Event | None = None,
    ) -> None:
        q = self._store_queue if is_store else self._load_queue
        q.put((src_blocks, dst_blocks, event_idx, events_list, wait_event))

    def shutdown(self) -> None:
        if self._shutdown:
            return
        self._shutdown = True
        self._store_queue.put(None)
        self._load_queue.put(None)
        if self._store_thread is not None:
            self._store_thread.join(timeout=10.0)
        if self._load_thread is not None:
            self._load_thread.join(timeout=10.0)
        if self._fd < 0:
            return
        # Slot contents can encode user prompts, so drop the name now rather
        # than leaving them readable until the next run overwrites the file.
        # Unlinking only removes the directory entry: any thread still holding
        # the fd keeps writing to the (now anonymous) inode, which the kernel
        # frees once the last fd goes away.
        with contextlib.suppress(OSError):
            os.unlink(self._disk_path)
        # Closing under a still-running IO thread would let the fd number be
        # reused by an unrelated open(), turning its next pwritev into a write
        # into that file. Leaking one fd for the remaining process lifetime is
        # the cheaper failure.
        if any(
            t is not None and t.is_alive()
            for t in (self._store_thread, self._load_thread)
        ):
            logger.warning(
                "IO thread still running after shutdown timeout; leaking fd %d",
                self._fd,
            )
            return
        self._close_uring()
        os.close(self._fd)
        self._fd = -1

    def _init_uring(self, disk_path: str, use_page_cache: bool) -> None:
        """Open io_uring handles on the slot file when that backend is
        selected; any failure (missing module, seccomp, old kernel) keeps the
        POSIX preadv/pwritev path."""
        uring = load_uring_module()
        if uring is None:
            return
        queue_depth = max(32, len(self._tensor_names))
        try:
            self._uring_store_file = uring.UringFile(
                disk_path,
                os.O_WRONLY | os.O_NOFOLLOW,
                queue_depth=queue_depth,
                direct_io=not use_page_cache,
            )
            self._uring_load_file = uring.UringFile(
                disk_path,
                os.O_RDONLY | os.O_NOFOLLOW,
                queue_depth=queue_depth,
                direct_io=not use_page_cache,
            )
        except Exception as exc:
            logger.warning(
                "mooncake.uring could not open %s (%s); using POSIX preadv/pwritev.",
                disk_path,
                exc,
            )
            self._close_uring()
            return
        assert self._staging_arena is not None
        if uring.register_global_buffer(
            self._staging_arena.data_ptr(), self._staging_arena.nbytes
        ):
            self._uring = uring
        else:
            logger.warning(
                "mooncake.uring could not register the staging buffers as fixed "
                "buffers; using unregistered io_uring buffers."
            )

    def _close_uring(self) -> None:
        for name in ("_uring_store_file", "_uring_load_file"):
            uring_file = getattr(self, name)
            if uring_file is not None:
                uring_file.close()
                setattr(self, name, None)
        if self._uring is not None:
            self._uring.unregister_global_buffer()
            self._uring = None

    def _uring_slot_io(self, op: Any, slot_ptrs: list[int], file_offset: int) -> int:
        """Submit one io_uring batch covering every tensor of a slot."""
        offsets = [file_offset + offset for offset in self._tensor_file_offsets]
        total = 0
        for ret in op(slot_ptrs, self._per_tensor_bpb, offsets):
            if ret < 0:
                raise OSError(-ret, os.strerror(-ret))
            total += ret
        return total

    def _store_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._store_queue.get()
            if item is None:
                return
            (src_blocks, dst_blocks, event_idx, events_list, wait_event) = item
            if wait_event is not None:
                stream.wait_event(wait_event)
            self._do_store(src_blocks, dst_blocks, stream)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _writev_slot(self, buf_slot: int, file_offset: int) -> None:
        if self._uring_store_file is not None:
            written = self._uring_slot_io(
                self._uring_store_file.batch_write,
                self._store_slot_ptrs[buf_slot],
                file_offset,
            )
        else:
            written = os.pwritev(
                self._fd, self._store_slot_views[buf_slot], file_offset
            )
        if written < self._total_block_bytes:
            raise OSError(
                f"Short write: expected {self._total_block_bytes} bytes, "
                f"wrote {written}"
            )

    def _readv_slot(self, buf_slot: int, file_offset: int) -> None:
        if self._uring_load_file is not None:
            bytes_read = self._uring_slot_io(
                self._uring_load_file.batch_read,
                self._load_slot_ptrs[buf_slot],
                file_offset,
            )
        else:
            bytes_read = os.preadv(
                self._fd, self._load_slot_views[buf_slot], file_offset
            )
        if bytes_read < self._total_block_bytes:
            raise OSError(
                f"Short read: expected {self._total_block_bytes} bytes, "
                f"read {bytes_read}"
            )

    def _load_loop(
        self,
        device: torch.device,
        stream: torch.cuda.Stream,
    ) -> None:
        current_platform.set_device(device)
        while True:
            item = self._load_queue.get()
            if item is None:
                return
            (src_blocks, dst_blocks, event_idx, events_list, wait_event) = item
            if wait_event is not None:
                stream.wait_event(wait_event)
            self._do_load(src_blocks, dst_blocks, stream)
            event = torch.Event()
            event.record(stream)
            events_list.append((event_idx, event))

    def _do_store(
        self,
        gpu_blocks: list[int],
        disk_slots: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """GPU -> buffer (DMA) -> disk (pwritev), interleaved double-buffer."""
        assert self._store_params is not None
        n = self._num_buffer_slots
        # (DMA event, file offset) of the block already staged in each slot.
        pending: list[tuple[torch.Event, int] | None] = [None] * n

        for i, (gpu_blk, disk_slot) in enumerate(zip(gpu_blocks, disk_slots)):
            buf_slot = i % n
            prev = pending[buf_slot]
            if prev is not None:
                prev[0].synchronize()
                self._writev_slot(buf_slot, prev[1])

            copy_blocks([gpu_blk], [buf_slot], self._store_params)
            ev = torch.Event()
            ev.record(stream)
            pending[buf_slot] = (ev, disk_slot * self._total_block_bytes)

        for slot, last in enumerate(pending):
            if last is not None:
                last[0].synchronize()
                self._writev_slot(slot, last[1])

    def _do_load(
        self,
        disk_slots: list[int],
        gpu_blocks: list[int],
        stream: torch.cuda.Stream,
    ) -> None:
        """Disk (preadv) -> buffer -> GPU (DMA), interleaved double-buffer."""
        assert self._load_params is not None
        n = self._num_buffer_slots
        prev_dma_events: list[torch.Event | None] = [None] * n

        for i, (disk_slot, gpu_blk) in enumerate(zip(disk_slots, gpu_blocks)):
            buf_slot = i % n
            prev = prev_dma_events[buf_slot]
            if prev is not None:
                prev.synchronize()

            self._readv_slot(buf_slot, disk_slot * self._total_block_bytes)

            copy_blocks([buf_slot], [gpu_blk], self._load_params)
            ev = torch.Event()
            ev.record(stream)
            prev_dma_events[buf_slot] = ev
