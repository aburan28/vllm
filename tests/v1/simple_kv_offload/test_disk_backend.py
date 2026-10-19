# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only tests for DiskBackend slot I/O on the POSIX and io_uring backends."""

import errno
import logging
from unittest.mock import MagicMock

import pytest
import torch

from tests.v1.kv_offload.fake_uring import install_fake_uring, make_fake_uring
from vllm.v1.kv_offload.tiering.fs.io import probe_o_direct
from vllm.v1.simple_kv_offload import disk_backend as disk_backend_module
from vllm.v1.simple_kv_offload.disk_backend import (
    _ALIGNMENT,
    DiskBackend,
    _alloc_aligned_regions,
)

_BPBS = (4096, 8192)
_TOTAL_BLOCK_BYTES = sum(_BPBS)


@pytest.fixture
def make_backend(tmp_path, monkeypatch):
    """Build DiskBackends without CUDA: pinning, batch-memcpy setup and the
    IO threads' device binding are stubbed; streams are mocks."""
    monkeypatch.setattr(disk_backend_module, "pin_tensor", lambda tensor: None)
    monkeypatch.setattr(disk_backend_module, "build_params", MagicMock())
    monkeypatch.setattr(disk_backend_module, "current_platform", MagicMock())
    backends: list[DiskBackend] = []

    def _make(use_page_cache: bool = True) -> DiskBackend:
        backend = DiskBackend()
        gpu_caches = {
            f"layer{i}": torch.zeros((4, bpb), dtype=torch.int8)
            for i, bpb in enumerate(_BPBS)
        }
        backend.init(
            gpu_caches,
            torch.device("cpu"),
            MagicMock(),
            MagicMock(),
            str(tmp_path / f"kv{len(backends)}" / "slots.bin"),
            num_disk_slots=3,
            total_block_bytes=_TOTAL_BLOCK_BYTES,
            use_page_cache=use_page_cache,
        )
        backends.append(backend)
        return backend

    yield _make
    for backend in backends:
        backend.shutdown()


def _fill_store_slot(backend: DiskBackend, slot: int, seed: int) -> None:
    for i, name in enumerate(backend._tensor_names):
        backend._store_buffer_caches[name][slot].fill_(seed + i)


def _assert_slot_round_trip(backend: DiskBackend) -> None:
    """Write store slot 1 to disk slot 2 and read it back into load slot 0."""
    _fill_store_slot(backend, 1, 7)
    file_offset = 2 * _TOTAL_BLOCK_BYTES
    backend._writev_slot(1, file_offset)
    backend._readv_slot(0, file_offset)
    for name in backend._tensor_names:
        assert torch.equal(
            backend._load_buffer_caches[name][0],
            backend._store_buffer_caches[name][1],
        )


def test_alloc_aligned_regions_aligns_every_region():
    arena, regions = _alloc_aligned_regions([2 * _ALIGNMENT, 6000, _ALIGNMENT])

    assert arena.data_ptr() % _ALIGNMENT == 0
    assert arena.numel() == 5 * _ALIGNMENT
    assert [r.numel() for r in regions] == [2 * _ALIGNMENT, 6000, _ALIGNMENT]
    assert [r.data_ptr() - arena.data_ptr() for r in regions] == [
        0,
        2 * _ALIGNMENT,
        4 * _ALIGNMENT,
    ]


def test_uring_slot_io_round_trip(make_backend, monkeypatch):
    """With the io_uring backend every slot write/read is one mooncake.uring
    batch over the per-tensor staging pointers, and the staging arena is
    registered once as the fixed buffer."""
    fake = make_fake_uring()
    install_fake_uring(monkeypatch, fake)

    backend = make_backend()

    assert backend._uring is fake
    arena = backend._staging_arena
    assert arena.data_ptr() % _ALIGNMENT == 0
    assert fake.registered == [(arena.data_ptr(), arena.nbytes)]
    assert [f.direct_io for f in fake.files] == [False, False]
    assert all(f.queue_depth >= 32 for f in fake.files)

    _assert_slot_round_trip(backend)

    file_offset = 2 * _TOTAL_BLOCK_BYTES
    offsets = [file_offset, file_offset + _BPBS[0]]
    assert fake.batches == [
        ("write", backend._store_slot_ptrs[1], list(_BPBS), offsets),
        ("read", backend._load_slot_ptrs[0], list(_BPBS), offsets),
    ]


def test_uring_io_errors_surface_as_oserror(make_backend, monkeypatch):
    fake = make_fake_uring()
    install_fake_uring(monkeypatch, fake)
    backend = make_backend()
    fake.fail_errno = errno.EIO

    with pytest.raises(OSError) as excinfo:
        backend._writev_slot(0, 0)

    assert excinfo.value.errno == errno.EIO


@pytest.mark.parametrize("reason", ["open_fails", "no_support", "missing_module"])
def test_uring_falls_back_to_posix(make_backend, monkeypatch, caplog_vllm, reason):
    """io_uring setup failures keep preadv/pwritev working with one warning."""
    fake = None
    if reason == "open_fails":
        fake = make_fake_uring()
        fake.raise_on_open = True
    elif reason == "no_support":
        fake = make_fake_uring(support=False)
    install_fake_uring(monkeypatch, fake)

    with caplog_vllm.at_level(logging.WARNING):
        backend = make_backend()

    assert backend._uring_store_file is None
    assert backend._uring_load_file is None
    assert backend._uring is None
    assert "POSIX" in caplog_vllm.text
    if fake is not None:
        assert fake.registered == []
    _assert_slot_round_trip(backend)


def test_posix_backend_is_the_default(make_backend, monkeypatch):
    fake = make_fake_uring()
    install_fake_uring(monkeypatch, fake)
    monkeypatch.delenv("VLLM_KV_OFFLOAD_DISK_IO_BACKEND")

    backend = make_backend()

    assert backend._uring_store_file is None
    assert fake.files == []
    _assert_slot_round_trip(backend)


def test_shutdown_closes_uring_handles_and_unregisters(make_backend, monkeypatch):
    fake = make_fake_uring()
    install_fake_uring(monkeypatch, fake)
    backend = make_backend()

    backend.shutdown()

    assert all(f.closed for f in fake.files)
    assert fake.unregister_calls == 1
    assert backend._uring is None


def test_direct_io_follows_page_cache_setting(make_backend, monkeypatch, tmp_path):
    if not probe_o_direct(str(tmp_path)):
        pytest.skip("O_DIRECT is unsupported on this filesystem")
    fake = make_fake_uring()
    install_fake_uring(monkeypatch, fake)

    backend = make_backend(use_page_cache=False)

    assert backend._uring_store_file is not None
    assert [f.direct_io for f in fake.files] == [True, True]
