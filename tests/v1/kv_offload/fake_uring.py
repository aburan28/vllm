# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""File-backed stand-in for the ``mooncake.uring`` pybind module.

The fake keeps the real file contents (pread/pwrite behind the io_uring
surface) so tests can check data integrity, and records every call so they
can check how the backends drive the module.
"""

import ctypes
import os
import sys
import types

import pytest


class FakeUringFile:
    def __init__(
        self,
        module: "FakeUringModule",
        path: str,
        flags: int,
        queue_depth: int = 32,
        direct_io: bool = False,
    ):
        if module.raise_on_open:
            raise RuntimeError("io_uring_setup: Operation not permitted")
        self.module = module
        self.path = path
        self.flags = flags
        self.queue_depth = queue_depth
        self.direct_io = direct_io
        self.fd = os.open(path, flags, 0o644)
        self.closed = False
        module.files.append(self)

    def _io(self, kind: str, buf_ptr: int, length: int, offset: int) -> int:
        self.module.calls.append((kind, self.path, buf_ptr, length, offset))
        if self.module.fail_errno:
            return -self.module.fail_errno
        if kind == "read":
            data = os.pread(self.fd, length, offset)
            ctypes.memmove(buf_ptr, data, len(data))
            return len(data)
        return os.pwrite(
            self.fd, (ctypes.c_char * length).from_address(buf_ptr), offset
        )

    def read_aligned(self, buf_ptr: int, length: int, offset: int) -> int:
        return self._io("read", buf_ptr, length, offset)

    def write_aligned(self, buf_ptr: int, length: int, offset: int) -> int:
        return self._io("write", buf_ptr, length, offset)

    def batch_read(
        self, buf_ptrs: list[int], lengths: list[int], offsets: list[int]
    ) -> list[int]:
        self.module.batches.append(
            ("read", list(buf_ptrs), list(lengths), list(offsets))
        )
        return [self._io("read", *desc) for desc in zip(buf_ptrs, lengths, offsets)]

    def batch_write(
        self, buf_ptrs: list[int], lengths: list[int], offsets: list[int]
    ) -> list[int]:
        self.module.batches.append(
            ("write", list(buf_ptrs), list(lengths), list(offsets))
        )
        return [self._io("write", *desc) for desc in zip(buf_ptrs, lengths, offsets)]

    def datasync(self) -> int:
        os.fdatasync(self.fd)
        return 0

    def close(self) -> None:
        if not self.closed:
            os.close(self.fd)
            self.closed = True


class FakeUringModule(types.ModuleType):
    def __init__(self, support: bool = True):
        super().__init__("mooncake.uring")
        self.SUPPORT_URING = support
        self.files: list[FakeUringFile] = []
        self.calls: list[tuple[str, str, int, int, int]] = []
        self.batches: list[tuple[str, list[int], list[int], list[int]]] = []
        self.registered: list[tuple[int, int]] = []
        self.unregister_calls = 0
        self.fail_errno = 0
        self.raise_on_open = False
        self.register_ok = True
        self.UringFile = self._uring_file

    def _uring_file(self, *args, **kwargs) -> FakeUringFile:
        return FakeUringFile(self, *args, **kwargs)

    def register_global_buffer(self, ptr: int, length: int) -> bool:
        self.registered.append((ptr, length))
        return self.register_ok

    def unregister_global_buffer(self) -> None:
        self.unregister_calls += 1


class _FakeMooncakePackage(types.ModuleType):
    uring: FakeUringModule


def make_fake_uring(support: bool = True) -> FakeUringModule:
    return FakeUringModule(support)


def install_fake_uring(
    monkeypatch: pytest.MonkeyPatch, module: FakeUringModule | None
) -> None:
    """Inject ``module`` as ``mooncake.uring`` (``None`` makes the import
    fail) and select the io_uring disk backend."""
    package = _FakeMooncakePackage("mooncake")
    if module is not None:
        package.uring = module
    monkeypatch.setitem(sys.modules, "mooncake", package)
    monkeypatch.setitem(sys.modules, "mooncake.uring", module)
    monkeypatch.setenv("VLLM_KV_OFFLOAD_DISK_IO_BACKEND", "io_uring")
