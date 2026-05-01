"""Tiny stdin/stdout msgpack-numpy framing for the nvblox sidecar IPC.

Each message is a 4-byte big-endian uint32 length prefix followed by the
msgpack-encoded payload. Numpy arrays are encoded with ``msgpack_numpy``
hooks. The framing is deliberately trivial (no len-zero terminator, no
keep-alives) — both ends call :func:`read_msg` in a loop until EOF.

The same module is imported by both the parent (RoboLab venv) and the
sidecar (``.venv_nvblox_sidecar``). It only depends on ``msgpack`` /
``msgpack_numpy`` / ``numpy``, which are installed in both venvs.
"""

from __future__ import annotations

import struct
from typing import Any

import msgpack
import msgpack_numpy as _mpn
import numpy as np  # noqa: F401  (msgpack_numpy hooks need it imported)


def pack(payload: Any) -> bytes:
    """Encode ``payload`` (with numpy support) into a single msgpack blob."""
    return msgpack.packb(payload, default=_mpn.encode, use_bin_type=True)


def unpack(blob: bytes) -> Any:
    return msgpack.unpackb(blob, object_hook=_mpn.decode, raw=False)


def write_msg(stream, payload: Any) -> None:
    body = pack(payload)
    stream.write(struct.pack(">I", len(body)))
    stream.write(body)
    stream.flush()


def _read_exact(stream, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def read_msg(stream) -> Any | None:
    """Read one length-prefixed msgpack message. Returns ``None`` at EOF."""
    hdr = _read_exact(stream, 4)
    if hdr is None:
        return None
    n = struct.unpack(">I", hdr)[0]
    body = _read_exact(stream, n)
    if body is None:
        return None
    return unpack(body)
