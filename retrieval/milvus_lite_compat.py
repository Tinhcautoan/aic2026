"""
Runtime compatibility shim for milvus-lite on Windows / newer Python.

milvus-lite 2.4.x (the version pinned for the Linux/vast.ai deployment) ships
no Windows wheels, so on Windows pip installs the pure-Python milvus-lite 3.x
instead.  That embedded server does **not** implement the ``AllocTimestamp``
RPC, yet pymilvus 3.x calls it while waiting for index creation
(``wait_for_creating_index``), which surfaces as::

    NotImplementedError: Method not implemented!

This module monkey-patches ``MilvusServicer`` — the in-process gRPC servicer
that milvus-lite 3.x registers for a ``.db`` URI — to implement
``AllocTimestamp`` with a monotonic Milvus hybrid timestamp, restoring the
behaviour the pinned 2.4.x server had for free.

The patch is a no-op whenever milvus-lite 3.x is not installed, e.g. the
Linux deployment where milvus-lite 2.4.x is a C++ binary that does not ship
the ``milvus_lite.adapter.grpc.servicer`` module at all.
"""

from __future__ import annotations

import logging
import threading
import time

logger = logging.getLogger(__name__)

_patched = False
_patch_lock = threading.Lock()

# Milvus hybrid timestamp: physical component is milliseconds << 18, the low
# 18 bits are a logical counter.  The value is only used as a read-guarantee
# timestamp; milvus-lite's DescribeIndex ignores it, so a monotonic value is
# all that is required.
_TS_PHYSICAL_SHIFT = 18
_TS_LOGICAL_MASK = (1 << _TS_PHYSICAL_SHIFT) - 1


def patch_milvus_lite_alloc_timestamp() -> bool:
    """Install the ``AllocTimestamp`` handler on milvus-lite 3.x's servicer.

    Must run before the first ``MilvusClient(uri=<file>.db)`` is created —
    the embedded gRPC server captures every RPC handler when it starts.
    Idempotent.  Returns True when the patch is in place (or already was),
    False when milvus-lite 3.x is not present and no patch is needed.
    """
    global _patched

    with _patch_lock:
        if _patched:
            return True

        try:
            from milvus_lite.adapter.grpc import servicer as servicer_mod
            from pymilvus.grpc_gen import common_pb2, milvus_pb2
        except Exception as exc:  # milvus-lite 2.4.x (binary) or not installed
            logger.debug("milvus_lite AllocTimestamp patch skipped: %s", exc)
            return False

        servicer_cls = servicer_mod.MilvusServicer
        # Only override if the class itself does not define AllocTimestamp
        # (i.e. it currently inherits the UNIMPLEMENTED stub from the base
        # MilvusServiceServicer).  Never clobber a real implementation.
        if "AllocTimestamp" in getattr(servicer_cls, "__dict__", {}):
            _patched = True
            return True

        counter = [0]

        def alloc_timestamp(self, request, context):
            counter[0] += 1
            timestamp = (int(time.time() * 1000) << _TS_PHYSICAL_SHIFT) | (
                counter[0] & _TS_LOGICAL_MASK
            )
            return milvus_pb2.AllocTimestampResponse(
                status=common_pb2.Status(
                    error_code=common_pb2.ErrorCode.Success,
                    code=0,
                    reason="",
                ),
                timestamp=timestamp,
            )

        servicer_cls.AllocTimestamp = alloc_timestamp
        _patched = True
        logger.info(
            "Patched milvus_lite AllocTimestamp (required by pymilvus %s)",
            _pymilvus_version(),
        )
        return True


def _pymilvus_version() -> str:
    try:
        from pymilvus import __version__

        return str(__version__)
    except Exception:
        return "?"
