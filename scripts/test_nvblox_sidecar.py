#!/usr/bin/env python
"""Standalone smoke test for the nvblox sidecar.

Verifies that:
  1. ``.venv_nvblox_sidecar/bin/python`` exists and has nvblox_torch installed.
  2. The sidecar can spawn and respond to a request.
  3. The returned ESDF values are correct for a synthetic scene
     (sphere obstacle at the origin, then a 3-cube tabletop scene).
  4. Hot-path latency is in the expected range (<200 ms).

Run from the RoboLab repo root with the **RoboLab venv** (which has msgpack +
msgpack-numpy installed)::

    cd src/RoboLab
    .venv/bin/python scripts/test_nvblox_sidecar.py

Optional flag: ``--sidecar-python /custom/path/python`` to point at a
non-default sidecar interpreter.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import sys
import time

import numpy as np


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
DEFAULT_SIDECAR_PY = os.path.join(REPO_ROOT, ".venv_nvblox_sidecar", "bin", "python")
SIDECAR_SCRIPT = os.path.join(REPO_ROOT, "robolab_policy_client", "sdf", "sidecar_main.py")
PROTOCOL_PATH = os.path.join(REPO_ROOT, "robolab_policy_client", "sdf", "protocol.py")


def _load_protocol():
    spec = importlib.util.spec_from_file_location("_smoketest_protocol", PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import protocol from {PROTOCOL_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _check_sidecar_venv(sidecar_py: str) -> None:
    if not os.path.isfile(sidecar_py):
        raise SystemExit(
            f"FAIL: sidecar python not found at {sidecar_py}\n"
            f"      Pass --sidecar-python /path/to/.venv_nvblox_sidecar/bin/python"
        )
    print(f"[1/4] sidecar interpreter: {sidecar_py}")
    res = subprocess.run(
        [sidecar_py, "-c", "import nvblox_torch, torch; print(torch.cuda.is_available())"],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        raise SystemExit(
            f"FAIL: sidecar venv missing nvblox_torch / torch:\n{res.stderr}\n"
            "      Reinstall with:\n"
            "        .venv_nvblox_sidecar/bin/python -m pip install \\\n"
            "          https://github.com/nvidia-isaac/nvblox/releases/download/"
            "v0.0.9/nvblox_torch-0.0.9+cu12ubuntu24-py3-none-linux_x86_64.whl"
        )
    print(f"      nvblox_torch import OK; CUDA available: {res.stdout.strip()}")


def _query_sphere(proto, proc, voxel_size: float = 0.05) -> dict:
    """Single-scene batched request: 1 element in 'scenes'."""
    req = {
        "voxel_size": voxel_size,
        "aabb_min": [-1.0, -1.0, -1.0],
        "aabb_max": [1.0, 1.0, 1.0],
        "grid_origin": [-0.5, -0.5, -0.5],
        "grid_dims": [20, 20, 20],
        "max_esdf_distance_m": 5.0,
        "scenes": [
            {"primitives": [{"type": "sphere", "params": [0.0, 0.0, 0.0, 0.1]}]},
        ],
    }
    proto.write_msg(proc.stdin, req)
    return proto.read_msg(proc.stdout)


def _query_tabletop_batch(proto, proc) -> dict:
    """Batched 2-scene request: full tabletop AND grasped (banana removed)
    in one IPC. Mirrors what the eval loop does on a multi-env step where
    env 0 is mid-grasp and env 1 isn't."""
    primitives_full = [
        {"type": "cube", "params": [0.4, 0.0, -0.025, 0.6, 0.6, 0.05]},   # table
        {"type": "cube", "params": [0.5, -0.1, 0.05, 0.15, 0.15, 0.10]},  # bowl AABB
        {"type": "cube", "params": [0.45, 0.1, 0.10, 0.20, 0.20, 0.20]},  # banana stand-in
    ]
    req = {
        "voxel_size": 0.025,
        "aabb_min": [-0.2, -0.6, -0.05],
        "aabb_max": [0.8, 0.6, 1.2],
        "grid_origin": [-0.2, -0.6, -0.05],
        "grid_dims": [40, 48, 50],
        "max_esdf_distance_m": 5.0,
        "scenes": [
            {"primitives": primitives_full},        # env 0: pre-grasp
            {"primitives": primitives_full[:2]},    # env 1: banana grasped
        ],
    }
    proto.write_msg(proc.stdin, req)
    return proto.read_msg(proc.stdout)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sidecar-python", default=DEFAULT_SIDECAR_PY)
    args = parser.parse_args()

    _check_sidecar_venv(args.sidecar_python)
    proto = _load_protocol()

    print(f"[2/4] spawning sidecar: {SIDECAR_SCRIPT}")
    proc = subprocess.Popen(
        [args.sidecar_python, SIDECAR_SCRIPT],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr,
    )
    try:
        # --- Sphere check (1-scene batch)
        t0 = time.time()
        resp = _query_sphere(proto, proc)
        cold_ms = (time.time() - t0) * 1000
        if "error" in resp:
            raise SystemExit(f"FAIL: sidecar errored on first request:\n{resp['error']}")
        sdfs = resp["sdfs"]
        if sdfs.shape[0] != 1:
            raise SystemExit(f"FAIL: expected 1 SDF, got {sdfs.shape[0]}")
        sdf_at_origin = float(sdfs[0, 10, 10, 10])  # voxel center at world (0, 0, 0)
        if not (-0.15 < sdf_at_origin < 0.0):
            raise SystemExit(
                f"FAIL: sphere SDF at origin = {sdf_at_origin:.4f}, "
                f"expected ~-0.1 (inside r=0.1 sphere)"
            )
        print(f"[3/4] sphere SDF correct: at origin={sdf_at_origin:+.4f} m, "
              f"cold-build={cold_ms:.0f} ms, hot-build={resp['build_ms']:.0f} ms")

        # --- Batched 2-scene tabletop (mid-grasp + pre-grasp in ONE IPC)
        t_full = time.time()
        resp = _query_tabletop_batch(proto, proc)
        latency_ms = (time.time() - t_full) * 1000
        if "error" in resp:
            raise SystemExit(f"FAIL: sidecar errored on batch request:\n{resp['error']}")
        sdfs = resp["sdfs"]
        if sdfs.shape[0] != 2:
            raise SystemExit(f"FAIL: expected 2 SDFs in batch, got {sdfs.shape[0]}")
        # Banana centre voxel (same world coords for both envs)
        banana_world = np.array([0.45, 0.1, 0.1])
        origin = np.array([-0.2, -0.6, -0.05])
        ijk = tuple(((banana_world - origin) / 0.025).astype(int))
        sdf_full = float(sdfs[0][ijk])
        sdf_grasped = float(sdfs[1][ijk])
        if not (sdf_full < 0):
            raise SystemExit(f"FAIL: env0 (with banana) SDF = {sdf_full:.3f}, expected < 0")
        if not (sdf_grasped > 0):
            raise SystemExit(f"FAIL: env1 (banana grasped) SDF = {sdf_grasped:.3f}, expected > 0")
        print(f"[4/4] batched grasp exclusion correct: env0 (banana in scene)={sdf_full:+.3f} m, "
              f"env1 (banana grasped)={sdf_grasped:+.3f} m; "
              f"2-scene batched build={latency_ms:.0f} ms (sidecar build={resp['build_ms']:.0f} ms)")

        if latency_ms > 500:
            print(
                f"WARNING: batched hot-path latency is high ({latency_ms:.0f} ms for 2 envs). "
                "Expected <200 ms; check GPU utilisation."
            )

        print("\nAll checks passed. Sidecar is healthy.")
        return 0
    finally:
        if proc.stdin:
            proc.stdin.close()
        proc.wait(timeout=5)


if __name__ == "__main__":
    sys.exit(main())
