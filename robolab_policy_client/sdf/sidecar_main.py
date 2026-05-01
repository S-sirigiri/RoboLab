"""nvblox sidecar process — runs under ``.venv_nvblox_sidecar``.

Launched as a child of :class:`robolab_policy_client.sdf.builder.SDFBuilder`.
Communicates with the parent over stdin/stdout using
:mod:`robolab_policy_client.sdf.protocol`.

Per request:
  * receives a list of cube/sphere primitives (world-frame parameters)
  * builds an nvblox :class:`Scene` with those primitives
  * appends the scene to a fresh :class:`Mapper` (TSDF + ESDF integration)
  * queries the dense ESDF at every voxel centre of a static workspace grid
  * returns ``{"sdf": float16 array (Nx, Ny, Nz)}``

Invariants kept across requests (so we don't pay CUDA init twice):
  * the precomputed ``points_4d`` voxel-centre tensor on the GPU
  * the ``MapperParams`` for the ESDF integrator (large max_distance so
    far-away voxels still report a positive distance)

The script writes diagnostic logs to stderr only; stdout is reserved for
the framed msgpack channel.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np
import torch

# Load ``protocol.py`` directly from disk, bypassing
# ``robolab_policy_client/__init__.py`` — that package imports IsaacLab /
# h5py / robolab modules which are not installed in the sidecar venv.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROTOCOL_PATH = os.path.join(_THIS_DIR, "protocol.py")


def _load_protocol():
    import importlib.util

    spec = importlib.util.spec_from_file_location("_sidecar_protocol", _PROTOCOL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load protocol module from {_PROTOCOL_PATH}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_proto = _load_protocol()


def _log(msg: str) -> None:
    sys.stderr.write(f"[nvblox-sidecar] {msg}\n")
    sys.stderr.flush()


class _GridCache:
    """Lazily allocate the (N, 4) voxel-centre query tensor on the GPU.

    The grid params (origin, voxel_size, dims) typically don't change across
    a session, so we keep a single CUDA tensor and reuse it. If they do
    change, we reallocate.
    """

    def __init__(self) -> None:
        self.origin: tuple[float, float, float] | None = None
        self.voxel_size: float | None = None
        self.dims: tuple[int, int, int] | None = None
        self.points_4d: torch.Tensor | None = None

    def get(self, origin: np.ndarray, voxel_size: float, dims: tuple[int, int, int]) -> torch.Tensor:
        origin_t = tuple(float(v) for v in origin)
        if (
            self.points_4d is not None
            and self.origin == origin_t
            and self.voxel_size == voxel_size
            and self.dims == dims
        ):
            return self.points_4d
        nx, ny, nz = dims
        xs = origin[0] + voxel_size * np.arange(nx, dtype=np.float32)
        ys = origin[1] + voxel_size * np.arange(ny, dtype=np.float32)
        zs = origin[2] + voxel_size * np.arange(nz, dtype=np.float32)
        X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
        pts = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)
        pts_4d = np.concatenate([pts, np.zeros((pts.shape[0], 1), dtype=np.float32)], axis=-1)
        self.points_4d = torch.from_numpy(pts_4d).cuda().contiguous()
        self.origin = origin_t
        self.voxel_size = voxel_size
        self.dims = dims
        return self.points_4d


def _build_sdf(req: dict, grid_cache: _GridCache) -> np.ndarray:
    # Lazy import so we don't pay nvblox init at module import time, only on
    # the first request.
    from nvblox_torch.scene import Scene
    from nvblox_torch.mapper import Mapper, QueryType
    from nvblox_torch.mapper_params import EsdfIntegratorParams, MapperParams
    from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType

    voxel_size = float(req["voxel_size"])
    aabb_min = [float(v) for v in req["aabb_min"]]
    aabb_max = [float(v) for v in req["aabb_max"]]
    grid_origin = np.asarray(req["grid_origin"], dtype=np.float32).reshape(3)
    grid_dims = tuple(int(d) for d in req["grid_dims"])
    primitives = list(req.get("primitives", []) or [])
    max_distance_m = float(req.get("max_esdf_distance_m", 5.0))

    # No obstacles in the workspace — short-circuit with a free-space grid.
    if not primitives:
        return np.full(grid_dims, max_distance_m, dtype=np.float16)

    scene = Scene()
    scene.set_aabb(aabb_min, aabb_max)
    for p in primitives:
        scene.add_primitive(str(p["type"]), [float(v) for v in p["params"]])

    esdf_params = EsdfIntegratorParams()
    esdf_params.esdf_integrator_max_distance_m = max_distance_m
    mp = MapperParams()
    mp.set_esdf_integrator_params(esdf_params)

    mapper = Mapper(
        voxel_sizes_m=[voxel_size],
        integrator_types=[ProjectiveIntegratorType.TSDF],
        mapper_parameters=mp,
    )
    scene.append_to_mapper(mapper, mapper_id=0)
    # Ensure the ESDF is in sync with the freshly baked TSDF.
    mapper.update_esdf()

    points_4d = grid_cache.get(grid_origin, voxel_size, grid_dims)
    sdf_t = mapper.query_differentiable_layer(QueryType.ESDF, points_4d)
    sdf = sdf_t.detach().cpu().numpy().reshape(grid_dims).astype(np.float16)
    return sdf


def main() -> int:
    parser = argparse.ArgumentParser(description="nvblox SDF sidecar")
    parser.add_argument("--ready-marker", default="", help="Optional path to touch when ready")
    args = parser.parse_args()

    # Force a CUDA context to exist before we tell the parent we're ready —
    # otherwise the first build_sdf call will pay ~1-2s just to init CUDA.
    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.zeros(1, device="cuda")  # warm up

    grid_cache = _GridCache()
    in_stream = sys.stdin.buffer
    out_stream = sys.stdout.buffer

    if args.ready_marker:
        try:
            open(args.ready_marker, "w").close()
        except OSError as exc:
            _log(f"could not touch ready marker {args.ready_marker}: {exc}")

    _log(f"PID={os.getpid()} ready, cuda_available={torch.cuda.is_available()}")

    while True:
        try:
            req = _proto.read_msg(in_stream)
        except Exception:
            _log("read_msg failed, exiting")
            _log(traceback.format_exc())
            return 1
        if req is None:
            _log("EOF on stdin, exiting")
            return 0
        t0 = time.monotonic()
        try:
            sdf = _build_sdf(req, grid_cache)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            _proto.write_msg(out_stream, {"sdf": sdf, "build_ms": elapsed_ms})
        except Exception:
            tb = traceback.format_exc()
            _log(tb)
            _proto.write_msg(out_stream, {"error": tb})


if __name__ == "__main__":
    sys.exit(main())
