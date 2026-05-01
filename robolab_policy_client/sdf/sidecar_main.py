"""nvblox sidecar process — runs under ``.venv_nvblox_sidecar``.

Launched as a child of :class:`robolab_policy_client.sdf.builder.SDFBuilder`.
Communicates with the parent over stdin/stdout using
:mod:`robolab_policy_client.sdf.protocol`.

**Batched protocol** (one IPC roundtrip handles N parallel envs):

Request payload::

    {
        "voxel_size":        float,
        "aabb_min":          [3] float,
        "aabb_max":          [3] float,
        "grid_origin":       [3] float,
        "grid_dims":         [3] int,
        "max_esdf_distance_m": float,
        "scenes": [          # one entry per env
            {"primitives": [{"type": "cube", "params": [...]}, ...]},
            {"primitives": [...]},
            ...
        ]
    }

Response payload::

    {
        "sdfs":     (N, Nx, Ny, Nz) float16  # one ESDF grid per scene
        "build_ms": float                    # total wall-clock of the build
    }

Mapper reuse: a single nvblox :class:`Mapper` configured for ``N`` parallel
mapper_ids is kept across requests as long as ``N`` and ``voxel_size`` don't
change. ``mapper.clear()`` between requests resets the TSDF/ESDF without
paying the per-construct overhead.

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
    """Lazily allocate the (Nv, 4) voxel-centre query tensor on the GPU.

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


class _MapperCache:
    """Hold a single nvblox :class:`Mapper` configured for ``N`` parallel
    mapper_ids.

    Recreated only when ``N`` or ``voxel_size`` changes; otherwise we just
    ``clear()`` between requests (which resets all TSDF/ESDF layers without
    re-allocating the GPU hashtables).
    """

    def __init__(self, max_distance_m_default: float = 5.0) -> None:
        self.mapper = None
        self.params = None
        self.n_scenes: int | None = None
        self.voxel_size: float | None = None
        self._max_distance_m = max_distance_m_default

    def get(self, n_scenes: int, voxel_size: float, max_distance_m: float):
        from nvblox_torch.mapper import Mapper
        from nvblox_torch.mapper_params import EsdfIntegratorParams, MapperParams
        from nvblox_torch.projective_integrator_types import ProjectiveIntegratorType

        if (
            self.mapper is not None
            and self.n_scenes == n_scenes
            and self.voxel_size == voxel_size
            and abs(self._max_distance_m - max_distance_m) < 1e-6
        ):
            self.mapper.clear()  # resets all mapper_ids
            return self.mapper

        # New configuration — rebuild from scratch.
        esdf_params = EsdfIntegratorParams()
        esdf_params.esdf_integrator_max_distance_m = max_distance_m
        mp = MapperParams()
        mp.set_esdf_integrator_params(esdf_params)
        self.params = mp
        self.mapper = Mapper(
            voxel_sizes_m=[voxel_size] * n_scenes,
            integrator_types=[ProjectiveIntegratorType.TSDF] * n_scenes,
            mapper_parameters=mp,
        )
        self.n_scenes = n_scenes
        self.voxel_size = voxel_size
        self._max_distance_m = max_distance_m
        return self.mapper


def _normalize_request(req: dict) -> dict:
    """Accept legacy single-scene requests by promoting ``primitives`` into
    a one-element ``scenes`` list."""
    if "scenes" in req:
        return req
    primitives = req.get("primitives", [])
    out = dict(req)
    out["scenes"] = [{"primitives": primitives}]
    out.pop("primitives", None)
    return out


def _build_sdfs(
    req: dict, grid_cache: _GridCache, mapper_cache: _MapperCache
) -> np.ndarray:
    from nvblox_torch.mapper import QueryType
    from nvblox_torch.scene import Scene

    voxel_size = float(req["voxel_size"])
    aabb_min = [float(v) for v in req["aabb_min"]]
    aabb_max = [float(v) for v in req["aabb_max"]]
    grid_origin = np.asarray(req["grid_origin"], dtype=np.float32).reshape(3)
    grid_dims = tuple(int(d) for d in req["grid_dims"])
    max_distance_m = float(req.get("max_esdf_distance_m", 5.0))
    scenes_spec = list(req["scenes"])
    n_scenes = len(scenes_spec)
    if n_scenes == 0:
        return np.empty((0, *grid_dims), dtype=np.float16)

    # Output buffer; empty scenes are filled with the truncation distance.
    sdfs = np.full((n_scenes, *grid_dims), np.float16(max_distance_m), dtype=np.float16)
    nonempty_ids: list[int] = [
        i for i, s in enumerate(scenes_spec) if (s.get("primitives") or [])
    ]
    if not nonempty_ids:
        return sdfs

    # Re-use the long-lived Mapper if the layout matches.
    mapper = mapper_cache.get(n_scenes, voxel_size, max_distance_m)

    # Build each non-empty scene into its own mapper_id.
    for i in nonempty_ids:
        primitives = scenes_spec[i]["primitives"]
        scene = Scene()
        scene.set_aabb(aabb_min, aabb_max)
        for p in primitives:
            scene.add_primitive(str(p["type"]), [float(v) for v in p["params"]])
        scene.append_to_mapper(mapper, mapper_id=i)

    # Single ESDF update covers every mapper_id we appended to.
    mapper.update_esdf()

    points_4d = grid_cache.get(grid_origin, voxel_size, grid_dims)
    out_buf = torch.zeros(points_4d.shape[0], 4, device=points_4d.device, dtype=points_4d.dtype)
    for i in nonempty_ids:
        sdf_t = mapper.query_differentiable_layer(
            QueryType.ESDF, points_4d, out_buf, mapper_id=i
        )
        sdfs[i] = (
            sdf_t.detach()
            .cpu()
            .numpy()
            .reshape(grid_dims)
            .astype(np.float16, copy=False)
        )
    return sdfs


def main() -> int:
    parser = argparse.ArgumentParser(description="nvblox SDF sidecar")
    parser.add_argument("--ready-marker", default="", help="Optional path to touch when ready")
    args = parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.init()
        _ = torch.zeros(1, device="cuda")  # warm up

    grid_cache = _GridCache()
    mapper_cache = _MapperCache()
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
            req = _normalize_request(req)
            sdfs = _build_sdfs(req, grid_cache, mapper_cache)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            _proto.write_msg(out_stream, {"sdfs": sdfs, "build_ms": elapsed_ms})
        except Exception:
            tb = traceback.format_exc()
            _log(tb)
            _proto.write_msg(out_stream, {"error": tb})


if __name__ == "__main__":
    sys.exit(main())
