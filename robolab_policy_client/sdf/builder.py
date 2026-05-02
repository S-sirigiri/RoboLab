"""Build a workspace ESDF from RoboLab's IsaacLab scene state.

For every non-grasped scene object the builder queries
:meth:`robolab.core.world.world_state.WorldState.get_bbox` to obtain the 8
world-frame OBB corners, derives a world-axis AABB cube primitive (a
conservative wrap of the OBB), and ships the primitive list to a long-lived
nvblox sidecar process running under ``.venv_nvblox_sidecar``.

The sidecar uses one nvblox :class:`Mapper` with multiple ``mapper_id``s — one
per parallel env — so a single IPC roundtrip covers all envs that need a
replan in the current step.

Per replan output (a ``dict[str, np.ndarray]`` ready to attach to the
websocket observation as ``fkc/*`` keys):

* ``fkc/sdf_grid``      : ``(Nx, Ny, Nz)`` float16 ESDF, positive outside.
* ``fkc/sdf_origin``    : ``(3,)`` float32 world position of voxel ``(0,0,0)``.
* ``fkc/sdf_voxel_size``: scalar float32, edge length in metres.

The ``safety_margin`` knob lives in the openpi YAML
(``collision.safety_margin``) and is not sent over the wire.

The sidecar is spawned once at :meth:`SDFBuilder.start` and torn down on
:meth:`close`.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# Load protocol.py without going through ``robolab_policy_client/__init__.py``
# (that module pulls in robolab.eval and friends, which we don't need here).
def _load_protocol_module():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location(
        "_robolab_sdf_protocol", os.path.join(here, "protocol.py")
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not locate sdf/protocol.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_proto = _load_protocol_module()


@dataclass(frozen=True)
class WorkspaceGrid:
    """Voxel-grid layout for the workspace SDF.

    The grid spans ``[origin, origin + voxel_size * dims]`` in world frame.
    """

    voxel_size: float
    grid_origin: tuple[float, float, float]
    grid_dims: tuple[int, int, int]

    @property
    def aabb_min(self) -> tuple[float, float, float]:
        return self.grid_origin

    @property
    def aabb_max(self) -> tuple[float, float, float]:
        ox, oy, oz = self.grid_origin
        sx, sy, sz = (d * self.voxel_size for d in self.grid_dims)
        return (ox + sx, oy + sy, oz + sz)

    @classmethod
    def from_bounds(
        cls,
        bounds: tuple[tuple[float, float, float], tuple[float, float, float]],
        voxel_size: float,
    ) -> "WorkspaceGrid":
        """Snap workspace bounds to an integer voxel-dim grid."""
        (xmin, ymin, zmin), (xmax, ymax, zmax) = bounds
        dims = (
            max(1, int(np.ceil((xmax - xmin) / voxel_size))),
            max(1, int(np.ceil((ymax - ymin) / voxel_size))),
            max(1, int(np.ceil((zmax - zmin) / voxel_size))),
        )
        return cls(
            voxel_size=float(voxel_size),
            grid_origin=(float(xmin), float(ymin), float(zmin)),
            grid_dims=dims,
        )


@dataclass
class SDFBuilderConfig:
    """Static knobs for :class:`SDFBuilder`. All values flow in from
    ``run_eval.py`` CLI flags — the openpi-side knobs (mode, safety_margin,
    softplus_beta, …) live in the FKC YAML and are NOT mirrored here."""

    sidecar_python: str
    """Absolute path to the .venv_nvblox_sidecar python interpreter."""

    workspace: WorkspaceGrid
    """Voxel-grid layout for the workspace SDF (in world frame)."""

    obstacle_object_names: tuple[str, ...]
    """Scene objects to consider as obstacles. The currently grasped subset
    of these is filtered out per-replan."""

    grasp_force_threshold: float = 0.1
    """Contact force above which an object counts as 'grasped' and is
    excluded from the obstacle SDF."""

    gripper_body_name: str = "gripper"
    """Name of the gripper body in :class:`WorldState`. RoboLab's DROID
    contact sensor is registered under this key."""

    aabb_padding: float = 0.0
    """Pad the world-axis AABB derived from each OBB by this many metres on
    every side. nvblox treats a layer of voxels as 'sites' so a small pad
    (~ voxel_size) reliably covers thin / rotated objects."""

    max_esdf_distance_m: float = 5.0
    """nvblox truncates ESDF values to this max distance. Set well above
    the workspace diagonal so far-from-obstacle voxels still report a sane
    positive value."""

    sidecar_args: tuple[str, ...] = field(default_factory=tuple)

    ooi_exclusion_mode: str = "dynamic"
    """How object-of-interest (OOI) collision exclusion is handled.

    - ``"dynamic"``: OOI is treated as an obstacle until the gripper
      contact force exceeds :attr:`grasp_force_threshold`, then excluded
      automatically by the contact-sensor safety net (current behavior).
    - ``"static"``: every name in :attr:`ooi_object_names` is *always*
      excluded from the SDF, regardless of grasp state. Non-OOI
      obstacles still get the dynamic grasp safety net.
    """

    ooi_object_names: tuple[str, ...] = ()
    """Per-task object-of-interest names — the objects being
    manipulated/grasped. Consulted only when
    :attr:`ooi_exclusion_mode` is ``"static"``."""

    def __post_init__(self) -> None:
        valid_modes = {"dynamic", "static"}
        if self.ooi_exclusion_mode not in valid_modes:
            raise ValueError(
                f"SDFBuilderConfig.ooi_exclusion_mode must be one of "
                f"{sorted(valid_modes)}, got {self.ooi_exclusion_mode!r}"
            )
        if self.ooi_exclusion_mode == "static" and not self.ooi_object_names:
            logger.warning(
                "SDFBuilderConfig: ooi_exclusion_mode='static' but "
                "ooi_object_names is empty; behavior reduces to 'dynamic'."
            )


class SDFBuilder:
    """Per-replan workspace ESDF generator with batched multi-env support.

    Threadsafe wrt a single inference thread — internally serialises sidecar
    requests with a lock so concurrent calls cannot interleave bytes on the
    pipe.

    Usage::

        builder = SDFBuilder(cfg).start().set_world(world)
        # ... once per env step, before the per-env infer loop:
        results = builder.build_batch([0, 1, 2, 3])
        # results[i] is the dict to merge into env i's websocket request.
    """

    def __init__(self, cfg: SDFBuilderConfig) -> None:
        self.cfg = cfg
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        # The live :class:`robolab.core.world.world_state.WorldState`. Wired
        # in via :meth:`set_world` after env creation. Held directly rather
        # than fetched through ``get_world()`` (whose global cache silently
        # invalidates when called with a different ``env`` argument,
        # including ``None``).
        self._world_state = None

    # ----- lifecycle ------------------------------------------------------
    def start(self) -> "SDFBuilder":
        if self._proc is not None:
            return self
        sidecar_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sidecar_main.py")
        if not os.path.isfile(sidecar_script):
            raise FileNotFoundError(f"sidecar script not found: {sidecar_script}")
        if not os.path.isfile(self.cfg.sidecar_python):
            raise FileNotFoundError(
                f"sidecar python interpreter not found: {self.cfg.sidecar_python}"
            )
        cmd = [self.cfg.sidecar_python, sidecar_script, *self.cfg.sidecar_args]
        logger.info("Spawning nvblox sidecar: %s", " ".join(cmd))
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,  # let logs flow through
        )
        return self

    def close(self) -> None:
        if self._proc is None:
            return
        try:
            if self._proc.stdin:
                self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
        self._proc = None

    def __enter__(self) -> "SDFBuilder":
        return self.start()

    def __exit__(self, *_exc) -> None:
        self.close()

    # ----- world wiring ---------------------------------------------------
    def set_world(self, world_state) -> "SDFBuilder":
        """Attach the live ``WorldState`` used to read object poses + grasp."""
        self._world_state = world_state
        return self

    # ----- per-replan -----------------------------------------------------
    def build(
        self,
        *,
        env_id: int = 0,
        explicit_excluded: Iterable[str] = (),
    ) -> dict[str, np.ndarray]:
        """Single-env convenience wrapper around :meth:`build_batch`."""
        results = self.build_batch(
            env_ids=[env_id],
            explicit_excluded_per_env={env_id: set(explicit_excluded)},
        )
        return results[0]

    def build_batch(
        self,
        env_ids: Sequence[int],
        *,
        explicit_excluded_per_env: Mapping[int, Iterable[str]] | None = None,
    ) -> list[dict[str, np.ndarray]]:
        """Build SDFs for ``env_ids`` in a single sidecar IPC roundtrip.

        Args:
            env_ids: parallel-env indices to build for. Order of the returned
                list matches.
            explicit_excluded_per_env: optional per-env set of names to drop
                in addition to the contact-sensor-detected grasped set.

        Returns:
            ``len(env_ids)`` dicts, each with the ``fkc/*`` keys ready to
            merge into that env's websocket request.
        """
        if self._proc is None:
            raise RuntimeError("SDFBuilder.start() must be called first")
        if self._world_state is None:
            raise RuntimeError(
                "SDFBuilder.set_world(world_state) must be called before build_batch()."
            )
        env_ids = list(env_ids)
        if not env_ids:
            return []
        explicit = dict(explicit_excluded_per_env or {})
        world_state = self._world_state

        # ----- vectorised OBB read across all (env, obstacle) pairs -------
        # ``world.get_bbox(name)`` already returns a (num_envs, 8, 3) tensor
        # so we make ONE call per name and slice per env, instead of N×M
        # individual calls. ``aabbs`` ends up shape (M, num_envs, 6) where
        # the trailing dim is (cx, cy, cz, sx, sy, sz). NaN rows are
        # zero-volume / missing objects and get filtered out below.
        names = list(self.cfg.obstacle_object_names)
        aabbs = self._all_world_aabbs(world_state, names)

        # ----- contact-based grasp detection (per env) --------------------
        grasped_per_env: dict[int, set[str]] = {}
        for eid in env_ids:
            try:
                grasped = set(
                    world_state.get_objects_in_contact_with(
                        self.cfg.gripper_body_name,
                        names,
                        force_threshold=self.cfg.grasp_force_threshold,
                        env_id=eid,
                    )
                )
            except Exception:
                logger.exception(
                    "get_objects_in_contact_with failed for env_id=%d; "
                    "assuming nothing is grasped", eid,
                )
                grasped = set()
            grasped_per_env[eid] = grasped

        # ----- assemble per-env primitives lists --------------------------
        static_ooi_excluded: set[str] = (
            set(self.cfg.ooi_object_names)
            if self.cfg.ooi_exclusion_mode == "static"
            else set()
        )
        scenes_payload = []
        for eid in env_ids:
            excluded = (
                grasped_per_env[eid]
                | set(explicit.get(eid, ()))
                | static_ooi_excluded
            )
            primitives = []
            for j, name in enumerate(names):
                if name in excluded:
                    continue
                aabb = aabbs[j, eid]
                if not np.all(np.isfinite(aabb)):
                    continue
                primitives.append({"type": "cube", "params": [float(v) for v in aabb]})
            scenes_payload.append({"primitives": primitives})

        ws = self.cfg.workspace
        req = {
            "voxel_size": ws.voxel_size,
            "aabb_min": list(ws.aabb_min),
            "aabb_max": list(ws.aabb_max),
            "grid_origin": list(ws.grid_origin),
            "grid_dims": list(ws.grid_dims),
            "max_esdf_distance_m": self.cfg.max_esdf_distance_m,
            "scenes": scenes_payload,
        }
        with self._lock:
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            _proto.write_msg(self._proc.stdin, req)
            resp = _proto.read_msg(self._proc.stdout)
        if resp is None:
            raise RuntimeError("nvblox sidecar closed unexpectedly")
        if "error" in resp:
            raise RuntimeError(f"nvblox sidecar error:\n{resp['error']}")
        sdfs = np.asarray(resp["sdfs"])  # (N, Nx, Ny, Nz) float16
        if sdfs.shape[0] != len(env_ids):
            raise RuntimeError(
                f"nvblox sidecar returned {sdfs.shape[0]} SDFs, "
                f"expected {len(env_ids)}"
            )

        # Origin / voxel_size are static across envs; ship cheap copies.
        origin_arr = np.asarray(ws.grid_origin, dtype=np.float32)
        vs_arr = np.float32(ws.voxel_size)
        results: list[dict[str, np.ndarray]] = []
        for i in range(len(env_ids)):
            results.append(
                {
                    "fkc/sdf_grid": sdfs[i],
                    "fkc/sdf_origin": origin_arr,
                    "fkc/sdf_voxel_size": vs_arr,
                }
            )
        return results

    # ----- helpers --------------------------------------------------------
    def _all_world_aabbs(self, world_state, names: Sequence[str]) -> np.ndarray:
        """Return ``(M, num_envs, 6)`` AABB params for every (name, env).

        Trailing dim layout: ``(cx, cy, cz, sx, sy, sz)``. Rows for objects
        whose ``get_bbox`` raised, or whose AABB has zero volume, are filled
        with NaN so the caller can filter them out.
        """
        if not names:
            return np.empty((0, 0, 6), dtype=np.float32)
        # Probe the first available object to learn num_envs.
        num_envs = None
        per_name_corners: list[np.ndarray | None] = []
        for name in names:
            try:
                corners_world, _centroid = world_state.get_bbox(name)
                arr = corners_world.detach().cpu().numpy().astype(np.float32, copy=False)
            except Exception:
                logger.exception("get_bbox failed for %r; skipping", name)
                arr = None
            per_name_corners.append(arr)
            if arr is not None and num_envs is None:
                num_envs = int(arr.shape[0])
        if num_envs is None:
            return np.full((len(names), 0, 6), np.nan, dtype=np.float32)

        out = np.full((len(names), num_envs, 6), np.nan, dtype=np.float32)
        pad = float(self.cfg.aabb_padding)
        for j, arr in enumerate(per_name_corners):
            if arr is None:
                continue
            # arr shape: (num_envs, 8, 3)
            lo = arr.min(axis=1)
            hi = arr.max(axis=1)
            size = hi - lo
            valid = np.all(size > 0, axis=-1)
            if not np.any(valid):
                continue
            size_padded = size + 2.0 * pad
            center = 0.5 * (lo + hi)
            out[j, valid, :3] = center[valid]
            out[j, valid, 3:] = size_padded[valid]
        return out
