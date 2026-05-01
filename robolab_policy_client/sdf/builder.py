"""Build a workspace ESDF from RoboLab's IsaacLab scene state.

For every non-grasped scene object the builder queries
:meth:`robolab.core.world.world_state.WorldState.get_bbox` to obtain the 8
world-frame OBB corners, derives a world-axis AABB cube primitive (a
conservative wrap of the OBB), and ships the primitive list to a long-lived
nvblox sidecar process running under ``.venv_nvblox_sidecar``.

Per replan output (a ``dict[str, np.ndarray]`` ready to attach to the
websocket observation as ``fkc/*`` keys):

* ``sdf_grid``      : ``(Nx, Ny, Nz)`` float16 ESDF values, positive outside.
* ``sdf_origin``    : ``(3,)`` float32 world position of voxel ``(0, 0, 0)``.
* ``sdf_voxel_size``: scalar float32, edge length in metres.
* ``safety_margin`` : scalar float32, mirrored from the builder config so
  the openpi server can override its FKCConfig default per-replan.

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
from typing import Iterable

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
    """Static knobs for :class:`SDFBuilder`."""

    sidecar_python: str
    """Absolute path to the .venv_nvblox_sidecar python interpreter."""

    workspace: WorkspaceGrid
    """Voxel-grid layout for the workspace SDF (in world frame)."""

    obstacle_object_names: tuple[str, ...]
    """Scene objects to consider as obstacles. The currently grasped subset
    of these is filtered out per-replan."""

    safety_margin: float = 0.02
    """Distance threshold (metres) below which the openpi-side hinge fires.
    Mirrored to ``fkc/safety_margin`` so the server can override its
    FKCConfig default."""

    grasp_force_threshold: float = 0.1
    """Contact force above which an object counts as 'grasped' and is
    excluded from the obstacle SDF."""

    gripper_body_name: str = "gripper"
    """Name of the gripper body in :class:`WorldState`. RoboLab's DROID
    contact sensor is registered under this key."""

    # OBB→AABB padding. nvblox treats a layer of voxels as 'sites', so
    # adding voxel_size of pad makes the cube fully contain the OBB.
    aabb_padding: float = 0.0

    max_esdf_distance_m: float = 5.0
    """nvblox truncates ESDF values to this max distance. Set well above
    the workspace diagonal so far-from-obstacle voxels still report a sane
    positive value."""

    sidecar_args: tuple[str, ...] = field(default_factory=tuple)


class SDFBuilder:
    """Per-replan workspace ESDF generator.

    Threadsafe wrt a single inference thread — internally serialises sidecar
    requests with a lock so concurrent calls (none expected today) cannot
    interleave bytes on the pipe.

    Usage::

        builder = SDFBuilder(cfg).start()
        try:
            extras = builder.build(world_state, env_id=0)
            # extras maps fkc/<key> -> np.ndarray
        finally:
            builder.close()
    """

    def __init__(self, cfg: SDFBuilderConfig) -> None:
        self.cfg = cfg
        self._proc: subprocess.Popen | None = None
        self._lock = threading.Lock()
        # The live :class:`robolab.core.world.world_state.WorldState`. Wired
        # in via :meth:`set_world` after env creation. We hold this directly
        # rather than fetching it through the ``get_world()`` global cache,
        # because that cache invalidates whenever ``get_world()`` is called
        # with a different ``env`` argument (including ``None``) — and we
        # may be called from threads / contexts that don't have ``env``.
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
        """Attach the live ``WorldState`` used to read object poses + grasp.

        Must be called once after ``start()`` and before the first ``build``.
        We hold a direct reference here instead of routing through RoboLab's
        ``get_world()`` global cache, which silently invalidates whenever a
        different ``env`` (including ``None``) is passed — that was the
        source of the ``self.env is None`` errors during early integration.
        """
        self._world_state = world_state
        return self

    # ----- per-replan -----------------------------------------------------
    def build(
        self,
        *,
        env_id: int = 0,
        explicit_excluded: Iterable[str] = (),
    ) -> dict[str, np.ndarray]:
        """Construct the SDF from the live IsaacLab scene state.

        Args:
            env_id: which parallel env to read pose / contact info from
                (single-env eval today, so 0).
            explicit_excluded: extra object names to drop from the obstacle
                set in addition to the contact-sensor-detected grasped set.

        Returns:
            ``{"sdf_grid", "sdf_origin", "sdf_voxel_size", "safety_margin"}``
            with the keys already prefixed for the websocket as ``fkc/*``.
        """
        if self._proc is None:
            raise RuntimeError("SDFBuilder.start() must be called first")
        if self._world_state is None:
            raise RuntimeError(
                "SDFBuilder.set_world(world_state) must be called before build(). "
                "Wire it in run_eval.py right after create_env() returns."
            )
        world_state = self._world_state

        # Currently grasped objects → excluded from the obstacle SDF so the
        # robot can manipulate them freely.
        try:
            grasped = set(
                world_state.get_objects_in_contact_with(
                    self.cfg.gripper_body_name,
                    list(self.cfg.obstacle_object_names),
                    force_threshold=self.cfg.grasp_force_threshold,
                    env_id=env_id,
                )
            )
        except Exception:
            # If contact querying fails (older RoboLab, no sensor), fall
            # back to whatever the caller passed explicitly.
            logger.exception("get_objects_in_contact_with failed; assuming nothing is grasped")
            grasped = set()
        excluded = grasped | set(explicit_excluded)

        primitives = []
        for name in self.cfg.obstacle_object_names:
            if name in excluded:
                continue
            try:
                aabb_params = self._object_world_aabb(world_state, name, env_id)
            except Exception:
                logger.exception("get_bbox failed for %r; skipping", name)
                continue
            if aabb_params is None:
                continue
            primitives.append({"type": "cube", "params": list(aabb_params)})

        ws = self.cfg.workspace
        req = {
            "voxel_size": ws.voxel_size,
            "aabb_min": list(ws.aabb_min),
            "aabb_max": list(ws.aabb_max),
            "grid_origin": list(ws.grid_origin),
            "grid_dims": list(ws.grid_dims),
            "primitives": primitives,
            "max_esdf_distance_m": self.cfg.max_esdf_distance_m,
        }
        with self._lock:
            assert self._proc is not None and self._proc.stdin and self._proc.stdout
            _proto.write_msg(self._proc.stdin, req)
            resp = _proto.read_msg(self._proc.stdout)
        if resp is None:
            raise RuntimeError("nvblox sidecar closed unexpectedly")
        if "error" in resp:
            raise RuntimeError(f"nvblox sidecar error:\n{resp['error']}")
        sdf = np.asarray(resp["sdf"])  # (Nx, Ny, Nz) float16
        return {
            "fkc/sdf_grid": sdf,
            "fkc/sdf_origin": np.asarray(ws.grid_origin, dtype=np.float32),
            "fkc/sdf_voxel_size": np.float32(ws.voxel_size),
            "fkc/safety_margin": np.float32(self.cfg.safety_margin),
        }

    # ----- helpers --------------------------------------------------------
    def _object_world_aabb(
        self, world_state, name: str, env_id: int
    ) -> tuple[float, float, float, float, float, float] | None:
        """Compute the world-axis AABB of an object's OBB.

        Returns ``(cx, cy, cz, sx, sy, sz)`` or ``None`` if the object has
        zero-volume geometry (e.g. a frame-only XForm).
        """
        # Pass env_id=None to avoid the legacy Gf.Vec3d return path; we want
        # tensors so we can use numpy directly.
        corners_world, _centroid = world_state.get_bbox(name)
        # corners_world: (num_envs, 8, 3) tensor.
        corners_np = np.asarray(corners_world[env_id].detach().cpu().numpy(), dtype=np.float32)
        lo = corners_np.min(axis=0)
        hi = corners_np.max(axis=0)
        size = hi - lo
        if not np.all(size > 0):
            return None
        # Pad the AABB so the discretised TSDF reliably covers the OBB.
        pad = float(self.cfg.aabb_padding)
        size = size + 2.0 * pad
        center = 0.5 * (lo + hi)
        return (
            float(center[0]),
            float(center[1]),
            float(center[2]),
            float(size[0]),
            float(size[1]),
            float(size[2]),
        )
