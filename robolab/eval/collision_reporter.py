"""Ground-truth collision reporter for FKC eval rollouts.

For every active env at every sim step, computes whether any robot link is
intersecting an OBB-bounded scene obstacle, after excluding objects that are
currently being grasped (so legitimate manipulation contacts don't count).
This is independent of the FKC SDF / nvblox path — it talks directly to
IsaacLab's articulation state and ``WorldState.get_bbox`` so the numbers it
produces are ground truth, not voxelised approximations.

Per-step cost in numbers (one env, ~5 obstacles, ~25 robot links):
  * 1 GPU read of ``articulation.data.body_link_state_w``  (already populated)
  * M ``world.get_bbox`` calls — each is a cached local-geometry lookup +
    an 8-corner transform; ~0.2 ms each.
  * 1 contact-sensor query for grasp detection.
  * O(B*M) numpy point-in-AABB tests; microseconds.
  * 1 dict append to the in-memory buffer.

So <5 ms per step for a typical scene. The buffer is flushed to one
compressed ``.npz`` per run at episode end; the rollout never blocks on
disk I/O.

Output schema (per ``.npz`` file)::

    steps              (N,)   int32   sim step index
    env_ids            (N,)   int32   parallel-env index
    in_collision       (N,)   bool    any robot link inside any non-grasped obstacle
    num_links_in_coll  (N,)   int16   how many robot links are in collision this step
    min_clearance_m    (N,)   float32 closest signed clearance to the boundary (negative = inside)
    grasped_count      (N,)   int8    how many manipulables the gripper is touching
    grasped_names      list[str] (per-record, comma-joined, stored separately)
    obstacle_names     list[str] static metadata for the run
    body_names         list[str] static metadata for the run
    body_clearance_m   (N, B) float32 per-body min clearance (handy for offline drilldowns)

Use :func:`load_run` from ``scripts/analyze_collisions.py`` to read it back.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CollisionRecord:
    step: int
    env_id: int
    in_collision: bool
    num_links_in_collision: int
    min_clearance: float
    body_clearance: np.ndarray  # (B,) per-link clearance (m), copied
    grasped: tuple[str, ...]


@dataclass
class CollisionReporterConfig:
    obstacle_names: tuple[str, ...]
    """Scene-object names checked at every step. The currently-grasped subset
    is filtered out per-step before the AABB intersection check, so
    manipulating an object never counts as a collision."""

    robot_articulation_name: str = "robot"
    """Key under :attr:`WorldState.articulations` whose bodies count as robot
    links. Defaults match RoboLab's DROID config."""

    body_radius_m: float = 0.045
    """Sphere radius used to inflate every robot body when computing the
    clearance to an obstacle AABB. Set to roughly the half-width of the
    Franka link cylinders so 'clearance < 0' lines up with real geometric
    overlap."""

    obstacle_padding_m: float = 0.0
    """Additional inflation applied to every obstacle's world-axis AABB.
    Set above 0 to be conservative about thin objects or to match the SDF
    constraint's ``aabb_padding``."""

    grasp_force_threshold: float = 0.1
    """Force (N) above which an object touched by the gripper counts as
    'grasped' and is excluded from collision checks."""

    gripper_body_name: str = "gripper"
    """Where the contact sensor lives (matches DROID)."""

    body_filter: tuple[str, ...] | None = None
    """If set, only check robot bodies whose name contains any of these
    substrings (case-insensitive). Useful to skip e.g. internal joint stubs
    that don't have meaningful geometry. ``None`` = check all bodies."""

    static_excluded_names: tuple[str, ...] = ()
    """Obstacle names to *always* exclude from the collision check, on top of
    the per-step grasp-based exclusion. Used to mirror the FKC SDF builder's
    ``ooi_exclusion_mode='static'`` so reported collisions stay consistent
    with what the policy was actually being penalised for."""


class CollisionReporter:
    """Per-step ground-truth collision logger with episode-end flush."""

    def __init__(self, world_state, cfg: CollisionReporterConfig) -> None:
        self._world = world_state
        self.cfg = cfg
        self._records: list[CollisionRecord] = []
        self._body_indices: np.ndarray | None = None  # selected body indices
        self._body_names: list[str] = []
        self._articulation = None
        self._init_robot_view()

    # ----- setup ----------------------------------------------------------
    def _init_robot_view(self) -> None:
        try:
            self._articulation = self._world.get_articulation(self.cfg.robot_articulation_name)
        except Exception as exc:  # pragma: no cover - sim env dependent
            logger.warning(
                "CollisionReporter: articulation %r not found (%s); "
                "the reporter will be a no-op.",
                self.cfg.robot_articulation_name, exc,
            )
            self._articulation = None
            return
        all_names = list(self._articulation.body_names)
        if self.cfg.body_filter:
            wanted = [s.lower() for s in self.cfg.body_filter]
            sel = [
                (i, n) for i, n in enumerate(all_names)
                if any(w in n.lower() for w in wanted)
            ]
        else:
            sel = list(enumerate(all_names))
        if not sel:
            logger.warning(
                "CollisionReporter: body_filter %r matched no bodies of %s; "
                "falling back to ALL bodies.",
                self.cfg.body_filter, all_names,
            )
            sel = list(enumerate(all_names))
        self._body_indices = np.asarray([i for i, _ in sel], dtype=np.int64)
        self._body_names = [n for _, n in sel]

    # ----- per-step -------------------------------------------------------
    def log_step(self, step: int, env_ids) -> None:
        """Record one row per env in ``env_ids``. Cheap; no I/O."""
        if self._articulation is None:
            return
        env_ids = [int(e) for e in env_ids]
        if not env_ids:
            return

        # 1. Pull all robot body world positions in ONE tensor read, then
        #    select the filtered subset.
        body_link_state = self._articulation.data.body_link_state_w  # (E, B_total, 13)
        try:
            body_pos = body_link_state[..., :3].detach().cpu().numpy()
        except Exception:
            body_pos = np.asarray(body_link_state[..., :3])
        body_pos = body_pos[:, self._body_indices, :]  # (E, B, 3)
        # WorldState.get_bbox() returns env-relative obstacle boxes by
        # default. Match that frame before comparing robot body centers.
        try:
            origins = self._world.env.scene.env_origins.detach().cpu().numpy()
        except Exception:
            origins = np.asarray(self._world.env.scene.env_origins)
        body_pos = body_pos - origins[:, None, :]

        # 2. Per-step grasped objects per env (excluded from collision check).
        grasped_per_env: dict[int, set[str]] = {}
        for eid in env_ids:
            try:
                g = set(
                    self._world.get_objects_in_contact_with(
                        self.cfg.gripper_body_name,
                        list(self.cfg.obstacle_names),
                        force_threshold=self.cfg.grasp_force_threshold,
                        env_id=eid,
                    )
                )
            except Exception:
                g = set()
            grasped_per_env[eid] = g

        # 3. Compute world-axis AABBs for all obstacles ONCE (batched across
        #    envs). ``world.get_bbox`` returns (num_envs, 8, 3) for the OBB
        #    corners, so a single .min/.max along the corner axis gives the
        #    world AABB for every (obstacle, env) pair.
        obstacle_aabbs = self._all_obstacle_aabbs()  # (M, num_envs, 6) or None per row

        body_radius = float(self.cfg.body_radius_m)
        obs_pad = float(self.cfg.obstacle_padding_m)
        static_excluded = set(self.cfg.static_excluded_names)

        for eid in env_ids:
            grasped = grasped_per_env[eid]
            excluded = grasped | static_excluded
            row_pos = body_pos[eid]  # (B, 3)
            # Per-body, per-obstacle clearance:
            # clearance = dist(point, AABB_inflated) - body_radius
            # negative clearance ⇒ overlap. We track the minimum over all
            # non-excluded obstacles per body.
            B = row_pos.shape[0]
            best_clearance = np.full((B,), np.inf, dtype=np.float32)
            for j, obs_name in enumerate(self.cfg.obstacle_names):
                if obs_name in excluded:
                    continue
                aabb = obstacle_aabbs[j, eid]
                if not np.all(np.isfinite(aabb)):
                    continue
                cx, cy, cz, sx, sy, sz = aabb
                lo = np.array([cx - 0.5 * sx - obs_pad, cy - 0.5 * sy - obs_pad, cz - 0.5 * sz - obs_pad])
                hi = np.array([cx + 0.5 * sx + obs_pad, cy + 0.5 * sy + obs_pad, cz + 0.5 * sz + obs_pad])
                # Vectorised point-to-AABB: dist² = sum(max(lo - p, 0, p - hi)²)
                lo_diff = np.maximum(lo - row_pos, 0.0)
                hi_diff = np.maximum(row_pos - hi, 0.0)
                dist = np.sqrt(np.sum((lo_diff + hi_diff) ** 2, axis=-1)) - body_radius
                best_clearance = np.minimum(best_clearance, dist.astype(np.float32))

            in_coll_mask = best_clearance < 0.0
            self._records.append(
                CollisionRecord(
                    step=int(step),
                    env_id=int(eid),
                    in_collision=bool(np.any(in_coll_mask)),
                    num_links_in_collision=int(np.sum(in_coll_mask)),
                    min_clearance=float(np.min(best_clearance)) if best_clearance.size else float("inf"),
                    body_clearance=best_clearance.copy(),
                    grasped=tuple(sorted(grasped)),
                )
            )

    def _all_obstacle_aabbs(self) -> np.ndarray:
        """Return ``(M, num_envs, 6)`` AABB params: ``(cx, cy, cz, sx, sy, sz)``.

        Rows for missing / zero-volume objects are NaN-filled.
        """
        names = list(self.cfg.obstacle_names)
        if not names:
            return np.empty((0, 0, 6), dtype=np.float32)
        per_name: list[np.ndarray | None] = []
        num_envs = None
        for name in names:
            try:
                corners_world, _centroid = self._world.get_bbox(name)
                arr = corners_world.detach().cpu().numpy().astype(np.float32, copy=False)
            except Exception:
                arr = None
            per_name.append(arr)
            if arr is not None and num_envs is None:
                num_envs = int(arr.shape[0])
        if num_envs is None:
            return np.full((len(names), 0, 6), np.nan, dtype=np.float32)
        out = np.full((len(names), num_envs, 6), np.nan, dtype=np.float32)
        for j, arr in enumerate(per_name):
            if arr is None:
                continue
            lo = arr.min(axis=1)
            hi = arr.max(axis=1)
            size = hi - lo
            valid = np.all(size > 0, axis=-1)
            if not np.any(valid):
                continue
            center = 0.5 * (lo + hi)
            out[j, valid, :3] = center[valid]
            out[j, valid, 3:] = size[valid]
        return out

    # ----- summary + flush ------------------------------------------------
    def summarize(self) -> dict:
        """Aggregate stats per env, computed in-memory."""
        result: dict[int, dict] = {}
        per_env: dict[int, list[CollisionRecord]] = {}
        for r in self._records:
            per_env.setdefault(r.env_id, []).append(r)
        for env_id, recs in per_env.items():
            n_total = len(recs)
            n_coll = sum(1 for r in recs if r.in_collision)
            min_clearance = min(r.min_clearance for r in recs)
            result[env_id] = {
                "steps_total": n_total,
                "steps_in_collision": n_coll,
                "fraction_in_collision": n_coll / n_total if n_total else 0.0,
                "min_clearance_m": float(min_clearance),
            }
        return result

    def flush(self, output_path: str) -> str | None:
        """Write all buffered records to ``output_path`` (a ``.npz`` file).
        Returns the path written, or ``None`` if there was nothing to flush.
        """
        if not self._records:
            return None
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        N = len(self._records)
        steps = np.fromiter((r.step for r in self._records), dtype=np.int32, count=N)
        env_ids = np.fromiter((r.env_id for r in self._records), dtype=np.int32, count=N)
        in_coll = np.fromiter((r.in_collision for r in self._records), dtype=np.bool_, count=N)
        num_links = np.fromiter(
            (r.num_links_in_collision for r in self._records), dtype=np.int16, count=N
        )
        min_clear = np.fromiter(
            (r.min_clearance for r in self._records), dtype=np.float32, count=N
        )
        grasped_count = np.fromiter(
            (len(r.grasped) for r in self._records), dtype=np.int8, count=N
        )
        body_clearance = np.stack([r.body_clearance for r in self._records], axis=0).astype(np.float32)
        # grasped names go in a parallel object-dtype array (msgpack/npz both
        # tolerate this and it's small).
        grasped_names = np.array([",".join(r.grasped) for r in self._records], dtype=object)

        np.savez_compressed(
            output_path,
            steps=steps,
            env_ids=env_ids,
            in_collision=in_coll,
            num_links_in_collision=num_links,
            min_clearance_m=min_clear,
            grasped_count=grasped_count,
            grasped_names=grasped_names,
            body_clearance_m=body_clearance,
            body_names=np.array(self._body_names, dtype=object),
            obstacle_names=np.array(list(self.cfg.obstacle_names), dtype=object),
            body_radius_m=np.float32(self.cfg.body_radius_m),
            obstacle_padding_m=np.float32(self.cfg.obstacle_padding_m),
            saved_unix_ts=np.float64(time.time()),
        )
        self._records.clear()
        return output_path
