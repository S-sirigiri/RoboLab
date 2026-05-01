# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

import logging

import numpy as np
from openpi_client import image_tools, websocket_client_policy

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)


class Pi0DroidJointposClient(InferenceClient):
    # Per-variant action horizons. One Pi0 server class serves multiple trained
    # variants; each has its own training-time action_horizon. Callers pass
    # ``policy_variant`` to select the right default, or override directly via
    # ``open_loop_horizon``.
    DEFAULT_HORIZONS: dict[str, int] = {
        "pi0": 10,
        "pi0_fast": 10,
        "paligemma": 10,
        "paligemma_fast": 10,
        "pi05": 15,
    }
    FALLBACK_HORIZON: int = 15

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int | None = None,
        remote_uri: str | None = None,
        policy_variant: str = "pi05",
        sdf_builder=None,
    ) -> None:
        super().__init__()
        if open_loop_horizon is None:
            open_loop_horizon = self.DEFAULT_HORIZONS.get(policy_variant, self.FALLBACK_HORIZON)
        self.open_loop_horizon = int(open_loop_horizon)
        self.policy_variant = policy_variant
        self._remote_uri = remote_uri
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._display = remote_uri if remote_uri is not None else f"{remote_host}:{remote_port}"
        # Optional :class:`robolab_policy_client.sdf.SDFBuilder`. When set,
        # the eval loop calls :meth:`precompute_sdf_batch` once per step
        # *before* the per-env infer loop, populating ``_sdf_cache`` with
        # one SDF per env that needs a replan. ``_pack_request`` then just
        # pops the cached SDF — no per-env IPC.
        self.sdf_builder = sdf_builder
        # env_id -> dict of fkc/* arrays (consumed by next ``_pack_request``).
        self._sdf_cache: dict[int, dict] = {}

        print(f"[{self.__class__.__name__}] Awaiting for server on {self._display} to be ready...")
        self.client = self._connect()
        print(f"[{self.__class__.__name__}] Connected to {self._display}.")

    def precompute_sdf_batch(self, env_ids) -> None:
        """Build SDFs for all ``env_ids`` that need a replan, in one IPC.

        Called by the eval loop once per env step *before* the per-env
        ``infer`` loop. Filters ``env_ids`` down to those that actually
        need a fresh action chunk via :meth:`_needs_refresh`, then issues
        a single batched sidecar request that produces one SDF per env.

        No-op when ``self.sdf_builder is None`` so non-SDF backends and
        baseline runs pay no cost.
        """
        if self.sdf_builder is None:
            return
        refresh_ids = [int(eid) for eid in env_ids if self._needs_refresh(int(eid))]
        if not refresh_ids:
            return
        try:
            results = self.sdf_builder.build_batch(refresh_ids)
        except Exception:
            logger.exception(
                "SDFBuilder.build_batch failed for env_ids=%s; falling back to "
                "no-SDF requests this step.", refresh_ids,
            )
            return
        for eid, result in zip(refresh_ids, results):
            self._sdf_cache[eid] = result

    def _connect(self):
        if self._remote_uri is not None:
            return websocket_client_policy.WebsocketClientPolicy(self._remote_uri)
        return websocket_client_policy.WebsocketClientPolicy(self._remote_host, self._remote_port)

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        """Call server, reconnecting up to ``max_retries`` times on connection drop."""
        import websockets.exceptions

        for attempt in range(max_retries):
            try:
                return self.client.infer(request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as e:
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "[%s] Connection lost (%s), reconnecting (attempt %d/%d)...",
                    self.__class__.__name__, e, attempt + 1, max_retries,
                )
                self.client = self._connect()
                # Flush chunk cache so all envs re-request on next step
                self._chunks.clear()
                self._counters.clear()

    # ---- required hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        right_image = raw_obs["image_obs"]["external_cam"][env_id].clone().detach().cpu().numpy()
        wrist_image = raw_obs["image_obs"]["wrist_cam"][env_id].clone().detach().cpu().numpy()

        robot_state = raw_obs["proprio_obs"]
        joint_position = robot_state["arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        gripper_position = robot_state["gripper_pos"][env_id].clone().detach().cpu().numpy()

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
            "_env_id": env_id,  # threaded through to _pack_request for SDF cache lookup
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        request = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                extracted_obs["right_image"], 224, 224
            ),
            "observation/wrist_image_left": image_tools.resize_with_pad(
                extracted_obs["wrist_image"], 224, 224
            ),
            "observation/joint_position": extracted_obs["joint_position"],
            "observation/gripper_position": extracted_obs["gripper_position"],
            "prompt": instruction,
        }
        if self.sdf_builder is not None:
            # Pop the per-env SDF that was precomputed for this step by
            # ``precompute_sdf_batch``. If the eval loop forgot to call it
            # (e.g. an older driver or an interactive REPL), fall back to a
            # single-env synchronous build so behaviour is still correct,
            # just slower.
            env_id = int(extracted_obs.get("_env_id", 0))
            fkc_extras = self._sdf_cache.pop(env_id, None)
            if fkc_extras is None:
                try:
                    fkc_extras = self.sdf_builder.build(env_id=env_id)
                except Exception:
                    logger.exception(
                        "SDFBuilder.build failed for env_id=%d; sending request without SDF",
                        env_id,
                    )
                    fkc_extras = None
            if fkc_extras is not None:
                request.update(fkc_extras)
        return request

    def _query_server(self, request: dict) -> dict:
        return self._infer_with_retry(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        return np.asarray(response["actions"])

    # ---- optional hooks -----------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        chunk = chunk.copy()
        chunk[..., -1] = (chunk[..., -1] > 0.5).astype(chunk.dtype)
        return chunk

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        img1 = image_tools.resize_with_pad(extracted_obs["right_image"], 224, 224)
        img2 = image_tools.resize_with_pad(extracted_obs["wrist_image"], 224, 224)
        return np.concatenate([img1, img2], axis=1)


if __name__ == "__main__":
    import time

    import torch

    client = Pi0DroidJointposClient()
    fake_obs = {
        "image_obs": {
            "external_cam": [torch.zeros((224, 224, 3), dtype=torch.uint8)],
            "wrist_cam": [torch.zeros((224, 224, 3), dtype=torch.uint8)],
        },
        "proprio_obs": {
            "arm_joint_pos": torch.zeros((1, 7), dtype=torch.float32),
            "gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
        },
    }
    fake_instruction = "pick up the object"

    start = time.time()
    client.infer(fake_obs, fake_instruction)  # warm up
    num = 20
    for _ in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
