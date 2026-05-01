"""nvblox-backed ESDF collision constraint builder for FKC guidance.

The :class:`SDFBuilder` is invoked once per ``Policy.infer`` call from
:class:`robolab_policy_client.pi0_family.Pi0DroidJointposClient`. It collects
the world-frame OBB for every non-grasped scene object via RoboLab's
``WorldState.get_bbox``, ships them to a long-lived nvblox sidecar process
running under ``.venv_nvblox_sidecar``, and returns a dense voxel-grid SDF
ready to attach to the websocket observation as ``fkc/sdf_*`` keys.

The sidecar lives in :mod:`robolab_policy_client.sdf.sidecar_main` and is
designed to be invoked by ``.venv_nvblox_sidecar/bin/python`` (the venv with
``nvblox_torch`` and torch+cu12 installed). Keeping nvblox in a separate
interpreter avoids any conflict with RoboLab's torch/IsaacLab pinning.
"""

from robolab_policy_client.sdf.builder import SDFBuilder

__all__ = ["SDFBuilder"]
