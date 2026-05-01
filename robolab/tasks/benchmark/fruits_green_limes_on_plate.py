# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

from dataclasses import dataclass

import isaaclab.envs.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_on_top, pick_and_place_on_surface
from robolab.core.task.task import Task


@configclass
class FruitsGreenLimesOnPlateTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_on_top,
        params={"object": ["lime01", "lime01_01"], "reference_object": "clay_plates", "logical": "all", "require_gripper_detached": True}
    )

@dataclass
class FruitsGreenLimesOnPlateTask(Task):
    contact_object_list = ["table", "lemon_01", "lemon_02", "lime01", "lime01_01", "orange_01", "orange_02", "pomegranate01", "pumpkinlarge", "pumpkinsmall", "redonion", "serving_bowl", "clay_plates", "wooden_spoons", "spatula", "storage_box"]
    scene = import_scene("fruits_out_of_basket.usda", contact_object_list)
    terminations = FruitsGreenLimesOnPlateTerminations
    instruction = {
        "default": "Put all the green fruit on the plate",
        "vague": "Plate green fruits",
        "specific": "Put all green limes on the table and place them on the flat, beige colored plate",
    }
    episode_length_s: int = 90
    attributes = ['color']

    # Workspace AABB consumed by ``robolab_policy_client.sdf.SDFBuilder``.
    # Slightly larger than BananaInBowl to accommodate the spread of fruit
    # and the storage box.
    sdf_workspace_bounds = ((-0.3, -0.7, -0.05), (0.9, 0.7, 1.2))

    subtasks = [
        pick_and_place_on_surface(
            object=["lime01", "lime01_01"],
            surface="clay_plates",
            logical="all",
            score=1.0
        )
    ]
