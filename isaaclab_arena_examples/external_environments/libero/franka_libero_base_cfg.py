# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Base LIBERO task config + scene definitions (self-contained Arena port).

Ported from ``isaaclab_playground...config.franka.franka_libero_env_cfg``. This contains
only the pieces reused by the Abs-IK RL env config:

* :class:`LiberoTaskConfig` — loads a single (suite, task_id) from the LIBERO JSON specs.
* :class:`EventCfgFrankaPanda` — the imitation-learning reset events (kept for parity).
* :class:`ActionsCfg` — placeholder arm/gripper action slots filled in by the env cfg.
* The four workspace scene cfgs (kitchen / living-room / floor / study table).

The teleop / OSC / multitask observation machinery from the original file is intentionally
omitted.
"""

import json
import os
from dataclasses import MISSING

from . import mdp

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events


@configclass
class LiberoTaskConfig:
    """Configuration for Libero task parameters (single task selected via env vars)."""

    config_dir: str = os.getenv("LIBERO_CONFIG_DIR", os.path.abspath("benchmarks/datasets/libero/config"))
    assets_dir: str = os.getenv("LIBERO_ASSETS_DATA_DIR", os.path.abspath("benchmarks/datasets/libero/USD"))

    def __post_init__(self):
        # Load task info
        self.task_suite = os.getenv("LIBERO_TASK_SUITE", "libero_10")
        self.task_id = os.getenv("LIBERO_TASK_ID", "0")
        self.task_id_int = int(self.task_id)
        task_suite_info_file = os.path.join(self.config_dir, f"{self.task_suite}.json")

        with open(task_suite_info_file) as f:
            task_suite_info = json.load(f)
            self.task_info = task_suite_info["tasks"][self.task_id_int]
            self.workspace_name = self.task_info["workspace_name"]
            self.fixtures = self.task_info["fixtures"]  # static objects/colliders: table, floor, etc.
            self.objects = self.task_info["objects"]  # dynamic objects: cream cheese, basket, etc.
            self.regions = self.task_info["regions"]  # regions: initial position of objects
            self.obj_of_interest = self.task_info["obj_of_interest"]  # objects of interest: objects to grasp.
            self.targets = self.task_info["targets"]  # targets: objects to place onto.
            self.goals = self.task_info["goals"]  # goals: trajectory success conditions.
            self.robot_base_pos = self.task_info["robot_base_pos"]  # robot base position: [x, y, z]
            self.robot_base_ori = self.task_info["robot_base_ori"]  # robot base orientation: [w, x, y, z]

    def get_task_assignment_for_env(self, env_idx: int) -> tuple[str, int]:
        """Expose a multitask-compatible assignment API for single-task environments."""

        del env_idx
        return self.task_suite, self.task_id_int


@configclass
class EventCfgFrankaPanda:
    """Configuration for events (imitation-learning style reset)."""

    init_franka_arm_pose = EventTerm(
        func=franka_stack_events.set_default_joint_pose,
        mode="reset",
        params={
            "default_pose": [
                -0.019882839432642387,
                -0.18734066496238144,
                0.0076694004538321505,
                -2.4034025985475256,
                0.004964681607500244,
                2.2453365042123963,
                0.7948478983158621,
                0.04,
                0.04,
            ],
        },
    )

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Create individual event terms for each object
    def __post_init__(self):

        libero_config = LiberoTaskConfig()
        # Create event terms for each object
        for obj in libero_config.objects.items():
            obj_name = obj[0]
            init_region_name = obj[1]["initial_region"]

            # Create a unique name for each object's event
            event_name = f"init_{obj_name}_pose"

            # Create the event term
            event_term = EventTerm(
                func=franka_stack_events.randomize_object_pose,
                mode="reset",
                params={
                    "pose_range": libero_config.regions[init_region_name]["pose_range"],
                    "asset_cfgs": [SceneEntityCfg(obj_name)],
                },
            )

            # Add the event term to the class
            setattr(self, event_name, event_term)


##
# Scene definition
##


@configclass
class KitchenTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the Kitchen Table scene with a robot and objects."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=200.0),
    )

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        # add table
        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(0.0, 0.0, 0.845)
            ),  # make sure the knob is not underneath the table, otherwise the knob will be stuck
            spawn=UsdFileCfg(
                usd_path=f"{libero_config.assets_dir}/kitchen_table/kitchen_table.usd", scale=(0.02, 0.02, 0.02)
            ),
        )


@configclass
class LivingRoomTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the Living Room Table scene with a robot and an object."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=200.0),
    )

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        # add table
        self.table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.14 * 1.5, -0.005), rot=(0.707, 0.0, 0.0, 0.707)),
            # table size: [1.2, 0.8, 0.3]
            spawn=UsdFileCfg(
                usd_path=f"{libero_config.assets_dir}/living_room_table/living_room_table.usd", scale=(1.5, 1.5, 1.5)
            ),
        )


@configclass
class FloorSceneCfg(InteractiveSceneCfg):
    """Configuration for the Floor scene with a robot and objects."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=200.0),
    )

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        # add floor
        self.floor = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Floor",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -0.025)),
            # floor size: [6.0, 6.0, 0.025]
            spawn=UsdFileCfg(usd_path=f"{libero_config.assets_dir}/floor/floor.usd", scale=(1.0, 1.0, 1.0)),
        )


@configclass
class StudyTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the Study Room Table scene with a robot and objects."""

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=200.0),
    )

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        # add table
        self.study_table = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/study_table",
            init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.2, 0, 0.867 - 0.85)),
            spawn=UsdFileCfg(usd_path=f"{libero_config.assets_dir}/study_table/study_table.usd", scale=(1.0, 1.0, 1.0)),
        )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
