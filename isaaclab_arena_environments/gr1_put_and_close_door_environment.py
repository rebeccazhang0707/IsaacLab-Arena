# Copyright (c) 2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import math
from typing import TYPE_CHECKING

from isaaclab_arena.tasks.common.mimic_default_params import MIMIC_DATAGEN_CONFIG_DEFAULTS
from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase

if TYPE_CHECKING:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment


RANDOMIZATION_HALF_RANGE_X_M = 0.03
RANDOMIZATION_HALF_RANGE_Y_M = 0.01
RANDOMIZATION_HALF_RANGE_Z_M = 0.0


class GR1PutAndCloseDoorEnvironment(ExampleEnvironmentBase):
    """
    A sequential task environment with two subtasks for GR1 humanoid robot:
    1. Pick and place object into the refrigerator shelf
    2. Close the refrigerator door
    The refrigerator starts open, the robot places the object inside, then closes it.
    Uses the lightwheel robocasa kitchen background.
    """

    name = "put_item_in_fridge_and_close_door"

    def get_env(self, args_cli: argparse.Namespace) -> IsaacLabArenaEnvironment:
        import isaaclab.envs.mdp as mdp_isaac_lab
        from dataclasses import MISSING

        from isaaclab.envs.mimic_env_cfg import MimicEnvCfg
        from isaaclab.managers import ObservationGroupCfg as ObsGroup
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils import configclass

        from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
        from isaaclab_arena.tasks.observations import observations
        from isaaclab_arena.assets.object_set import RigidObjectSet
        from isaaclab_arena.embodiments.common.arm_mode import ArmMode
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.relations.relations import (
            AtPosition,
            IsAnchor,
            On,
            RandomAroundSolution,
            RotateAroundSolution,
        )
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.tasks.close_door_task import CloseDoorTask
        from isaaclab_arena.tasks.pick_and_place_task import PickAndPlaceTask
        from isaaclab_arena.tasks.sequential_task_base import SequentialTaskBase
        from isaaclab_arena.tasks.task_base import TaskBase
        from isaaclab_arena.utils.pose import Pose

        # Custom task class for this environment
        class PutAndCloseDoorTask(SequentialTaskBase):
            def __init__(
                self,
                subtasks: list[TaskBase],
                episode_length_s: float | None = None,
                observation_cfg: object | None = None,
            ):
                super().__init__(
                    subtasks=subtasks, episode_length_s=episode_length_s, desired_subtask_success_state=[True, True]
                )
                # Critic-only privileged obs (asymmetric AC). Held here so the env builder's
                # combine_configclass_instances(scene, embodiment, task) picks it up as an extra
                # observation group; None ⇒ task contributes no observations (legacy behaviour).
                self._observation_cfg = observation_cfg

            def get_observation_cfg(self):
                return self._observation_cfg

            def get_viewer_cfg(self):
                return self.subtasks[0].get_viewer_cfg()

            def get_prompt(self):
                return None

            def get_mimic_env_cfg(self, arm_mode: ArmMode):
                mimic_env_cfg = PutAndCloseDoorTaskMimicEnvCfg()
                mimic_env_cfg.subtask_configs = self.combine_mimic_subtask_configs(ArmMode.RIGHT)

                # Override default subtask term offset range and action noise
                for eef_name, subtask_list in mimic_env_cfg.subtask_configs.items():
                    for subtask_config in subtask_list:
                        subtask_config.subtask_term_offset_range = (0, 0)
                        subtask_config.action_noise = 0.003

                return mimic_env_cfg

        @configclass
        class PutAndCloseDoorTaskMimicEnvCfg(MimicEnvCfg):
            """
            Isaac Lab Mimic environment config class for GR1 put and close door task.
            """

            def __post_init__(self):
                # post init of parents
                super().__post_init__()

                # Override the existing values
                self.datagen_config.name = "put_and_close_door_task_D0"
                # Use default mimic datagen config parameters
                for key, value in MIMIC_DATAGEN_CONFIG_DEFAULTS.items():
                    setattr(self.datagen_config, key, value)

        camera_offset = Pose(position_xyz=(0.12515, 0.0, 0.06776), rotation_xyzw=(0.11204, -0.17712, -0.79108, 0.57469))
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras, camera_offset=camera_offset
        )
        kitchen_background = self.asset_registry.get_asset_by_name("lightwheel_robocasa_kitchen")(
            style_id=args_cli.kitchen_style
        )

        kitchen_counter_top = ObjectReference(
            name="kitchen_counter_top",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/counter_right_main_group/top_geometry",
            parent_asset=kitchen_background,
        )
        kitchen_counter_top.add_relation(IsAnchor())

        light = self.asset_registry.get_asset_by_name("light")()

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        # Set initial poses
        embodiment.set_initial_pose(
            Pose(
                position_xyz=(3.943, -1.0, 0.995),
                rotation_xyzw=(0.0, 0.0, 0.7071068, 0.7071068),
            )
        )

        # Create refrigerator reference (OpenableObjectReference)
        refrigerator = OpenableObjectReference(
            name="refrigerator",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/fridge_main_group",
            parent_asset=kitchen_background,
            openable_joint_name="fridge_door_joint",
            openable_threshold=0.5,
        )

        # Create refrigerator shelf reference (destination for pick and place)
        refrigerator_shelf = ObjectReference(
            name="refrigerator_shelf",
            prim_path="{ENV_REGEX_NS}/lightwheel_robocasa_kitchen/fridge_main_group/Refrigerator034",
            parent_asset=kitchen_background,
        )

        if args_cli.object_set is not None and len(args_cli.object_set) > 0:
            objects = [self.asset_registry.get_asset_by_name(obj)() for obj in args_cli.object_set]
            pickup_object = RigidObjectSet(name="object_set", objects=objects)
        else:
            pickup_object = self.asset_registry.get_asset_by_name(args_cli.object)()

        pickup_object.add_relation(On(kitchen_counter_top))
        pickup_object.add_relation(AtPosition(x=4.05, y=-0.58))
        # Consider changing to other values for different objects, below is for ranch dressing bottle.
        yaw_rad = math.radians(-111.55)
        pickup_object.add_relation(RotateAroundSolution(yaw_rad=yaw_rad))
        pickup_object.add_relation(
            RandomAroundSolution(x_half_m=RANDOMIZATION_HALF_RANGE_X_M, y_half_m=RANDOMIZATION_HALF_RANGE_Y_M)
        )
        scene = Scene(
            assets=[kitchen_background, kitchen_counter_top, pickup_object, light, refrigerator, refrigerator_shelf]
        )

        # Create pick and place task
        pick_and_place_task = PickAndPlaceTask(
            pick_up_object=pickup_object,
            destination_object=refrigerator,
            destination_location=refrigerator_shelf,
            background_scene=kitchen_background,
        )

        # Create close door task
        close_door_task = CloseDoorTask(
            openable_object=refrigerator,
            closedness_threshold=0.10,
            reset_openness=0.5,
        )

        # --- Asymmetric actor-critic: privileged critic-only observation group ---
        # A single `critic_privileged` group fed ONLY to the SAC critic (the policy/proprio
        # groups are unchanged). It carries the manipulated object's pose RELATIVE TO THE
        # PLACEMENT TARGET (the fridge shelf) plus the fridge door joint angle — goal-centric,
        # well-scaled signals for value learning. The shelf is a static reference (a fixed
        # XForm with no runtime root_pose_w), so its pose is captured once here and rebuilt at
        # runtime from env_origins; the door joint is read live from the fridge articulation.
        shelf_pose = refrigerator_shelf.get_initial_pose()
        shelf_pos = tuple(shelf_pose.position_xyz)
        _qx, _qy, _qz, _qw = shelf_pose.rotation_xyzw  # arena Pose stores xyzw; Isaac math wants wxyz
        shelf_quat_wxyz = (_qw, _qx, _qy, _qz)

        @configclass
        class PrivilegedObservationsCfg:
            """Critic-only privileged observations for the put-and-close-door task."""

            critic_privileged: ObsGroup = MISSING

            def __init__(self, object_name: str, fridge_name: str, door_joint_name: str):
                @configclass
                class CriticPrivilegedCfg(ObsGroup):
                    # Object pose (pos + quat = 7) in the fridge-shelf frame.
                    object_pose = ObsTerm(
                        func=observations.object_pose_in_static_frame,
                        params={
                            "object_cfg": SceneEntityCfg(object_name),
                            "frame_pos": shelf_pos,
                            "frame_quat_wxyz": shelf_quat_wxyz,
                        },
                    )
                    # Fridge door joint angle (1), read live from the articulation.
                    door_joint = ObsTerm(
                        func=mdp_isaac_lab.joint_pos,
                        params={"asset_cfg": SceneEntityCfg(fridge_name, joint_names=[door_joint_name])},
                    )

                    def __post_init__(self):
                        self.enable_corruption = False
                        self.concatenate_terms = True

                self.critic_privileged = CriticPrivilegedCfg()

        # NOTE: assumes a single rigid pickup object (the default). For --object_set the name
        # resolves to a RigidObjectCollection whose pose API differs; extend the mdp term then.
        privileged_observation_cfg = PrivilegedObservationsCfg(
            object_name=pickup_object.name,
            fridge_name=refrigerator.name,
            door_joint_name="fridge_door_joint",
        )

        # Create sequential task
        sequential_task = PutAndCloseDoorTask(
            subtasks=[pick_and_place_task, close_door_task],
            episode_length_s=10.0,
            observation_cfg=privileged_observation_cfg,
        )

        # Create and return environment
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=sequential_task,
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--object",
            type=str,
            default="ranch_dressing_hope_robolab",
            help="Object to pick and place",
        )
        parser.add_argument(
            "--object_set",
            nargs="+",
            type=str,
            default=None,
            help=(
                "Used in heterogeneous environments where each environment has a different object spawned from this"
                " set."
            ),
        )
        parser.add_argument(
            "--kitchen_style", type=int, default=2, help="Kitchen style ID for lightwheel robocasa kitchen"
        )
        parser.add_argument("--teleop_device", type=str, default=None, help="Teleoperation device to use")
        parser.add_argument("--embodiment", type=str, default="gr1_pink", help="Robot embodiment to use")
