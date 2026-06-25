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

# --- RL reward-shaping knobs (task-owned) ----------------------------------------------------
# Dense shaping signals for the pick-and-place phase, ADDED to the verl RL reward. They belong
# to THIS task (they encode what "lifted"/"dropped" means for the ranch bottle on the kitchen
# counter), so they live here rather than in the generic verl env wrapper. The verl side only
# evaluates whatever reward terms get_rewards_cfg() declares (see arena_env._calc_shaping_reward)
# and routes them through a separate reward channel, so they do NOT affect success/done.
#   * object_lifted: + reward while the bottle is raised >= LIFT_HEIGHT_M above its post-reset
#     resting height (encourages a clean grasp + lift off the counter).
#   * object_dropped: - penalty while the bottle has fallen to/below the floor (world z below the
#     background's object_min_z), i.e. it was dropped.
# Per-(chunk-)step continuous signals; keep the weights modest vs the +1 success reward.
OBJECT_LIFTED_HEIGHT_M = 0.05
OBJECT_LIFTED_REWARD_WEIGHT = 0.25
OBJECT_DROPPED_PENALTY_WEIGHT = 0.5


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
        from isaaclab.managers import EventTermCfg
        from isaaclab.managers import ObservationGroupCfg as ObsGroup
        from isaaclab.managers import ObservationTermCfg as ObsTerm
        from isaaclab.managers import RewardTermCfg, SceneEntityCfg
        from isaaclab.utils import configclass

        from isaaclab_arena.utils.configclass import combine_configclass_instances

        from isaaclab_arena.assets.object_reference import ObjectReference, OpenableObjectReference
        from isaaclab_arena.tasks.observations import observations
        from isaaclab_arena.tasks.events import capture_object_init_z
        from isaaclab_arena.tasks.rewards.lift_object_rewards import object_dropped_below, object_lifted_above_reset
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
                pickup_object_name: str | None = None,
                object_min_z: float | None = None,
            ):
                super().__init__(
                    subtasks=subtasks, episode_length_s=episode_length_s, desired_subtask_success_state=[True, True]
                )
                # Critic-only privileged obs (asymmetric AC). Held here so the env builder's
                # combine_configclass_instances(scene, embodiment, task) picks it up as an extra
                # observation group; None ⇒ task contributes no observations (legacy behaviour).
                self._observation_cfg = observation_cfg
                # Pick-up object + floor threshold for the dense shaping rewards (see
                # get_rewards_cfg / get_events_cfg). Both are properties of THIS task.
                self._pickup_object_name = pickup_object_name
                self._object_min_z = object_min_z

            def get_observation_cfg(self):
                return self._observation_cfg

            def get_rewards_cfg(self):
                # Task-owned dense shaping for the pick-and-place phase: + while the bottle is
                # lifted off the counter, - while it has been dropped to the floor. The verl RL
                # wrapper routes these through a SEPARATE reward channel (they do not change
                # success/done); pure-Arena eval/mimic just ignore the reward values. Disabled
                # when the pick-up object name was not supplied (e.g. object_set).
                if self._pickup_object_name is None:
                    return None

                @configclass
                class ShapingRewardsCfg:
                    object_lifted: RewardTermCfg = MISSING
                    object_dropped: RewardTermCfg = MISSING

                rewards = ShapingRewardsCfg()
                rewards.object_lifted = RewardTermCfg(
                    func=object_lifted_above_reset,
                    weight=OBJECT_LIFTED_REWARD_WEIGHT,
                    params={
                        "object_cfg": SceneEntityCfg(self._pickup_object_name),
                        "lift_height": OBJECT_LIFTED_HEIGHT_M,
                    },
                )
                rewards.object_dropped = RewardTermCfg(
                    func=object_dropped_below,
                    weight=-OBJECT_DROPPED_PENALTY_WEIGHT,
                    params={
                        "object_cfg": SceneEntityCfg(self._pickup_object_name),
                        "minimum_height": self._object_min_z,
                    },
                )
                return rewards

            def get_events_cfg(self):
                # Combine the sequential-task events (subtask-state reset, etc.) with a reset
                # event that captures the bottle's resting world-z as the per-env lift baseline
                # used by object_lifted_reward. Task events run after scene/embodiment events
                # (after the object is placed), so the baseline reflects the randomized start.
                base_events = super().get_events_cfg()
                if self._pickup_object_name is None:
                    return base_events

                @configclass
                class ShapingEventsCfg:
                    capture_object_init_z: EventTermCfg = MISSING

                shaping_events = ShapingEventsCfg()
                shaping_events.capture_object_init_z = EventTermCfg(
                    func=capture_object_init_z,
                    mode="reset",
                    params={"object_cfg": SceneEntityCfg(self._pickup_object_name)},
                )
                return combine_configclass_instances("EventsCfg", base_events, shaping_events)

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

        # Head-camera offset tuned so a single view captures both the ranch bottle on the
        # counter and the interior of the fridge on the robot's right: the camera is turned
        # ~22 deg to the right and shifted slightly right for 6cm (head_yaw_link frame, opengl).
        camera_offset = Pose(
            position_xyz=(0.12515, -0.06, 0.06776),
            rotation_xyzw=(-0.04096, -0.28352, -0.79792, 0.53034),
        )
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(
            enable_cameras=args_cli.enable_cameras, camera_offset=camera_offset
        )
        # Add a right-wrist camera (in addition to the head POV camera) so that replayed demos also
        # capture a close-up of the fridge interior and the bottle being placed. The extra view is
        # exposed as camera_obs["right_wrist_cam_rgb"] and recorded alongside the head camera.
        if args_cli.enable_cameras and hasattr(embodiment, "camera_config"):
            from isaaclab_arena.embodiments.gr1t2.gr1t2 import GR1T2WristCameraCfg

            wrist_cam_config = GR1T2WristCameraCfg()
            # Mirror the head-camera settings the embodiment applied to its default camera config.
            wrist_cam_config._is_tiled_camera = getattr(embodiment.camera_config, "_is_tiled_camera", False)
            wrist_cam_config._camera_offset = getattr(embodiment.camera_config, "_camera_offset", camera_offset)
            wrist_cam_config.__post_init__()
            embodiment.camera_config = wrist_cam_config
        # Slightly widen the field of view (lower focal length) so both targets fit in frame.
        if hasattr(embodiment, "camera_config") and hasattr(embodiment.camera_config, "robot_pov_cam"):
            embodiment.camera_config.robot_pov_cam.spawn.focal_length = 13.0
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
                    object_pos = ObsTerm(
                        func=observations.object_position_in_static_frame,
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

        # Create sequential task. Pass the pick-up object + floor threshold so the task can
        # declare its dense shaping rewards (object_lifted / object_dropped). Shaping is only
        # wired for a single rigid object (skipped for --object_set, whose scene entity is a
        # RigidObjectCollection with a different pose API).
        shaping_object_name = None if (args_cli.object_set and len(args_cli.object_set) > 0) else pickup_object.name
        sequential_task = PutAndCloseDoorTask(
            subtasks=[pick_and_place_task, close_door_task],
            episode_length_s=10.0,
            observation_cfg=privileged_observation_cfg,
            pickup_object_name=shaping_object_name,
            object_min_z=kitchen_background.object_min_z,
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
