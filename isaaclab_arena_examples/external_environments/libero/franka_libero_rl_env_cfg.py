# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Compact Franka Libero env config for VLA RL post-training (self-contained Arena port).

Ported verbatim (modulo local imports) from
``isaaclab_playground...config.franka.franka_libero_rl_env_cfg``. Compared to the full
``franka_libero_env_cfg.py``:

- Built-in agentview + wrist cameras (no separate CameraEnvCfg layer)
- Policy-only observations (no subtask / grasp-detection obs group)
- Minimal sensors: ee_frame + goal contact sensors only
- No teleop devices, no debug frame transformers
- Single Abs-IK action space (what VLA models use)
"""

import os
from dataclasses import MISSING

from . import mdp
from .camera_factory import CameraConfigFactory
from .quat_utils import wxyz_to_xyzw

import isaaclab.sim as sim_utils
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from .franka_asset import FRANKA_PANDA_LIBERO_HIGH_PD_CFG  # isort: skip

from isaaclab.managers import EventTermCfg as EventTerm

from .franka_libero_base_cfg import (
    LiberoTaskConfig,
    EventCfgFrankaPanda,  # noqa: F401
    ActionsCfg,
    KitchenTableSceneCfg,
    LivingRoomTableSceneCfg,
    FloorSceneCfg,
    StudyTableSceneCfg,
)

from isaaclab_tasks.manager_based.manipulation.stack.mdp import franka_stack_events  # noqa: F401

from .multitask_utils import (
    ARTICULATION_DEFAULT_JOINTS,
    ARTICULATION_TYPES,
)

# Agentview camera parameters per workspace
_AGENTVIEW_PARAMS: dict[str, dict] = {
    "living_room_table": dict(
        pos=(0.6065773716836134, 0.0, 0.96),
        rot=(0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979),
    ),
    "kitchen_table": dict(
        pos=(0.6586131746834771, 0.0, 1.6103500240372423),
        rot=(0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349),
    ),
    "table": dict(
        pos=(0.6586131746834771, 0.0, 1.6103500240372423),
        rot=(0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349),
    ),
    "floor": dict(
        pos=(0.8965773716836134, 5.216182733499864e-07, 0.65),
        rot=(0.6182166934013367, 0.3432307541370392, 0.3432314395904541, 0.6182177066802979),
    ),
    "study_table": dict(
        pos=(0.4586131746834771, 0.0, 1.6103500240372423),
        rot=(0.6380177736282349, 0.3048497438430786, 0.30484986305236816, 0.6380177736282349),
    ),
}

# Rigid-body prim paths used by goal contact sensors
_RIGID_BODY_PRIM_NAMES: dict[str, str] = {
    "flat_stove_1/burnerplate": "flat_stove_1/flat_stove/burnerplate",
    "microwave_1": "microwave_1/microwave/microwave/Xform",
    "white_cabinet_1": "white_cabinet_1/wooden_cabinet_base_col/wooden_cabinet_bottom/drawer_bottom",
    "wooden_cabinet_1/drawer_top": "wooden_cabinet_1/wooden_cabinet_base_col/wooden_cabinet_top/drawer_top",
    "wooden_cabinet_1": "wooden_cabinet_1/wooden_cabinet_base_col/wooden_cabinet_base_col",
}

# Per-suite episode horizon (seconds). The horizon is a property of the LIBERO task
# suite, not a caller-supplied knob: LIBERO episodes are short, fixed-horizon
# manipulation tasks, so the env cfg owns the value
_EPISODE_LENGTH_S_BY_SUITE: dict[str, float] = {"libero_10": 26.0}
_DEFAULT_EPISODE_LENGTH_S: float = 10.0


def episode_length_s_for_suite(task_suite: str) -> float:
    """Return the episode horizon (s) the given LIBERO task suite defines."""
    return _EPISODE_LENGTH_S_BY_SUITE.get(task_suite, _DEFAULT_EPISODE_LENGTH_S)


# Default object rigid-body properties
_OBJECT_RIGID_PROPS = RigidBodyPropertiesCfg(
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=1,
    max_angular_velocity=1000.0,
    max_linear_velocity=1000.0,
    max_depenetration_velocity=5.0,
    disable_gravity=False,
)


##
# Observations — policy only, no subtask grasp detection
##


@configclass
class RLObservationsCfg:
    """Minimal observations for VLA RL / Arena GR00T closed-loop.

    Layout matches the G1/GR1 Arena convention so consumers can share one abstraction:

    - ``policy``: concatenated eef_pose(7) + gripper_pos(2) Box (task-space state).
    - ``camera_obs``: per-camera RGB dict with ``{sensor}_{dtype}`` keys
      (``agentview_cam_rgb``, ``eye_in_hand_cam_rgb``), ``concatenate_terms=False``.

    Scene entity names stay ``agentview_cam`` / ``eye_in_hand_cam``; only the obs-term
    field names carry the ``_rgb`` suffix (same as ``make_camera_observation_cfg``).
    """

    @configclass
    class PolicyCfg(ObsGroup):

        eef_pose = ObsTerm(func=mdp.ee_frame_pose_in_base_frame)
        gripper_pos = ObsTerm(func=mdp.gripper_pos)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CameraObsCfg(ObsGroup):
        """Per-camera RGB group aligned with Arena ``camera_obs`` / ``*_rgb`` naming."""

        agentview_cam_rgb = ObsTerm(
            func=mdp.image,  # type: ignore[attr-defined]
            params={
                "sensor_cfg": SceneEntityCfg("agentview_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )
        eye_in_hand_cam_rgb = ObsTerm(
            func=mdp.image,  # type: ignore[attr-defined]
            params={
                "sensor_cfg": SceneEntityCfg("eye_in_hand_cam"),
                "data_type": "rgb",
                "normalize": False,
            },
        )

        def __post_init__(self):
            self.enable_corruption = True
            # Keep per-camera tensors (do NOT flatten/concatenate images into one vector).
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera_obs: CameraObsCfg = CameraObsCfg()


##
# Terminations
##

@configclass
class RLTerminationsCfg:
    """Termination terms for VLA RL post-training.

    Must include a ``success`` DoneTerm (same predicate as the sparse reward): LIBERO
    otherwise only ends via ``time_out``, so a solved episode keeps stepping until the
    horizon and reports ``truncated=True, terminated=False`` even though
    ``libero_goals_reached`` already fired. Mirrors the IL ``TerminationsCfg.success``.
    """

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        self.success = DoneTerm(
            func=mdp.libero_goals_reached,
            params={"goals": libero_config.goals},
        )

##
# Rewards
##


@configclass
class RLRewardsCfg:
    """Reward terms for VLA RL post-training: sparse goal-completion signal."""

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        self.libero_goals_reached = RewTerm(
            func=mdp.libero_goals_reached, params={"goals": libero_config.goals}, weight=20.0
        )

##
# Events — RL training with demo-sampled robot init + optional noise
##


@configclass
class RLEventCfgFrankaPanda:
    """Event configuration for single-task Libero RL training.

    Defaults to the same fixed arm pose + object randomization as the IL env.
    Optionally, the full scene can be restored from HDF5 demo initial_state and
    object randomization can be toggled independently.
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    def __post_init__(self):
        libero_config = LiberoTaskConfig()
        randomize_object_pose = os.getenv("LIBERO_RANDOMIZE_OBJECT_POSE", "False").lower() in ["true", "1", "t"]

        # reset all robots & objects to their initial states sampled from demonstrations
        self.reset_libero_initial_states = EventTerm(
            func=mdp.reset_libero_scene_to_initial_state,
            mode="reset",
            params={
                "datasets_root": os.getenv(
                    "LIBERO_ASSEMBLED_DATASET_DIR",
                    os.path.abspath("benchmarks/datasets/libero/assembled_hdf5"),
                ),
                "include_articulations": True,
                "include_rigid_objects": True,
                "robot_joint_noise_std": float(os.getenv("ROBOT_INIT_NOISE_STD", "0.0")),
            },
        )

        if not randomize_object_pose:
            return

        for obj_name, obj_info in libero_config.objects.items():
            obj_type = obj_info["type"]
            init_region_name = obj_info["initial_region"]
            pose_range = libero_config.regions[init_region_name]["pose_range"]

            setattr(
                self,
                f"init_{obj_name}_pose",
                EventTerm(
                    func=franka_stack_events.randomize_object_pose,
                    mode="reset",
                    params={
                        "pose_range": pose_range,
                        "asset_cfgs": [SceneEntityCfg(obj_name)],
                    },
                ),
            )

            if obj_type in ARTICULATION_TYPES:
                jpr = pose_range.get("joint_pos_range", {})
                joint_names = list(ARTICULATION_DEFAULT_JOINTS.get(obj_type, {}).keys())
                if jpr and joint_names:
                    setattr(
                        self,
                        f"init_{obj_name}_joints",
                        EventTerm(
                            func=mdp.reset_articulation_joints,
                            mode="reset",
                            params={
                                "asset_cfg": SceneEntityCfg(obj_name),
                                "joint_pos_range": jpr,
                                "joint_names": joint_names,
                            },
                        ),
                    )


##
# Environment configuration
##


@configclass
class AbsIKLiberoRLEnvCfg(ManagerBasedRLEnvCfg, CameraConfigFactory):
    """All-in-one Abs-IK Libero env with cameras for VLA RL post-training.

    Features:
      - Abs-IK action space (pos3 + quat4 + gripper1)
      - Built-in agentview + eye-in-hand cameras
      - Minimal sensors: ee_frame for state, contact sensors for goal success only
      - No teleop, no gripper force obs, no subtask obs
    """

    scene: InteractiveSceneCfg = MISSING
    actions: ActionsCfg = ActionsCfg()

    use_tiled_camera: bool = True
    camera_height: int = 224
    camera_width: int = 224

    curriculum = None
    commands = None

    def __post_init__(self):
        self.libero_config = LiberoTaskConfig()

        # MDP managers
        self.observations = RLObservationsCfg()
        self.terminations = RLTerminationsCfg()
        self.events = RLEventCfgFrankaPanda()
        self.rewards = RLRewardsCfg()

        # --- Scene ---
        self._setup_scene()
        self._setup_robot()
        self._setup_actions()
        self._setup_objects()
        self._setup_ee_frame()
        self._setup_goal_contact_sensors()
        self._setup_cameras()

        # --- Sim parameters ---
        self.sim.dt = 1 / 60
        self.sim.render_interval = 3
        self.decimation = 3
        # Episode horizon is defined by the LIBERO task suite (short fixed-horizon tasks),
        # so the sim owns per-step ``time_out`` auto-reset. One rollout pass that runs the
        # full horizon (e.g. 160 steps @ 8 s for libero_spatial) covers exactly one episode.
        self.episode_length_s = episode_length_s_for_suite(self.libero_config.task_suite)

        # Isaac Lab 3.0 moved PhysX params from SimulationCfg.physx to SimulationCfg.physics
        # (a PhysxCfg, default None). Fall back to .physx on Isaac Lab 2.x.
        _physx_params = dict(
            bounce_threshold_velocity=0.01,
            gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity=16 * 1024,
            friction_correlation_distance=0.00625,
        )
        if hasattr(self.sim, "physx"):
            for _k, _v in _physx_params.items():
                setattr(self.sim.physx, _k, _v)
        else:
            from isaaclab_physx.physics import PhysxCfg

            self.sim.physics = PhysxCfg(**_physx_params)

        self.viewer.eye = [1.0, 0.0, 2.0]
        self.viewer.lookat = [0.0, 0.0, 1.0]
        self.rerender_on_reset = True

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_scene(self):
        workspace = self.libero_config.workspace_name
        scene_map = {
            "living_room_table": LivingRoomTableSceneCfg,
            "floor": FloorSceneCfg,
            "study_table": StudyTableSceneCfg,
            "kitchen_table": KitchenTableSceneCfg,
            "table": KitchenTableSceneCfg,
        }
        scene_cls = scene_map.get(workspace)
        if scene_cls is None:
            raise ValueError(f"Unsupported workspace: {workspace}")
        self.scene = scene_cls(num_envs=4096, env_spacing=2.0, replicate_physics=False)

    def _setup_robot(self):
        self.scene.robot = FRANKA_PANDA_LIBERO_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = self.libero_config.robot_base_pos
        # NOTE(quat-order change from Lab 2.x to Lab 3.0): (w, x, y, z) to (x, y, z, w)
        self.scene.robot.init_state.rot = wxyz_to_xyzw(self.libero_config.robot_base_ori)
        self.gripper_joint_names = ["panda_finger.*"]
        self.gripper_open_val = 0.04
        self.gripper_threshold = 0.01

    def _setup_actions(self):
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=["panda_joint.*"],
            body_name="panda_hand",
            controller=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            scale=1.0,
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.1034]),
        )
        self.actions.gripper_action = mdp.AbsBinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["panda_finger.*"],
            threshold=0.5,
            open_command_expr={"panda_finger_.*": 0.04},
            close_command_expr={"panda_finger_.*": 0.0},
        )

    def _joint_pos_from_region(self, obj_name: str, obj_type: str) -> dict[str, float]:
        """Read ``joint_pos_range`` from the region JSON and return midpoint values
        keyed by joint name.  Falls back to ``ARTICULATION_DEFAULT_JOINTS``."""
        default_joints = ARTICULATION_DEFAULT_JOINTS.get(obj_type, {})
        region_name = self.libero_config.objects[obj_name]["initial_region"]
        pose_range = self.libero_config.regions.get(region_name, {}).get("pose_range", {})
        jpr = pose_range.get("joint_pos_range", {})
        result: dict[str, float] = {}
        for jid, jname in enumerate(default_joints.keys()):
            r = jpr.get(str(jid))
            if r is not None:
                result[jname] = (r[0] + r[1]) / 2.0
            else:
                result[jname] = default_joints[jname]
        return result

    def _setup_objects(self):
        """Add task objects to the scene (same objects as base config, no extra sensors)."""
        for obj_name, obj_info in self.libero_config.objects.items():
            obj_type = obj_info["type"]
            if obj_type == "flat_stove":
                jp = self._joint_pos_from_region(obj_name, obj_type)
                self.scene.flat_stove_1 = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/flat_stove_1",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=f"{self.libero_config.assets_dir}/{obj_type}/{obj_type}.usd",
                        activate_contact_sensors=True,
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.8, 0.0, 0.0),
                        rot=(0.0, 0.0, 0.0, 1.0),
                        joint_pos=jp,
                    ),
                    actuators={
                        "knob_actuator": ImplicitActuatorCfg(
                            joint_names_expr=["button"],
                            effort_limit_sim=10.0,
                            velocity_limit_sim=2.0,
                            stiffness=0.0,
                            damping=0.0,
                            friction=0.5,
                        ),
                    },
                )
            elif obj_type == "microwave":
                jp = self._joint_pos_from_region(obj_name, obj_type)
                self.scene.microwave_1 = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/microwave_1",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=f"{self.libero_config.assets_dir}/{obj_type}/{obj_type}.usd",
                        activate_contact_sensors=True,
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.36, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        joint_pos=jp,
                    ),
                    actuators={
                        "door_actuator": ImplicitActuatorCfg(
                            joint_names_expr=["microjoint"],
                            effort_limit_sim=0.01,
                            velocity_limit_sim=20.0,
                            stiffness=0.0,
                            damping=0.0,
                            friction=0.2,
                        ),
                    },
                )
            elif obj_type == "white_cabinet":
                jp = self._joint_pos_from_region(obj_name, obj_type)
                self.scene.white_cabinet_1 = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/white_cabinet_1",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=f"{self.libero_config.assets_dir}/{obj_type}/{obj_type}.usd",
                        activate_contact_sensors=True,
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.36, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        joint_pos=jp,
                    ),
                    actuators={
                        "drawer_bottom_actuator": ImplicitActuatorCfg(
                            joint_names_expr=["bottom_level"],
                            effort_limit_sim=0.0,
                            velocity_limit_sim=20.0,
                            stiffness=0.0,
                            damping=0.0,
                            friction=0.0,
                        ),
                        "drawer_top_middle_actuator": ImplicitActuatorCfg(
                            joint_names_expr=["top_level", "middle_level"],
                            effort_limit_sim=10.0,
                            velocity_limit_sim=20.0,
                            stiffness=1000.0,
                            damping=0.0,
                        ),
                    },
                )
            elif obj_type == "wooden_cabinet":
                jp = self._joint_pos_from_region(obj_name, obj_type)
                self.scene.wooden_cabinet_1 = ArticulationCfg(
                    prim_path="{ENV_REGEX_NS}/wooden_cabinet_1",
                    spawn=sim_utils.UsdFileCfg(
                        usd_path=f"{self.libero_config.assets_dir}/{obj_type}/{obj_type}.usd",
                        activate_contact_sensors=True,
                    ),
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.36, 0.0),
                        rot=(1.0, 0.0, 0.0, 0.0),
                        joint_pos=jp,
                    ),
                    actuators={
                        "drawer_actuator": ImplicitActuatorCfg(
                            joint_names_expr=["top_level", "middle_level", "bottom_level"],
                            effort_limit_sim=10.0,
                            velocity_limit_sim=2.0,
                            stiffness=0.0,
                            damping=0.0,
                            friction=0.01,
                        ),
                    },
                )
            else:
                setattr(
                    self.scene,
                    obj_name,
                    RigidObjectCfg(
                        prim_path="{ENV_REGEX_NS}" + f"/{obj_name}",
                        init_state=RigidObjectCfg.InitialStateCfg(),
                        spawn=UsdFileCfg(
                            usd_path=f"{self.libero_config.assets_dir}/{obj_type}/{obj_type}.usd",
                            activate_contact_sensors=(obj_name in self.libero_config.targets),
                            scale=obj_info["scale"],
                            rigid_props=_OBJECT_RIGID_PROPS,
                        ),
                    ),
                )

    def _setup_ee_frame(self):
        """EE FrameTransformer — the only one needed (for eef_pose observation)."""
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/panda_link0",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/panda_hand",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.1034]),
                ),
            ],
        )

    def _setup_goal_contact_sensors(self):
        """Only the contact sensors required by TerminationsCfg.success (libero_goals_reached)."""
        for item in self.libero_config.goals:
            if "relationship" not in item:
                continue
            target_name = item["target"]
            obj_name = item["ref_obj"]
            target_prim = _RIGID_BODY_PRIM_NAMES.get(target_name, target_name)
            setattr(
                self.scene,
                f"contact_{target_name}_{obj_name}",
                ContactSensorCfg(
                    prim_path="{ENV_REGEX_NS}/" + target_prim,
                    update_period=0.0,
                    history_length=6,
                    debug_vis=False,
                    filter_prim_paths_expr=["{ENV_REGEX_NS}/" + f"{obj_name}"],
                ),
            )

    def _setup_cameras(self):
        """Built-in eye-in-hand + agentview cameras."""
        self.scene.eye_in_hand_cam = self._create_camera(
            prim_path="{ENV_REGEX_NS}/Robot/panda_hand/eye_in_hand_camera",
            pos=(0.05, 0.0, 0.0),
            rot=(0.0, 0.707108, 0.707108, 0.0),
            focal_length=9.77,
            focus_distance=400.0,
            horizontal_aperture=15.0,
            clipping_range=(0.001, 1.0),
            convention="opengl",
        )

        workspace = self.libero_config.workspace_name
        params = _AGENTVIEW_PARAMS.get(workspace)
        if params is None:
            raise ValueError(f"Workspace '{workspace}' has no agentview camera parameters.")
        self.scene.agentview_cam = self._create_camera(
            prim_path="{ENV_REGEX_NS}/agentview_camera",
            focal_length=18.0,
            focus_distance=400.0,
            horizontal_aperture=15.0,
            clipping_range=(0.1, 1.9),
            convention="opengl",
            **params,
        )
