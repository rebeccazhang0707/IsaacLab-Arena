# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Replay demonstrations and re-record them with the current (updated) camera configuration.

This is a variant of ``replay_demos.py`` tailored for regenerating a dataset after changing the
robot's camera setup (e.g. head-camera pose / focal length in the environment config). It replays the
recorded *actions* of each source demo, re-simulates the trajectory, and re-records the episode --
including the camera observations rendered from the new camera view -- into a new HDF5 dataset.

Because action replay re-simulates the scene, some demos may drift and fail to complete the task. The
script evaluates the environment's task-success termination at every step and writes **only the demos
that succeed on replay** to the output dataset (failed demos are dropped). Everything else (actions,
states, processed actions, non-camera observations) is reproduced from the source trajectory.

See ``README_replay_demos_w_new_cam.md`` for usage and details.
"""

"""Launch Isaac Sim Simulator first."""


from isaaclab.app import AppLauncher

from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
from isaaclab_arena_environments.cli import add_example_environments_cli_args, get_arena_builder_from_cli

# add argparse arguments
parser = get_isaaclab_arena_cli_parser()
parser.add_argument(
    "--select_episodes",
    type=int,
    nargs="+",
    default=[],
    help="A list of episode indices to be replayed. Keep empty to replay all in the dataset file.",
)
parser.add_argument("--dataset_file", type=str, default="datasets/dataset.hdf5", help="Dataset file to be replayed.")
parser.add_argument(
    "--output_file",
    type=str,
    default=None,
    help=(
        "If set, replay the recorded actions and re-record the resulting episodes (actions, states, obs,"
        " processed_actions, and camera observations with the current camera configuration) into this new"
        " hdf5 file. The original demo indices and success flags are preserved."
    ),
)
parser.add_argument(
    "--decimation",
    type=int,
    default=None,
    help=(
        "Override the environment control decimation (physics steps per control step). Increasing it gives"
        " the low-level controller more sub-steps to converge to each recorded action target, which keeps the"
        " replayed trajectory closer to the original and reduces accumulated tracking error."
    ),
)
parser.add_argument(
    "--save_video",
    action="store_true",
    default=False,
    help="Save a visualization mp4 (robot point-of-view camera) for every replayed episode. Requires --enable_cameras.",
)
parser.add_argument(
    "--num_success_steps",
    type=int,
    default=10,
    help=(
        "Number of consecutive steps the task-success condition must hold during replay for an episode to be"
        " considered a successful replay. Only successful replays are written to --output_file."
    ),
)
parser.add_argument(
    "--keep_failed_videos",
    action="store_true",
    default=False,
    help="Also save videos of failed replays (to a 'failed/' subdirectory) for inspection. Off by default.",
)
parser.add_argument(
    "--video_dir",
    type=str,
    default=None,
    help=(
        "Directory to write per-episode replay videos to. Defaults to '<output_file_dir>/replay_videos' when"
        " --output_file is set, otherwise './replay_videos'."
    ),
)
parser.add_argument(
    "--validate_states",
    action="store_true",
    default=False,
    help=(
        "Validate if the states, if available, match between loaded from datasets and replayed. Only valid if"
        " --num_envs is 1."
    ),
)
# Add the example environments CLI args
# NOTE(alexmillane, 2025.09.04): This has to be added last, because
# of the app specific flags being parsed after the global flags.
add_example_environments_cli_args(parser)

# parse the arguments
args_cli = parser.parse_args()
# args_cli.headless = True

# launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import contextlib
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import os
import torch

import isaaclab_tasks  # noqa: F401
import isaaclab_tasks.manager_based.manipulation.pick_place  # noqa: F401
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers import DatasetExportMode, RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler

is_paused = False


class PreStepFlatCameraObservationsRecorder(RecorderTerm):
    """Recorder term that records the camera observations in each step."""

    def record_pre_step(self):
        return "camera_obs", self._env.obs_buf["camera_obs"]


@configclass
class PreStepFlatCameraObservationsRecorderCfg(RecorderTermCfg):
    """Configuration for the camera observation recorder term."""

    class_type: type[RecorderTerm] = PreStepFlatCameraObservationsRecorder


@configclass
class ArenaEnvRecorderManagerCfg(ActionStateRecorderManagerCfg):
    """Action/state recorder manager that also records flat camera observations."""

    record_pre_step_flat_camera_observations = PreStepFlatCameraObservationsRecorderCfg()


def play_cb():
    global is_paused
    is_paused = False


def pause_cb():
    global is_paused
    is_paused = True


def compare_states(state_from_dataset, runtime_state, runtime_env_index) -> (bool, str):
    """Compare states from dataset and runtime.

    Args:
        state_from_dataset: State from dataset.
        runtime_state: State from runtime.
        runtime_env_index: Index of the environment in the runtime states to be compared.

    Returns:
        bool: True if states match, False otherwise.
        str: Log message if states don't match.
    """
    states_matched = True
    output_log = ""
    for asset_type in ["articulation", "rigid_object"]:
        for asset_name in runtime_state[asset_type].keys():
            for state_name in runtime_state[asset_type][asset_name].keys():
                runtime_asset_state = runtime_state[asset_type][asset_name][state_name][runtime_env_index]
                dataset_asset_state = state_from_dataset[asset_type][asset_name][state_name]
                if len(dataset_asset_state) != len(runtime_asset_state):
                    raise ValueError(f"State shape of {state_name} for asset {asset_name} don't match")
                for i in range(len(dataset_asset_state)):
                    if abs(dataset_asset_state[i] - runtime_asset_state[i]) > 0.01:
                        states_matched = False
                        output_log += f'\tState ["{asset_type}"]["{asset_name}"]["{state_name}"][{i}] don\'t match\r\n'
                        output_log += f"\t  Dataset:\t{dataset_asset_state[i]}\r\n"
                        output_log += f"\t  Runtime: \t{runtime_asset_state[i]}\r\n"
    return states_matched, output_log


def main():  # noqa: C901
    """Replay episodes loaded from a file."""
    global is_paused

    # Load dataset
    if not os.path.exists(args_cli.dataset_file):
        raise FileNotFoundError(f"The dataset file {args_cli.dataset_file} does not exist.")
    dataset_file_handler = HDF5DatasetFileHandler()

    dataset_file_handler.open(args_cli.dataset_file)
    env_name = dataset_file_handler.get_env_name()
    episode_count = dataset_file_handler.get_num_episodes()

    if episode_count == 0:
        print("No episodes found in the dataset.")
        exit()

    episode_indices_to_replay = args_cli.select_episodes
    if len(episode_indices_to_replay) == 0:
        episode_indices_to_replay = list(range(episode_count))

    num_envs = args_cli.num_envs

    # Compile an IsaacLab compatible arena environment configuration
    arena_builder = get_arena_builder_from_cli(args_cli)
    env_name, env_cfg = arena_builder.build_registered()

    # Optionally increase control decimation so the controller converges better to each recorded
    # action target during replay (reduces accumulated tracking error vs. the original trajectory).
    if args_cli.decimation is not None:
        print(f"Overriding control decimation: {env_cfg.decimation} -> {args_cli.decimation}")
        env_cfg.decimation = args_cli.decimation

    # Extract the task-success termination so we can evaluate it manually during replay and only keep
    # episodes whose replay actually succeeds (action replay re-simulates, so some demos may fail).
    success_term = None
    if hasattr(env_cfg.terminations, "success"):
        success_term = env_cfg.terminations.success

    # Optionally re-record the replayed episodes (with the current camera view) into a new dataset.
    recording_enabled = args_cli.output_file is not None
    if recording_enabled:
        if success_term is None:
            print(
                "Warning: no 'success' termination term found; cannot filter by replay success. "
                "All replayed episodes will be saved."
            )
        output_dir = os.path.dirname(os.path.abspath(args_cli.output_file))
        output_file_name = os.path.splitext(os.path.basename(args_cli.output_file))[0]
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        # Use the camera-aware recorder when cameras are enabled, matching dataset generation.
        if args_cli.enable_cameras:
            env_cfg.recorders = ArenaEnvRecorderManagerCfg()
        else:
            env_cfg.recorders = ActionStateRecorderManagerCfg()
        env_cfg.recorders.dataset_export_dir_path = output_dir
        env_cfg.recorders.dataset_filename = output_file_name
        # Only successful replays are written to the output dataset.
        env_cfg.recorders.dataset_export_mode = DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        # Export episodes manually at episode boundaries (see main loop) so we can preserve the
        # original demo indexing and set the success flag from our own evaluation.
        env_cfg.recorders.export_in_record_pre_reset = False
    else:
        # Disable all recorders (visualization-only replay)
        env_cfg.recorders = {}

    # Set up per-episode visualization video saving.
    video_enabled = args_cli.save_video
    if video_enabled:
        if not args_cli.enable_cameras:
            raise ValueError("--save_video requires --enable_cameras so the point-of-view camera can be captured.")
        if args_cli.video_dir is not None:
            video_dir = args_cli.video_dir
        elif args_cli.output_file is not None:
            video_dir = os.path.join(os.path.dirname(os.path.abspath(args_cli.output_file)), "replay_videos")
        else:
            video_dir = os.path.abspath("replay_videos")
        os.makedirs(video_dir, exist_ok=True)
        failed_video_dir = os.path.join(video_dir, "failed")
        if args_cli.keep_failed_videos:
            os.makedirs(failed_video_dir, exist_ok=True)
        print(f"Saving per-episode replay videos to: {video_dir}")

    # Disable terminations so episodes are driven purely by the recorded actions (success is evaluated
    # manually via success_term so we can filter the recorded dataset to successful replays only).
    env_cfg.terminations = {}

    # create environment from loaded config
    env = gym.make(env_name, cfg=env_cfg)
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import reapply_viewer_cfg

    reapply_viewer_cfg(env)
    env = env.unwrapped

    teleop_interface = Se3Keyboard(Se3KeyboardCfg(pos_sensitivity=0.1, rot_sensitivity=0.1))
    teleop_interface.add_callback("N", play_cb)
    teleop_interface.add_callback("B", pause_cb)
    print('Press "B" to pause and "N" to resume the replayed actions.')

    # Determine if state validation should be conducted
    state_validation_enabled = False
    if args_cli.validate_states and num_envs == 1:
        state_validation_enabled = True
    elif args_cli.validate_states and num_envs > 1:
        print("Warning: State validation is only supported with a single environment. Skipping state validation.")

    # Get idle action (idle actions are applied to envs without next action)
    if hasattr(env_cfg, "idle_action"):
        idle_action = env_cfg.idle_action.repeat(num_envs, 1)
    else:
        idle_action = torch.zeros(env.action_space.shape)

    # reset before starting
    env.reset()
    teleop_interface.reset()

    # When recording, clear the episode buffer captured by the initial env.reset() so the first
    # recorded episode does not include the spurious initial-reset frame.
    if recording_enabled:
        env.recorder_manager.reset()

    # Track the source demo index currently loaded in each env (for export naming and video naming).
    current_demo_ids: dict[int, int | None] = {index: None for index in range(num_envs)}
    # Per-env, per-camera list of frames for the current episode (for video saving). Each env maps a
    # camera observation key (e.g. "robot_pov_cam_rgb", "right_wrist_cam_rgb") to its list of frames,
    # so every camera in the env produces its own video.
    episode_frames: dict[int, dict[str, list]] = {index: {} for index in range(num_envs)}
    # Per-env replay-success tracking: whether the success condition was ever met during the episode,
    # and how many steps ago it last held (an episode counts as a successful replay if the task succeeded
    # within the final ``--num_success_steps`` steps, i.e. the replayed trajectory ends in the goal state).
    _NEVER = 1 << 30
    ever_success: dict[int, bool] = {index: False for index in range(num_envs)}
    steps_since_success: dict[int, int] = {index: _NEVER for index in range(num_envs)}
    # Source demo indices that succeeded / failed on replay (for the end-of-run summary).
    succeeded_demos: list[int] = []
    failed_demos: list[int] = []
    # Frame rate for saved videos derived from the (possibly overridden) control step duration.
    video_fps = round(1.0 / env.step_dt) if getattr(env, "step_dt", 0) else 30

    def capture_camera_frames(env_id: int) -> None:
        """Append the current frame of every camera for ``env_id`` to its per-camera episode buffer.

        Captures all cameras present in ``camera_obs`` (e.g. the head point-of-view camera and the
        right-wrist camera) so each one is written out as its own video.
        """
        camera_obs = env.obs_buf.get("camera_obs") if isinstance(env.obs_buf, dict) else None
        if not camera_obs:
            return
        for key, value in camera_obs.items():
            frame = value[env_id, ..., :3]
            episode_frames[env_id].setdefault(key, []).append(frame.detach().cpu().numpy().astype(np.uint8))

    def evaluate_success(env_id: int) -> None:
        """Evaluate the task-success term for ``env_id`` and update its replay-success trackers.

        The success term is stateful (it advances a per-env subtask state machine), so it must be
        called exactly once per step; the per-episode state is reset by ``reset_to`` via the
        ``reset_subtask_success_state`` event.
        """
        if success_term is None:
            # No success term available: treat every replayed episode as successful.
            ever_success[env_id] = True
            steps_since_success[env_id] = 0
            return
        is_success = bool(success_term.func(env, **success_term.params)[env_id])
        if is_success:
            ever_success[env_id] = True
            steps_since_success[env_id] = 0
        else:
            steps_since_success[env_id] += 1

    def finalize_episode(env_id: int) -> None:
        """Export the successful recorded episode and/or write its video, then reset trackers."""
        demo_id = current_demo_ids[env_id]
        if demo_id is None:
            return
        # A replay counts as successful if the task reached success within the final num_success_steps steps.
        succeeded = ever_success[env_id] and steps_since_success[env_id] < args_cli.num_success_steps
        (succeeded_demos if succeeded else failed_demos).append(demo_id)
        status = "SUCCESS" if succeeded else "FAILED"
        since = "never" if not ever_success[env_id] else f"{steps_since_success[env_id]} steps ago"
        print(f"      demo_{demo_id} replay {status} (task success last held: {since})")
        if recording_enabled:
            # Set the real success flag; EXPORT_SUCCEEDED_ONLY drops failed episodes automatically.
            env.recorder_manager.set_success_to_episodes(
                [env_id], torch.tensor([[succeeded]], dtype=torch.bool, device=env.device)
            )
            env.recorder_manager.export_episodes([env_id], demo_ids=[demo_id])
        if video_enabled and episode_frames[env_id]:
            write_dir = video_dir if succeeded else (failed_video_dir if args_cli.keep_failed_videos else None)
            if write_dir is not None:
                status_label = "video" if succeeded else "FAILED video"
                # Write one video per camera. The head point-of-view camera keeps the bare
                # "demo_<id>.mp4" name; any additional camera (e.g. the right-wrist camera) is
                # suffixed with its name so the views are easy to tell apart.
                for cam_key, frames in episode_frames[env_id].items():
                    if not frames:
                        continue
                    cam_name = cam_key[: -len("_rgb")] if cam_key.endswith("_rgb") else cam_key
                    suffix = "" if cam_key == "robot_pov_cam_rgb" else f"_{cam_name}"
                    video_path = os.path.join(write_dir, f"demo_{demo_id}{suffix}.mp4")
                    imageio.mimwrite(video_path, frames, fps=video_fps, macro_block_size=None)
                    print(f"      saved {status_label}: {video_path} ({len(frames)} frames)")
        episode_frames[env_id] = {}
        ever_success[env_id] = False
        steps_since_success[env_id] = _NEVER
        current_demo_ids[env_id] = None

    # simulate environment -- run everything in inference mode
    episode_names = list(dataset_file_handler.get_episode_names())
    replayed_episode_count = 0
    with contextlib.suppress(KeyboardInterrupt) and torch.inference_mode():
        while simulation_app.is_running() and not simulation_app.is_exiting():
            env_episode_data_map = {index: EpisodeData() for index in range(num_envs)}
            first_loop = True
            has_next_action = True
            while has_next_action:
                # initialize actions with idle action so those without next action will not move
                actions = idle_action
                has_next_action = False
                for env_id in range(num_envs):
                    env_next_action = env_episode_data_map[env_id].get_next_action()
                    if env_next_action is None:
                        next_episode_index = None
                        while episode_indices_to_replay:
                            next_episode_index = episode_indices_to_replay.pop(0)
                            if next_episode_index < episode_count:
                                break
                            next_episode_index = None

                        if next_episode_index is not None:
                            # Flush the episode that just finished in this env before loading a new one.
                            finalize_episode(env_id)
                            replayed_episode_count += 1
                            print(f"{replayed_episode_count :4}: Loading #{next_episode_index} episode to env_{env_id}")
                            episode_name = episode_names[next_episode_index]
                            episode_data = dataset_file_handler.load_episode(episode_name, env.device)
                            env_episode_data_map[env_id] = episode_data
                            # Preserve the original demo index (e.g. "demo_5" -> 5) for export/video naming.
                            current_demo_ids[env_id] = int(episode_name.split("_")[-1])
                            # Set initial state for the new episode
                            initial_state = episode_data.get_initial_state()
                            env.reset_to(initial_state, torch.tensor([env_id], device=env.device), is_relative=True)
                            # Get the first action for the new episode
                            env_next_action = env_episode_data_map[env_id].get_next_action()
                            has_next_action = True
                        else:
                            continue
                    else:
                        has_next_action = True
                    actions[env_id] = env_next_action
                # If no environment has a next action (all episodes exhausted), stop before stepping so
                # we don't record a spurious trailing idle frame into the last recorded episode.
                if not has_next_action:
                    break
                if first_loop:
                    first_loop = False
                else:
                    while is_paused:
                        env.sim.render()
                        continue
                env.step(actions)

                # For each env with an active episode: evaluate task success and capture a video frame.
                for env_id in range(num_envs):
                    if current_demo_ids[env_id] is None:
                        continue
                    evaluate_success(env_id)
                    if video_enabled:
                        capture_camera_frames(env_id)

                if state_validation_enabled:
                    state_from_dataset = env_episode_data_map[0].get_next_state()
                    if state_from_dataset is not None:
                        print(
                            f"Validating states at action-index: {env_episode_data_map[0].next_state_index - 1 :4}",
                            end="",
                        )
                        current_runtime_state = env.scene.get_state(is_relative=True)
                        states_matched, comparison_log = compare_states(state_from_dataset, current_runtime_state, 0)
                        if states_matched:
                            print("\t- matched.")
                        else:
                            print("\t- mismatched.")
                            print(comparison_log)
            break

    # Flush the last in-progress episode(s) that were not followed by another load.
    for env_id in range(num_envs):
        finalize_episode(env_id)

    # Summary of replay-success filtering.
    num_success = len(succeeded_demos)
    num_failed = len(failed_demos)
    print(
        f"\nReplay success summary: {num_success}/{replayed_episode_count} succeeded, {num_failed} failed."
    )
    if failed_demos:
        print(f"  Failed source demos: {sorted(failed_demos)}")
    if recording_enabled:
        exported_path = os.path.join(
            env.recorder_manager.cfg.dataset_export_dir_path,
            f"{env.recorder_manager.cfg.dataset_filename}.hdf5",
        )
        print(f"Saved {num_success} successful episode(s) with current camera view to: {exported_path}")

    # Close environment after replay in complete
    plural_trailing_s = "s" if replayed_episode_count > 1 else ""
    print(f"Finished replaying {replayed_episode_count} episode{plural_trailing_s}.")
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
