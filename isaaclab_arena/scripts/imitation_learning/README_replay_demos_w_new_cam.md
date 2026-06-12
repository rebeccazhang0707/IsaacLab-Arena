# `replay_demos_w_new_cam.py`

Replay existing demonstrations and **re-record them with the current (updated) camera
configuration**, keeping only the demos whose replay actually succeeds.

This is a variant of the upstream `replay_demos.py`. Use it when you have changed the robot's
camera setup in the environment config (e.g. head-camera pose / focal length) and want to
regenerate an existing dataset so that the recorded camera observations reflect the new view —
without re-collecting demonstrations from scratch.

---

## What it does

For each source demo it:

1. Resets the environment to the demo's initial state.
2. Replays the recorded **actions** step by step (re-simulating the trajectory).
3. Re-records the episode — including **camera observations rendered from the current camera
   config** — plus actions, states, and processed actions.
4. Evaluates the environment's task-success termination every step.
5. Writes the episode to the output dataset **only if the replay succeeded**.

Everything other than the camera view is reproduced from the source trajectory, so the new
dataset is a drop-in replacement with updated camera data.

### Why success filtering is needed

Action replay re-simulates the scene, so small physics/controller differences accumulate and some
demos drift enough to fail the task (e.g. the bottle is not placed, or the door does not close).
Those failed replays would otherwise pollute the dataset, so they are dropped.

A replay counts as **successful** if the task-success condition holds within the final
`--num_success_steps` steps of the episode (i.e. the replayed trajectory ends in the goal state).
The success term for sequential/composite tasks is stateful (it advances a per-env subtask state
machine), so it is evaluated exactly once per step; the per-episode state is reset automatically by
`reset_to`.

### Output naming

Successful demos keep their **original source index** (e.g. source `demo_5` → `demo_5` in the
output). Failed demos leave a gap in the numbering, which makes it easy to trace which source demos
were dropped.

---

## Prerequisites

- Run inside the Docker container (see repo `AGENTS.md`). All commands below assume the container
  shell with `python` aliased to `/isaac-sim/python.sh`.
- The camera change must already be applied in the environment config (this script records whatever
  the env currently produces). For the ranch-bottle task that is the head camera in
  `isaaclab_arena_environments/gr1_put_and_close_door_environment.py`.
- `--enable_cameras` is required to record camera observations.
- On mounted/network filesystems that do not support file locking, set
  `HDF5_USE_FILE_LOCKING=FALSE` to avoid `BlockingIOError` when creating the output file.

---

## Usage

### Full dataset, re-record with new camera (no videos)

```bash
docker exec -e HDF5_USE_FILE_LOCKING=FALSE isaaclab_arena-latest bash -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
   isaaclab_arena/scripts/imitation_learning/replay_demos_w_new_cam.py \
   --headless --enable_cameras \
   --dataset_file  datasets/ranch_bottle_into_fridge_generated_100.hdf5 \
   --output_file   datasets/ranch_bottle_into_fridge_newcam_100.hdf5 \
   put_item_in_fridge_and_close_door"
```

### Quick check on a few episodes, with videos (incl. failed ones)

```bash
docker exec -e HDF5_USE_FILE_LOCKING=FALSE isaaclab_arena-latest bash -c \
  "cd /workspaces/isaaclab_arena && /isaac-sim/python.sh \
   isaaclab_arena/scripts/imitation_learning/replay_demos_w_new_cam.py \
   --select_episodes 0 1 2 3 \
   --headless --enable_cameras --save_video --keep_failed_videos \
   --dataset_file  datasets/ranch_bottle_into_fridge_generated_100.hdf5 \
   --output_file   datasets/replay_filtertest.hdf5 \
   --video_dir     datasets/replay_videos_ft \
   put_item_in_fridge_and_close_door"
```

> The positional task name (e.g. `put_item_in_fridge_and_close_door`) and any embodiment flags are
> the standard Arena environment CLI args — pass the same ones you use for evaluation/generation.

---

## Arguments (added by this script)

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset_file` | `datasets/dataset.hdf5` | Source HDF5 dataset to replay. |
| `--output_file` | `None` | If set, re-record replayed episodes into this new HDF5. If omitted, the script only visualizes/validates (no recording). |
| `--select_episodes` | `[]` (all) | List of episode **positions** in the dataset to replay. Note: this indexes the episode-name list (lexicographic: `demo_0, demo_1, demo_10, ...`), so it is mainly for quick tests; leave empty for full runs. |
| `--num_success_steps` | `10` | A replay counts as successful if task success held within the final N steps. |
| `--save_video` | `False` | Save an mp4 (robot point-of-view camera) per successful replayed episode. Requires `--enable_cameras`. |
| `--keep_failed_videos` | `False` | Also save videos of **failed** replays into a `failed/` subdirectory (for inspection). |
| `--video_dir` | `None` | Where to write videos. Defaults to `<output_dir>/replay_videos`, else `./replay_videos`. |
| `--decimation` | `None` | Override env control decimation (physics sub-steps per control step). Usually leave default; increasing it does **not** reliably reduce drift and can make it worse. |
| `--validate_states` | `False` | Compare replayed states against the dataset (only with `--num_envs 1`). |

Plus the standard Isaac Lab / Arena flags: `--headless`, `--enable_cameras`, `--num_envs`, the
positional task name, embodiment flags, etc.

---

## Output

- **Dataset**: `--output_file` HDF5 containing only successful replays, each with the new camera
  observations. Demo groups keep their original source indices (gaps mark dropped demos).
- **Videos** (if `--save_video`): one mp4 per successful demo in `--video_dir`, named by source
  index; failed demos go to `<video_dir>/failed/` when `--keep_failed_videos` is set.
- **Console summary** at the end:

  ```
  Replay success summary: 70/100 succeeded, 30 failed.
    Failed source demos: [1, 7, 8, ...]
  Saved 70 successful episode(s) with current camera view to: .../newcam_100.hdf5
  ```

> Note: Isaac Sim's kit logger buffers Python stdout, so the per-episode `SUCCESS/FAILED` prints and
> the final summary may not appear in a piped log. Verify the result by inspecting the output HDF5
> (number of `demo_*` groups) rather than the log.


