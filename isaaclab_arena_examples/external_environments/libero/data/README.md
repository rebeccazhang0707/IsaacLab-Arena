# LIBERO To Isaac Lab Best Practice
LIBERO is a popular lifelong learning benchmarks, providing simulation envs/datasets to pretrain/evaluate varied visuomotor policy algorithms.
We provide tools to bring LIBERO tasks/envs to Lab.

🔥 Checkout our **Libero2Lab Environments** [here](../../../guide/available_envs.md#libero2lab-environments). 🔥


## Libero HDF5 Trajectory Dataset
### 1. Overview:
First, follow LIBERO official [README](https://github.com/Lifelong-Robot-Learning/LIBERO) to install its conda environment called "libero".

Libero robot motion trajectory HDF5 structure as below:
```
data/
└── demo_0/
    ├── actions [N, 7]          # delta eef pos, delta eef rpy for OSC_POSE controller in robosuite, gripper binary action
    ├── dones [N]
    ├── obs/
    │   ├── agentview_rgb [N, 128, 128, 3]
    │   ├── eye_in_hand_rgb [N, 128, 128, 3]
    │   ├── ee_ori [N, 3]
    │   ├── ee_pos [N, 3]
    │   ├── ee_states [N, 6]    # ee_pos, ee_ori
    │   ├── gripper_states [N, 2]
    │   └── joint_states [N, 7]
    ├── rewards [N]
    ├── robot_states [N, 9]     # gripper_qpos, eef_pos, eef_quat
    └── states [N, 1 + nq + nv]     # timestamp, num_q, num_vel
```

### 2. Add Missing Information:
HDF5 has not recorded initial states of the scenen, so we need to get Initial States of All Rigid Bodies and Robots if we would like to replay demos.
- Use below script to check the "DOF" of all robots and rigid_bodies, preparing for parsing the flattened vector in "states".

```bash
# active libero env first
conda activate libero
python benchmarks/datasets/libero/utils/get_body_dof_info.py --task_suite libero_10 --task_id 0

```

## Libero Task Envs Import To Lab
### 1. Prepare Datasets
Download all the **SimReady USD assets** and **hdf5 demo files** from [HuggingFace](https://huggingface.co/datasets/china-sae-robotics/IsaacLabPlayGround_Dataset).

```bash
git clone https://huggingface.co/datasets/china-sae-robotics/IsaacLabPlayGround_Dataset $DATASET
# copy all the data (USD assets, hdf5 demos, etc.) folder to your workspace
cp -r $DATASET/libero/* $WORKSPACE/benchmarks/datasets/libero/
```

### 2. Generate configs for each task suite:
**Libero** conda requirements may have conflict with **RobotLearningLab**.
So we don't online parse the configs, but in 'libero' env, offline generate the json config files for each task suite.
- task basic info
- all rigid_object lists
- all initial_regions of both rigid_objects (like cup, etc.) and colliders/fixtures (like table, etc.)
- obj_of_interest: objects to grasp
- target_obj
- goals (success terminations: including "relationship" and "operation" two classes. )
- workspace name, size, offset, etc.

```bash
conda activate libero
python benchmarks/datasets/libero/utils/generate_configs.py --task_suite libero_10
```

### 3. Record demos with teleoperation:
**Notes:** the init_states of all the objects in the scene are sampled from the distribution of libero expert datasets.
If you would like to increase the randomization range of the objects, you need to modify the "regions" properties of "libero_*.json" files in the "config" folders.
```bash
# use isaaclab conda env
conda activate env_isaaclab

# for Pick and Place tasks
python scripts/playground/record_demos.py \
    --dataset_file record_demos/libero_10_task0_demo10.hdf5 \
    --task Isaac-Libero-Franka-IK-Abs-v0 \
    --task_suite libero_10 \
    --task_id 0 \
    --teleop_device keyboard \
    --num_demos 10 \
    --enable_cameras \
    --sensitivity 0.2

# for articulated tasks
python scripts/playground/record_demos.py \
    --dataset_file record_demos/libero_10_task9_demo10.hdf5 \
    --task Isaac-Libero-Franka-IK-Abs-v0 \
    --task_suite libero_10 \
    --task_id 9 \
    --teleop_device keyboard \
    --num_demos 10 \
    --enable_cameras \
    --sensitivity 0.2
```



## How to Replay Trajectory in Isaac Lab
We have done the data generation process already, and released the dataset for your out-of-the-box usage.
You can directly download the datasets from [HuggingFace](https://huggingface.co/datasets/china-sae-robotics/IsaacLabPlayGround_Dataset/tree/main/libero).

Below shows the process of how these datasets are generated in **RobotLearningLab**.

### 1. Generate hdf5 for replay in Lab:

Generate assembled_hdf5 file of each task (50 demos) to replay in Lab:
- Save "actions" in Joint Position space
- Add "initial_state" group
```bash
conda activate libero
python benchmarks/datasets/libero/utils/generate_hdf5_for_lab_replay.py --task_suite libero_10 --task_id 0
```
### 2. Replay the hdf5 demo
- Use JointPosition Controller
- Use HIGH_PD ActuatorCfg
- Record demos with AbsEEFPoseAction to fit for Mimic workflow

For example, "libero_10" tasks:
```bash
# use isaaclab conda env
conda activate env_isaaclab
# input and output file path is auto configured within our code, you can specify your own path by given --dataset_file xxx.hdf5 --outptu_file xxx.hdf5
python scripts/playground/replay_demos.py \
    --task Isaac-Libero-Franka-Replay-v0 \
    --task_suite libero_10 \
    --task_id 0 \
    --num_envs 1 \
    --dump_data
```

### 3. Replay the hdf5 demo and record videos in the meanwhile
- Record videos (by default to 'generated_videos' folder), given a list of "camera_views"

```bash
# input and output file path is auto configured within our code, you can specify your own path by given --dataset_file xxx.hdf5 --outptu_file xxx.hdf5
python scripts/playground/replay_demos_with_camera.py \
    --task Isaac-Libero-Franka-Replay-Camera-v0 \
    --task_suite libero_10 \
    --task_id 0 \
    --num_envs 1 \
    --dump_data \
    --video \
    --camera_view_list eye_in_hand agentview
```
### 4. Pitfalls for Sim2Lab Replay
- BinaryGripperAction is better than AbsGripperAction for carefully replay demos with high accuracy requirement. You don't need to tune a specific threshold for each task.
- Use high gains (stiffer PD) for the actuator.
- Carefully check the initial_states of both robots and articulations, e.g. joint_pos, joint_vel, etc. And make sure to parse the values that are actually working. e.g., use the first state from 'states' instead of 'initial_states' of env.
- Use Joint-space robot controller for replay: original OSC_POSE relative task-space controller in robosuite can result in error accumulation during trajectory replay.
- Add appropriate "Mass" property to the rigid_object USD prim to grasp:
   - (e.g., mass = 0.1 kg) Thus, enabling robust grasp w/o sliding.
- Give appropriate "friction" property for "ImplicitActuatorCfg" for the passive joints in articulated objects:
   - like "button" joint in "flat_stove": (e.g. friction = 0.05) so that the joint can be easily rotated by external force, but still with small resistance.
- Use some simpler geometry (Cube/Cylinder) to replace the default ConvexHull/ConvexDecomposition collider, can increase the grasping stability.


### 5. Trajectory Replay Success Rate Analysis
Most modern VLA algorithms do experiments based on OpenVLA's [Libero Dataset](https://huggingface.co/datasets/openvla/modified_libero_rlds). So we compare the distribution of our replayed dataset with OpenVLA's Dataset.

#### Task Suite Performance Comparison (each task out of 50 demos)

| Task Suite    | Tasks | Avg Success Rate (Isaac Lab) | OpenVLA Dataset Success Rate |
|---------------|-------|------------------------|------------------------------|
| libero_goal   | 10    | 76.4%                  | 85.60%                       |
| libero_object | 10    | 79.0%                  | 90.80%                       |
| libero_10     | 10    | 64.8%                  | 75.80%                       |
| libero_spatial| 10    | 75.2%                  | 86.40%                       |
| **OVERALL AVG** | **40** | **73.80%**          | **80.25%**                   |

## Generate LeRobot Dataset for GR00T Nx Finetuning
### 1. Generate lerobot dataset (in task space):
```bash
## generate separate dataset for each task suite, e.g. libero_10
python benchmarks/datasets/libero/convert_single_libero_to_lerobot_task_space.py  --task_suite libero_10

## generate a whole dataset for all 4 task suites
python benchmarks/datasets/libero/convert_all_libero_to_lerobot_task_space.py
```
You will see logs like below for a successful dataset conversion.
```bash
Conversion completed:
Total episodes processed: 1421
Total frames: 235369
Total videos: 2842
Total tasks: 41
Output directory: /data/Projects/Robotics/IsaacLab/IsaacLabPlayground/benchmarks/datasets/libero/lerobot_task_space/all_libero_suites

Task Suite Breakdown:
libero_10: 10 tasks (IDs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
libero_spatial: 10 tasks (IDs: [10, 11, 12, 13, 14, 15, 16, 17, 18, 19])
libero_goal: 10 tasks (IDs: [20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
libero_object: 10 tasks (IDs: [30, 31, 32, 33, 34, 35, 36, 37, 38, 39])
```
### 2. Create a new DataConfig for the new embodiment:

You can directly copy our config to your local: ("libero_franka_task_space")
```bash
cp benchmarks/datasets/libero/gr00t/data_config.py $ISAAC_GR00T_PATH/gr00t/experiment/data_config.py
```
### 3. Finetuning: (on all_libero_suites)
```bash
conda activate gr00t && cd $ISAAC_GR00T_PATH
python scripts/gr00t_finetune.py \
   --dataset-path lerobot_task_space/all_libero_suites/ \
   --num-gpus 8 \
   --output-dir ./exps/all_libero_suites_bs160_gpu8  \
   --max-steps 20000 \
   --data-config libero_franka_task_space \
   --video-backend decord \
   --save_steps 5000 \
   --batch_size 160
```
### 4. Close-loop evaluation in Isaac Lab:
- First start the gr00t inference server within gr00t env:
```bash
conda activate gr00t
cd $ISAACLAB_PATH
python benchmarks/gr00t/gr00t_inference_server.py --port 5555 \
  --model_path $ISAAC_GR00T_PATH/exps/all_libero_suites_bs160_gpu8/checkpoint-20000 \
  --data_config libero_franka_task_space
```
- Run inference client in Lab and count the Success Rate out of "num_total_experiments", given task_suite and task_id: this will load all the 50 initial_states from each libero task env to initialize the scene.
```bash
python benchmarks/datasets/libero/gr00t_inference_client.py --server_port 5555 \
  --num_total_experiments 50 --num_success_steps 8 --policy_type task_space \
  --task Isaac-Libero-Franka-IK-Abs-v0 --task_suite libero_10 --task_id 0
```
- Run script to automatic evaluate policy on libero task_suite: this will dump a "success_rates.txt" file in "./evaluation_results" folder.
```bash
# evaluate on a specific task suite
python benchmarks/datasets/libero/run_task_evaluations.py --policy_model gr00t --server_port 5555 --num_total_experiments 50 --task_suites libero_spatial
# evaluate on all libero task suites
python benchmarks/datasets/libero/run_task_evaluations.py --policy_model gr00t --server_port 5555 --num_total_experiments 50
```

- Results:
    - VLA policy trained on all libero task suites dataset.
    - success rate obtained out of 50 demos each task, with seed = 11.

**GR00T-N1.5, 45k Iters, task space (rotation_6d)**

| Task Suite    | Tasks | Replay Success Rate (Isaac Lab) | Evaluation SR (45k) |
|---------------|-------|--------------------------------|---------------|
| libero_goal   | 10    | 76.4%                  | 63.0%         |
| libero_object | 10    | 79.0%                  | 78.2%         |
| libero_10     | 10    | 64.8%                  | 69.0%         |
| libero_spatial| 10    | 75.2%                  | 82.4%         |
| **OVERALL AVG** | **40** | **73.80%**          | **73.15%**    |


## Generate LeRobot Dataset for OpenPI Finetuning and Closed-loop Simulation
### 1.  OpenPI Finetuning on all_libero_suites
**Preparing Steps**
- Clone the **[openpi](https://github.com/Physical-Intelligence/openpi)** repo. (Current commit:**36dc3c**).
- Following  instructions in the [README](https://github.com/Physical-Intelligence/openpi/blob/main/README.md) to
set up openpi finetuning environment. We are using uv to manage dependencies as recommended.
- The following steps - including LeRobot dataset generation, finetuning and server running - should be executed in the OpenPI env.

**Generate LeRobot Dataset (in Task Space) for OpenPI Finetuning**
- Copy `benchmarks/datasets/libero/config` from RobotLearningLib to the `scripts` folder of OpenPI.
- Copy `benchmarks/datasets/libero/convert_all_libero_to_lerobot_openpi.py` from RobotLearningLib to the `scripts` folder of OpenPI.
- Run the following command to generate LeRobot dataset for openpi finetuning. The generated dataset will be located in `lerobot_all_libero_suites` folder in LeRobot Dataset.
```bash
## generate a whole dataset for all 4 task suites
uv run scripts/convert_all_libero_to_lerobot_openpi.py --data_root /home/weihuaz/workspace8T/Libero_isaaclab
```

**OpenPI Finetuning on all_libero_suites**
- To use the generated dataset in training, modify the `src/openpi/training/config.py` file in OpenPI as shown below(including the `name` and `local_files_only` parts).
```python
    TrainConfig(
        name="pi0_libero_low_mem_finetune",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi0.Pi0Config(paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"),
        data=LeRobotLiberoDataConfig(
            repo_id="lerobot_all_libero_suites",
            base_config=DataConfig(
                local_files_only=True,   # For Libero local file
                prompt_from_task=True,
            ),
        )
```
- Before finetuning, compute the normalization statistics for the training data.
```bash
uv run scripts/compute_norm_stats.py --config-name pi0_libero_low_mem_finetune
```

- Start finetuning with:
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi0_libero_low_mem_finetune --exp-name=my_experiment --overwrite
```

### 2. Close-loop evaluation in Isaac Lab:
- First start the openpi inference server in the OpenPI environment:
```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi0_libero_low_mem_finetune --policy.dir=checkpoints/pi0_libero_low_mem_finetune/my_experiment/29999
```

- Run the inference client in RobotLearningLab. This loads 50 initial states from each Libero task environment to initialize the scene
and computes the success rate out of "num_total_experiments", given task_suite and task_id:
```bash
python benchmarks/datasets/libero/openpi_inference_client.py --server_port 8000 \
  --num_total_experiments 50 --num_success_steps 8 --policy_type task_space \
  --task Isaac-Libero-Franka-IK-Abs-v0 --task_suite libero_10 --task_id 4 --max_inference_steps 30 --debug_mode 2
```

- To automatically evaluate the policy, use:
```bash
# evaluate on a specific task suite
python benchmarks/datasets/libero/run_task_evaluations.py --policy_model openpi --server_port 8000 --num_total_experiments 50 --max_inference_steps 80 --task_suites libero_spatial
# evaluate on all libero task suites
python benchmarks/datasets/libero/run_task_evaluations.py --policy_model openpi --server_port 8000 --num_total_experiments 50 --max_inference_steps 80
```
This will dump a "success_rates.txt" file in "./evaluation_results" folder.

- Results:
    - VLA policy trained on all libero task suites dataset.
    - success rate obtained out of 50 demos each task, with seed = 11.

**pi0, 30k Iters, task space (axisangle; rotation_6d may achieve slightly better result)**

| Task Suite    | Tasks | Replay Success Rate (Isaac Lab) | Evaluation SR (30k) |
|---------------|-------|--------------------------------|---------------------|
| libero_goal   | 10    | 76.4%                  | 57.40%              |
| libero_object | 10    | 79.0%                  | 84.40%              |
| libero_10     | 10    | 64.8%                  | 73.40%              |
| libero_spatial| 10    | 75.2%                  | 89.40%              |
| **OVERALL AVG** | **40** | **73.80%**          | **76.15%**          |
