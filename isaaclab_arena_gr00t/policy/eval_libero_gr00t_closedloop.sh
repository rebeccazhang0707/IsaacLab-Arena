#!/usr/bin/env bash
# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0
#
# Closed-loop GR00T eval of "LIBERO spatial task 3" in Isaac Lab Arena, via the Arena
# policy_runner (mirrors the GR1 ranch-bottle policy_runner command, but for the `libero`
# external environment + the Franka eef-pose closed-loop policy).
#
# It runs the standard Arena policy_runner with:
#   * policy : Gr00tLiberoClosedloopPolicy (eef-pose I/O; rel_rotvec checkpoint)
#   * env    : isaaclab_arena_examples.external_environments.libero:LiberoEnvironment
#              (libero --task_suite libero_spatial --task_id 3)
#
# Env-overridable knobs (defaults below):
set -euo pipefail
set -x

# --- Locate the IsaacLab-Arena repo root (this script lives under
#     <root>/isaaclab_arena_examples/external_environments/libero/) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ARENA_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

TASK_SUITE=${TASK_SUITE:-libero_spatial}
TASK_ID=${TASK_ID:-3}
NUM_STEPS=${NUM_STEPS:-512}
NUM_ENVS=${NUM_ENVS:-1}
# Visualizer: "none" (headless, default) for batch eval; set VIZ=kit for the live Kit viewer.
VIZ=${VIZ:-none}

# Policy config YAML (eef-pose closed-loop GR00T for LIBERO Franka).
POLICY_CONFIG=${POLICY_CONFIG:-isaaclab_arena_gr00t/policy/config/franka_libero_gr00t_closedloop_config.yaml}
# Optional: override the checkpoint in the YAML without editing it (e.g. the container path
# /models/checkpoint-20000). When set, we render a temp config with model_path swapped in.
MODEL_PATH=${MODEL_PATH:-}

CLASS_PATH=${CLASS_PATH:-isaaclab_arena_examples.external_environments.libero:LiberoEnvironment}
POLICY_TYPE=${POLICY_TYPE:-isaaclab_arena_gr00t.policy.gr00t_libero_closedloop_policy.Gr00tLiberoClosedloopPolicy}

# Python launcher: inside the Arena GR00T container python == /isaac-sim/python.sh; on the host
# uv checkout use `ISAAC_PYTHON="uv run python"`. Falls back to plain `python` if the default
# /isaac-sim/python.sh is not present (i.e. not in the container).
ISAAC_PYTHON=${ISAAC_PYTHON:-/isaac-sim/python.sh}
if ! command -v "${ISAAC_PYTHON%% *}" >/dev/null 2>&1; then
    ISAAC_PYTHON=python
fi

# GR00T/Eagle needs transformers 4.51.3 (in /opt/groot_deps) PREPENDED inside the container;
# harmless on the host (dir simply does not exist). Arena root must be importable so that the
# `isaaclab_arena_examples` package resolves.
export PYTHONPATH="/opt/groot_deps:${ARENA_ROOT}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# The LIBERO env auto-detects /libero_in_lab; set it explicitly to be safe.
export LIBERO_IN_LAB_ROOT=${LIBERO_IN_LAB_ROOT:-/libero_in_lab}

cd "${ARENA_ROOT}"

# Optionally swap the checkpoint path into a temp config.
RUN_CONFIG="${POLICY_CONFIG}"
if [[ -n "${MODEL_PATH}" ]]; then
    RUN_CONFIG="$(mktemp --suffix=.yaml)"
    sed "s#^model_path:.*#model_path: ${MODEL_PATH}#" "${POLICY_CONFIG}" > "${RUN_CONFIG}"
    trap 'rm -f "${RUN_CONFIG}"' EXIT
fi

"${ISAAC_PYTHON}" isaaclab_arena/evaluation/policy_runner.py \
    --viz "${VIZ}" \
    --policy_type "${POLICY_TYPE}" \
    --policy_config_yaml_path "${RUN_CONFIG}" \
    --num_steps "${NUM_STEPS}" \
    --num_envs "${NUM_ENVS}" \
    --enable_cameras \
    --external_environment_class_path "${CLASS_PATH}" \
    libero \
    --task_suite "${TASK_SUITE}" \
    --task_id "${TASK_ID}" \
    "$@"
