# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Small quaternion helpers shared across the LIBERO external environment package.

Standalone (only stdlib typing) so both ``camera_factory.py`` and ``franka_libero_rl_env_cfg.py``
can import it without creating a circular import.
"""


def wxyz_to_xyzw(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    """Reorder a (w, x, y, z) quaternion to (x, y, z, w).

    Isaac Lab 3.0 consumes/produces quaternions in (x, y, z, w) order globally (the LIBERO task
    specs and the legacy Lab-2.x sources author them as (w, x, y, z)), so any wxyz literal copied
    from the original ``libero_in_lab`` config must be reordered before it reaches a Lab 3.0 API.
    """
    w, x, y, z = q
    return (x, y, z, w)
