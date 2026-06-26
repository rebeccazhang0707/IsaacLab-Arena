# Copyright (c) 2025-2026, The Isaac Lab Arena Project Developers (https://github.com/isaac-sim/IsaacLab-Arena/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: Apache-2.0

"""Camera configuration factory with TiledCamera support (self-contained Arena port).

Ported from ``isaaclab_playground.utils.camera_factory:CameraConfigFactory``. The
``PerEnvCamera`` / ``PerEnvTiledCamera`` per-env variants (used only by the multi-task
LIBERO layout) are dropped, since the single-task RL env uses plain ``Camera`` /
``TiledCamera``.
"""

import isaaclab.sim as sim_utils
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg

from .quat_utils import wxyz_to_xyzw


class CameraConfigFactory:
    """Factory class providing camera configuration with TiledCamera support."""

    use_tiled_camera: bool = False
    camera_height: int = 512
    camera_width: int = 512

    def _create_camera(
        self,
        prim_path: str,
        pos: tuple,
        rot: tuple,
        focal_length: float = 18.0,
        focus_distance: float = 400.0,
        horizontal_aperture: float = 15.0,
        clipping_range: tuple = (0.1, 1.0e5),
        convention: str = "opengl",
    ):
        """Create camera based on the ``use_tiled_camera`` attribute."""

        # NOTE(quat-order change from Lab 2.x to Lab 3.0): (w, x, y, z) to (x, y, z, w)
        rot = wxyz_to_xyzw(rot)

        spawn_cfg = sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=focus_distance,
            horizontal_aperture=horizontal_aperture,
            clipping_range=clipping_range,
        )

        if self.use_tiled_camera:
            return TiledCameraCfg(
                prim_path=prim_path,
                class_type=TiledCamera,
                update_period=0.0,
                height=self.camera_height,
                width=self.camera_width,
                data_types=["rgb"],
                spawn=spawn_cfg,
                offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention=convention),
            )
        else:
            return CameraCfg(
                prim_path=prim_path,
                class_type=Camera,
                update_period=0.05,
                height=self.camera_height,
                width=self.camera_width,
                data_types=["rgb", "distance_to_image_plane"],
                spawn=spawn_cfg,
                offset=CameraCfg.OffsetCfg(pos=pos, rot=rot, convention=convention),
            )
