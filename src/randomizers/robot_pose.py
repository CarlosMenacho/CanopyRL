from __future__ import annotations
from typing import List, Optional, Sequence
import numpy as np
import mujoco

from .base import Randomizer


def _quat_mul(q2, q1):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    return np.array(
        [
            w2 * w1 - x2 * x1 - y2 * y1 - z2 * z1,
            w2 * x1 + x2 * w1 + y2 * z1 - z2 * y1,
            w2 * y1 - x2 * z1 + y2 * w1 + z2 * x1,
            w2 * z1 + x2 * y1 - y2 * x1 + z2 * w1,
        ],
        dtype=np.float32,
    )


def _axis_angle_to_quat(axis, angle):
    s = np.sin(angle / 2.0)
    return np.array([np.cos(angle / 2.0), *(s * axis)], dtype=np.float32)


class RobotPoseRandomizer(Randomizer):
    """Randomises the robot's initial pose each episode.

    Two modes, selected automatically:

    **Mocap mode** (model has mocap bodies):
        Offsets ``data.mocap_pos[0]`` and optionally ``data.mocap_quat[0]``
        so the robot's tracked target moves slightly.

    **Joint-noise mode** (no mocap bodies — e.g. fixed-base xarm7):
        Adds small uniform noise to the ``qpos`` of each named joint,
        simulating encoder offsets and home-position calibration errors.
        Joint angles are clamped to their MuJoCo limits afterwards.
    """

    affects_spec = False
    needs_ctx = False

    def __init__(self,
                 pos_lo: Sequence[float] = (-0.04, -0.05, 0.00),
                 pos_hi: Sequence[float] = (0.04, 0.05, 0.1),
                 rot_enabled: bool = False,
                 ang_range: tuple[float, float] = (-0.15, 0.15),
                 yaw_only: bool = False,
                 joint_names: Optional[List[str]] = None,
                 joint_noise_range: tuple[float, float] = (-0.08, 0.08)):
        self.pos_lo = np.asarray(pos_lo, dtype=float)
        self.pos_hi = np.asarray(pos_hi, dtype=float)
        self.ang_lo, self.ang_hi = ang_range
        self.root_enabled = rot_enabled
        self.yaw_only = yaw_only
        # Joint-noise fallback parameters.
        self.joint_names: Optional[List[str]] = list(joint_names) if joint_names else None
        self.jnoise_lo, self.jnoise_hi = joint_noise_range

    def apply(self, *, spec, model, data, rng, ext=None):
        # ── mocap mode ────────────────────────────────────────────────────
        if model.nmocap > 0:
            dpos = rng.uniform(self.pos_lo, self.pos_hi)
            data.mocap_pos[0] += dpos

            if not self.root_enabled:
                return

            axis = (np.array([0, 0, 1], dtype=float)
                    if self.yaw_only else rng.normal(size=3))
            axis /= np.linalg.norm(axis)
            angle = rng.uniform(self.ang_lo, self.ang_hi)
            dq = _axis_angle_to_quat(axis, angle)
            data.mocap_quat[0] = _quat_mul(dq, data.mocap_quat[0])
            return

        # ── joint-noise fallback (fixed-base robot, no mocap) ─────────────
        # Resolve joint names once; fall back to all hinge joints.
        names = self.joint_names
        if names is None:
            names = [
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
                for i in range(model.njnt)
                if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE
                and mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            ]

        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid == -1:
                continue
            if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            qadr = model.jnt_qposadr[jid]
            noise = float(rng.uniform(self.jnoise_lo, self.jnoise_hi))
            new_val = data.qpos[qadr] + noise
            # Clamp to joint limits when limits are enabled.
            if model.jnt_limited[jid]:
                lo = model.jnt_range[jid, 0]
                hi = model.jnt_range[jid, 1]
                new_val = float(np.clip(new_val, lo, hi))
            data.qpos[qadr] = new_val
