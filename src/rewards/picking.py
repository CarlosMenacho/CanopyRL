import mujoco
import numpy as np

from .base import BaseReward


class PickingReward(BaseReward):
    """Combined reward for fruit-picking tasks.

    Terms
    -----
    r_grasp  : 1 if both finger bodies contact the target body, else 0
    r_prox   : 1 - tanh(5 * ||tcp - target||)
    r_red    : 1 - tanh(5 * Σ_i ||fruit_i^t - fruit_i^t0||)
    r_e      : -||action||
    r_s      : -||action - action_prev||

    Total
    -----
    R = w_grasp*r_grasp + w_prox*r_prox + w_red*r_red + w_e*r_e + w_s*r_s

    All weights and body names are read from cfg (set via YAML).
    """

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self._prev_action: np.ndarray | None = None
        # Captured lazily on the first compute() call after each reset()
        self._init_fruit_pos: dict[str, np.ndarray] | None = None

    # ------------------------------------------------------------------
    # BaseReward interface
    # ------------------------------------------------------------------

    def reset(self) -> None:
        self._prev_action = None
        self._init_fruit_pos = None

    def compute(self, model, data, info: dict) -> float:
        cfg = self.cfg

        # ── weights ───────────────────────────────────────────────────
        w_grasp = float(cfg.get("w_grasp", 1.0))
        w_prox  = float(cfg.get("w_prox",  1.0))
        w_red   = float(cfg.get("w_red",   0.5))
        w_e     = float(cfg.get("w_e",     0.01))
        w_s     = float(cfg.get("w_s",     0.01))

        # ── body / site ids ──────────────────────────────────────────
        target_body = cfg.get("target_body", "tomato_a")
        target_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body)
        tcp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "link_tcp")

        left_finger_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")

        fruit_bodies = list(cfg.get("fruit_bodies", [target_body]))

        # ── lazy init: record fruit positions at episode start ────────
        if self._init_fruit_pos is None:
            self._init_fruit_pos = {
                name: data.xpos[
                    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
                ].copy()
                for name in fruit_bodies
            }

        action = np.asarray(info.get("action", np.zeros(model.nu)), dtype=float)

        # ── r_grasp: 1 if both fingers contact the target ────────────
        left_contact  = False
        right_contact = False
        for i in range(data.ncon):
            c  = data.contact[i]
            b1 = model.geom_bodyid[c.geom1]
            b2 = model.geom_bodyid[c.geom2]
            if target_id in (b1, b2):
                if left_finger_id in (b1, b2):
                    left_contact = True
                if right_finger_id in (b1, b2):
                    right_contact = True

        r_grasp = 1.0 if (left_contact and right_contact) else 0.0

        # ── r_prox: proximity of TCP to target picking point ─────────
        tcp_pos    = data.site_xpos[tcp_site_id]
        target_pos = data.xpos[target_id]
        r_prox = float(1.0 - np.tanh(5.0 * np.linalg.norm(tcp_pos - target_pos)))

        # ── r_red: penalise displacement of fruit from episode start ──
        total_displacement = sum(
            np.linalg.norm(
                data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)]
                - self._init_fruit_pos[name]
            )
            for name in fruit_bodies
        )
        r_red = float(1.0 - np.tanh(5.0 * total_displacement))

        # ── r_e: energy penalty ──────────────────────────────────────
        r_e = float(-np.linalg.norm(action))

        # ── r_s: smoothness penalty ──────────────────────────────────
        if self._prev_action is None:
            r_s = 0.0
        else:
            r_s = float(-np.linalg.norm(action - self._prev_action))
        self._prev_action = action.copy()

        return (
            w_grasp * r_grasp
            + w_prox  * r_prox
            + w_red   * r_red
            + w_e     * r_e
            + w_s     * r_s
        )
