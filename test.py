"""Domain-randomisation viewer for CanopyRL.

Launches the MuJoCo 3-D viewer alongside a live eye-in-hand camera window.
Press SPACE or R in the cv2 window to apply a new random episode.
Press Q or Escape to quit.

Usage::

    python test.py [--seed SEED]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from src.randomizers.factory import build_randomisers

WORLD_XML = "ufactory_xarm7/world.xml"
CAM_NAME = "eye_in_hand"
IMG_H, IMG_W = 480, 640

# Robot arm joints to apply pose noise to.  Limiting to the base and
# shoulder keeps the end-effector near the tree canopy.
_ARM_JOINTS = ["joint1", "joint2", "joint3"]

# ── helpers ───────────────────────────────────────────────────────────────────


def _reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """Reset to the home keyframe (or all-zeros if the model has none)."""
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)


def _randomise(randomisers, *, model: mujoco.MjModel, data: mujoco.MjData,
               rng: np.random.Generator) -> None:
    """Apply all model-level randomisers then recompute forward kinematics."""
    for r in randomisers:
        if not r.affects_spec:
            r.apply(spec=None, model=model, data=data, rng=rng)
    # mj_forward (not mj_step) so the arm stays frozen at its
    # randomised pose — no gravity drift, no falling to equilibrium.
    mujoco.mj_forward(model, data)


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CanopyRL domain-randomisation viewer")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    # ── randomiser config ─────────────────────────────────────────────────────
    cfg = {
        "lights": {
            "enabled": True,
            "diffuse_range": (0.05, 0.30),
            "ambient_range": (0.00, 0.20),
            "specular_range": (0.00, 0.50),
        },
        # Scope joint noise to base + shoulder only and keep the range
        # small so the end-effector stays near the canopy.
        "robot_pose": {
            "enabled": True,
            "joint_names": _ARM_JOINTS,
            "joint_noise_range": (-1.0, 0.04),
        },
        "fruit_pose": {
            "enabled": True
        },
        "camera_pose": {
            "enabled": True,
            "rot_enabled": True
        },
        "fruit_color": {
            "enabled": True
        },
        "fruit_mesh_variant": {
            "enabled": True,
            "scale_range": (0.85, 1.15)
        },
        "skybox": {
            "enabled": True,
            "tint_range": (0.55, 1.45)
        },
        "table": {
            "enabled": True
        },
        "tree_pose": {
            "enabled": True
        },
        "background_image": {
            "enabled": True
        },
        # spec-modifying — applied before compile() in training env
        "mesh_variant": {
            "enabled": True
        },
    }

    randomisers = build_randomisers(cfg)
    print(f"[viewer] {len(randomisers)} randomiser(s) active:")
    for r in randomisers:
        print(f"  {type(r).__name__:<32s} affects_spec={r.affects_spec}")

    # ── compile once ──────────────────────────────────────────────────────────
    spec = mujoco.MjSpec.from_file(WORLD_XML)
    model = spec.compile()
    data = mujoco.MjData(model)

    _reset(model, data)
    _randomise(randomisers, model=model, data=data, rng=rng)

    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)
    episode = 1

    print()
    print("[viewer] SPACE / R  — new random episode")
    print("[viewer] Q / Esc    — quit")
    print("[viewer] (click the cv2 window to receive key presses)")
    print(f"[viewer] episode {episode}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth = 150
        viewer.cam.elevation = -20
        viewer.cam.distance = 2.0
        viewer.cam.lookat[:] = [0.5, 0.0, 0.8]

        while viewer.is_running():
            # ── key input ──────────────────────────────────────────────────
            key = cv2.waitKey(1)
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("r")):
                episode += 1
                _reset(model, data)
                _randomise(randomisers, model=model, data=data, rng=rng)
                print(f"[viewer] episode {episode}")

            # mj_forward keeps the arm frozen at its randomised pose.
            # Swap for mj_step if you add a position controller later.
            mujoco.mj_forward(model, data)

            # ── eye-in-hand camera window ──────────────────────────────────
            renderer.update_scene(data, camera=CAM_NAME)
            rgb = renderer.render()
            cv2.imshow(
                f"eye_in_hand  ep={episode}  [SPACE=randomise  Q=quit]",
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

            viewer.sync()

    renderer.close()
    cv2.destroyAllWindows()
    print("[viewer] done.")


if __name__ == "__main__":
    main()
