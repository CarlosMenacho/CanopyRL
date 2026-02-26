"""CanopyRL — trained policy visualiser.

Loads a checkpoint and runs the agent live in the MuJoCo interactive viewer.
Simultaneously shows the eye-in-hand camera in a cv2 window with a stats overlay.

Controls (focus the cv2 window):
    R / SPACE  — reset to a new episode (re-randomise if enabled)
    Q / Esc    — quit

Usage::

    python test.py --checkpoint outputs/.../checkpoints/best.pt
    python test.py --checkpoint best.pt --episodes 5 --no-randomize
    python test.py --checkpoint best.pt --stochastic --seed 42
    python test.py --checkpoint best.pt --reward-config config/rewards/grasp.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import mujoco
import mujoco.viewer
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).parent))

from src.algorithms.sac import SAC
from src.rewards.factory import build_reward
from src.randomizers.factory import build_randomisers

# ─── defaults ─────────────────────────────────────────────────────────────────

WORLD_XML    = "ufactory_xarm7/world.xml"
EYE_CAM      = "eye_in_hand"
IMG_H, IMG_W = 480, 640


# ─── helpers ──────────────────────────────────────────────────────────────────

def _overlay(
    img: np.ndarray,
    ep: int,
    step: int,
    ep_reward: float,
    total_reward: float,
    deterministic: bool,
) -> np.ndarray:
    """Burn episode stats onto *img*."""
    out = img.copy()
    mode = "det" if deterministic else "stoch"
    lines = [
        f"Episode : {ep}",
        f"Step    : {step}",
        f"Ep rew  : {ep_reward:+.3f}",
        f"Total   : {total_reward:+.3f}",
        f"Policy  : {mode}",
    ]
    y = 28
    for line in lines:
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(out, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.60, (230, 230, 230), 1, cv2.LINE_AA)
        y += 26
    return out


def _reset(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
    else:
        mujoco.mj_resetData(model, data)


def _apply_randomisers(randomisers, model, data, rng) -> None:
    for r in randomisers:
        if not r.affects_spec:
            r.apply(spec=None, model=model, data=data, rng=rng)
    mujoco.mj_forward(model, data)


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="CanopyRL policy visualiser")
    parser.add_argument(
        "--checkpoint", required=True,
        help="Path to .pt checkpoint (e.g. checkpoints/best.pt)",
    )
    parser.add_argument("--xml", default=WORLD_XML,
                        help=f"Path to world XML (default: {WORLD_XML})")
    parser.add_argument("--alg-config", default="config/algorithm/sac.yaml",
                        help="Algorithm YAML — must match the trained network architecture")
    parser.add_argument("--reward-config", default="config/rewards/reach.yaml",
                        help="Reward YAML used for the live reward signal")
    parser.add_argument("--rand-config", default="config/randomization/rand_conf.yaml",
                        help="Randomisation YAML (pass '' to skip)")
    parser.add_argument("--episodes", type=int, default=0,
                        help="Number of episodes to run (0 = infinite)")
    parser.add_argument("--max-steps", type=int, default=400,
                        help="Max steps per episode before auto-reset")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-randomize", action="store_true",
                        help="Disable domain randomisation")
    parser.add_argument("--stochastic", action="store_true",
                        help="Stochastic policy (default: deterministic)")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    deterministic = not args.stochastic

    # ── randomisers ───────────────────────────────────────────────────────────
    randomisers = []
    if not args.no_randomize and args.rand_config:
        rand_cfg = OmegaConf.to_container(
            OmegaConf.load(args.rand_config), resolve=True
        )
        randomisers = build_randomisers(rand_cfg)
        print(f"[test] {len(randomisers)} randomiser(s) active:")
        for r in randomisers:
            print(f"       {type(r).__name__}")

    # ── load MuJoCo model ─────────────────────────────────────────────────────
    model = mujoco.MjModel.from_xml_path(args.xml)  # type: ignore[attr-defined]
    data  = mujoco.MjData(model)
    obs_dim    = model.nq + model.nv
    action_dim = model.nu
    print(f"[test] obs_dim={obs_dim}  action_dim={action_dim}  device={device}")

    # ── load agent ────────────────────────────────────────────────────────────
    alg_cfg = OmegaConf.load(args.alg_config)
    agent   = SAC(alg_cfg, obs_dim=obs_dim, action_dim=action_dim, device=device)
    agent.load(args.checkpoint)
    agent.actor.eval()
    print(f"[test] checkpoint: {args.checkpoint}")

    # ── reward function ───────────────────────────────────────────────────────
    reward_fn = build_reward(OmegaConf.load(args.reward_config))

    # ── offscreen renderer (eye-in-hand) ──────────────────────────────────────
    renderer = mujoco.Renderer(model, height=IMG_H, width=IMG_W)

    # ── episode state ─────────────────────────────────────────────────────────
    ep           = 0
    step         = 0
    ep_reward    = 0.0
    total_reward = 0.0
    obs          = np.zeros(obs_dim, dtype=np.float32)

    def new_episode() -> None:
        nonlocal ep, step, ep_reward, obs
        ep       += 1
        step      = 0
        ep_reward = 0.0
        _reset(model, data)
        if randomisers:
            _apply_randomisers(randomisers, model, data, rng)
        else:
            mujoco.mj_forward(model, data)
        obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        reward_fn.reset()
        print(f"[test] ── episode {ep} ──")

    new_episode()

    print()
    print("[test] R / SPACE  — reset episode")
    print("[test] Q / Esc    — quit")
    print("[test] (click the cv2 window to receive key presses)")
    print()

    # ── interactive viewer loop ───────────────────────────────────────────────
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 150
        viewer.cam.elevation = -20
        viewer.cam.distance  = 2.0
        viewer.cam.lookat[:] = [0.5, 0.0, 0.8]

        while viewer.is_running():

            # ── keyboard input (cv2 window) ───────────────────────────────────
            key = cv2.waitKey(16)   # ≈60 Hz
            if key in (ord("q"), 27):
                break
            if key in (ord(" "), ord("r")):
                print(f"[test] episode {ep} reset | "
                      f"steps={step} | ep_reward={ep_reward:+.3f}")
                new_episode()

            # ── policy step ───────────────────────────────────────────────────
            action = agent.select_action(obs, deterministic=deterministic)
            np.clip(action, -1.0, 1.0, out=action)
            data.ctrl[:] = action
            mujoco.mj_step(model, data)  # type: ignore[attr-defined]
            step += 1

            reward        = reward_fn.compute(model, data, {})
            ep_reward    += reward
            total_reward += reward
            obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])

            # ── auto-reset ────────────────────────────────────────────────────
            if step >= args.max_steps:
                print(f"[test] episode {ep} done | "
                      f"steps={step} | ep_reward={ep_reward:+.3f}")
                if args.episodes > 0 and ep >= args.episodes:
                    break
                new_episode()

            # ── eye-in-hand camera window ─────────────────────────────────────
            renderer.update_scene(data, camera=EYE_CAM)
            rgb = renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = _overlay(bgr, ep, step, ep_reward, total_reward, deterministic)
            cv2.imshow("eye_in_hand  [R=reset  Q=quit]", bgr)

            viewer.sync()

    renderer.close()
    cv2.destroyAllWindows()
    print(f"\n[test] done  |  episodes={ep}  total_reward={total_reward:+.3f}")


if __name__ == "__main__":
    main()
