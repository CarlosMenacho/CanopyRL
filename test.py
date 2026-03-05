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
from src.envs.mujoco_env import MujocoEnv
from src.rewards.factory import build_reward

# ─── defaults ─────────────────────────────────────────────────────────────────

WORLD_XML    = "ufactory_xarm7/world.xml"
EYE_CAM      = "eye_in_hand"
DISPLAY_H, DISPLAY_W = 480, 640   # high-res for the cv2 viewer window


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
    parser.add_argument("--model-config", default="config/models/default.yaml",
                        help="Model YAML (encoder latent_dim etc.)")
    parser.add_argument("--env-config", default="config/env/mujoco.yaml",
                        help="Env YAML (image size, camera name, etc.)")
    parser.add_argument("--reward-config", default="config/rewards/picking.yaml",
                        help="Reward YAML used for the live reward signal")
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

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    deterministic = not args.stochastic

    # ── load configs ──────────────────────────────────────────────────────────
    alg_cfg   = OmegaConf.load(args.alg_config)
    model_cfg = OmegaConf.load(args.model_config)
    env_cfg   = OmegaConf.load(args.env_config)

    # ── build a minimal cfg compatible with MujocoEnv ─────────────────────────
    cfg = OmegaConf.create({
        "seed": args.seed,
        "env": {
            "xml_path":          args.xml,
            "max_episode_steps": args.max_steps,
            "frame_skip":        int(env_cfg.get("frame_skip", 1)),
            "cam_name":          env_cfg.get("cam_name", EYE_CAM),
            "img_height":        int(env_cfg.get("img_height", 84)),
            "img_width":         int(env_cfg.get("img_width", 84)),
            "target_body":       env_cfg.get("target_body", "tomato_a"),
        },
        "randomization": {} if args.no_randomize else {},
    })

    # ── environment ───────────────────────────────────────────────────────────
    env = MujocoEnv(cfg, xml_path=args.xml)
    img_h, img_w, _ = env.img_shape
    latent_dim = int(model_cfg.encoder.get("latent_dim", 128))
    print(f"[test] obs_dim={env.obs_dim}  action_dim={env.action_dim}  "
          f"img={img_h}x{img_w}  latent={latent_dim}  device={device}")

    # ── load agent ────────────────────────────────────────────────────────────
    agent = SAC(alg_cfg, obs_dim=env.obs_dim, action_dim=env.action_dim,
                device=device, img_h=img_h, img_w=img_w, latent_dim=latent_dim)
    agent.load(args.checkpoint)
    agent.actor.eval()
    print(f"[test] checkpoint: {args.checkpoint}")

    # ── reward function ───────────────────────────────────────────────────────
    reward_fn = build_reward(OmegaConf.load(args.reward_config))

    # ── high-res renderer for the cv2 viewer window ───────────────────────────
    display_renderer = mujoco.Renderer(env.model, height=DISPLAY_H, width=DISPLAY_W)

    # ── episode state ─────────────────────────────────────────────────────────
    ep           = 0
    step         = 0
    ep_reward    = 0.0
    total_reward = 0.0
    obs, _       = env.reset()

    def new_episode() -> dict:
        nonlocal ep, step, ep_reward
        ep       += 1
        step      = 0
        ep_reward = 0.0
        reward_fn.reset()
        print(f"[test] ── episode {ep} ──")
        return env.reset()[0]

    ep = 1
    print(f"[test] ── episode {ep} ──")

    print()
    print("[test] R / SPACE  — reset episode")
    print("[test] Q / Esc    — quit")
    print("[test] (click the cv2 window to receive key presses)")
    print()

    # ── interactive viewer loop ───────────────────────────────────────────────
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
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
                obs = new_episode()

            # ── policy step ───────────────────────────────────────────────────
            action = agent.select_action(obs, deterministic=deterministic)
            obs, _, terminated, truncated, info = env.step(action)
            info["action"] = action
            step += 1

            reward        = reward_fn.compute(env.model, env.data, info)
            ep_reward    += reward
            total_reward += reward

            # ── auto-reset ────────────────────────────────────────────────────
            done = terminated or truncated
            if done:
                print(f"[test] episode {ep} done | "
                      f"steps={step} | ep_reward={ep_reward:+.3f}")
                if args.episodes > 0 and ep >= args.episodes:
                    break
                obs = new_episode()

            # ── eye-in-hand camera window (high-res display) ──────────────────
            display_renderer.update_scene(env.data, camera=EYE_CAM)
            rgb = display_renderer.render()
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr = _overlay(bgr, ep, step, ep_reward, total_reward, deterministic)
            cv2.imshow("eye_in_hand  [R=reset  Q=quit]", bgr)

            viewer.sync()

    display_renderer.close()
    env.close()
    cv2.destroyAllWindows()
    print(f"\n[test] done  |  episodes={ep}  total_reward={total_reward:+.3f}")


if __name__ == "__main__":
    main()
