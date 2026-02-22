import logging
from pathlib import Path

import cv2
import mujoco
import numpy as np

log = logging.getLogger(__name__)


def record_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    agent,
    reward_fn,
    n_episodes: int,
    output_dir: str,
    cam_name: str = "scene_cam",
    img_height: int = 720,
    img_width: int = 1280,
    fps: int = 30,
    max_episode_steps: int = 200,
) -> None:
    """Roll out *agent* deterministically for *n_episodes* and save MP4 files.

    Renders from the fixed ``scene_cam`` so the full robot + plant is visible.
    Each episode is written to ``output_dir/episode_NNN.mp4``.
    Rendering is done fully offscreen via ``mujoco.Renderer`` — works headless.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    renderer = mujoco.Renderer(model, height=img_height, width=img_width)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    for ep in range(1, n_episodes + 1):
        mujoco.mj_resetData(model, data)  # type: ignore[attr-defined]
        reward_fn.reset()

        video_path = str(out / f"episode_{ep:03d}.mp4")
        writer = cv2.VideoWriter(video_path, fourcc, fps, (img_width, img_height))

        obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
        total_reward = 0.0
        step = 0
        done = False

        while not done:
            action = agent.select_action(obs, deterministic=True)
            np.clip(action, -1.0, 1.0, out=action)
            data.ctrl[:] = action
            mujoco.mj_step(model, data)  # type: ignore[attr-defined]
            step += 1

            renderer.update_scene(data, camera=cam_name)
            frame_rgb = renderer.render()
            writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))

            reward = reward_fn.compute(model, data, {})
            total_reward += reward
            obs = np.concatenate([data.qpos.copy(), data.qvel.copy()])
            done = step >= max_episode_steps

        writer.release()
        log.info(f"[video] ep {ep}/{n_episodes}  steps={step}  "
                 f"reward={total_reward:+.2f}  → {video_path}")

    renderer.close()
    log.info(f"[video] {n_episodes} episode(s) saved to {output_dir}")
