import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time

model = mujoco.MjModel.from_xml_path("ufactory_xarm7/world.xml") # type: ignore
data = mujoco.MjData(model) # type: ignore

renderer = mujoco.Renderer(model, height=480, width=640)

cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "eye_in_hand") # type: ignore
print(f"Camera 'eye_in_hand' ID: {cam_id}")
print(f"Camera body ID: {model.cam_bodyid[cam_id]}")

print("Available joints:")
for i in range(model.njnt):
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) # type: ignore
    if joint_name:
        print(f"  Joint {i}: {joint_name}")
    else:
        print(f"  Joint {i}: [unnamed]")

joint_names = [
    "joint1", "joint2", "joint3", "joint4", 
    "joint5", "joint6", "joint7"
]

joint_indices = []
for name in joint_names:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) # type: ignore
    if joint_id != -1:
        joint_indices.append(joint_id)
        print(f"Found joint '{name}' with ID: {joint_id}")
    else:
        print(f"Joint '{name}' not found in model")

step_count = 0
with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()

    while viewer.is_running():
        sim_time = data.time

        for i in range(7):
            data.ctrl[i] = 0.5 * np.sin(sim_time * 2 + i)

        # Step simulation
        mujoco.mj_step(model, data) # type: ignore

        if step_count % 500 == 0:
            cam_body = model.cam_bodyid[cam_id]
            print(f"[step {step_count}] cam body pos: {data.xpos[cam_body]}")
        step_count += 1

        # Render RGB from eye-in-hand camera
        renderer.update_scene(data, camera="eye_in_hand")
        rgb = renderer.render()

        # Render depth
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="eye_in_hand")
        depth = renderer.render()
        renderer.disable_depth_rendering()

        # Display with cv2
        cv2.imshow("RGB", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        cv2.imshow("Depth", depth_normalized)
        cv2.waitKey(1)

        # Sync viewer
        viewer.sync()