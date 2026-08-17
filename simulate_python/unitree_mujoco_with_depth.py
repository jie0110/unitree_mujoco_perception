import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
from dataclasses import dataclass

import numpy as np
import cv2

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config
from elevation_map import MujocoElevationMap


locker = threading.Lock()
bridge_ready = threading.Event()
unitree_bridge = None

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

env_map = MujocoElevationMap(size_x=1.6, size_y=1.0, resolution=0.1, ray_start_height=0.13792)
pelvis_link_id = mj_model.body("torso_link").id


if config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


# ---------------------------------------------------------------------------
# Depth image processing
# ---------------------------------------------------------------------------

@dataclass
class DepthProcessorConfig:
    sim_width:  int   = 64
    sim_height: int   = 36

    # Rows/cols to remove after rendering (in rendered-image coordinate space)
    crop_up:    int   = 18
    crop_down:  int   = 0
    crop_left:  int   = 16
    crop_right: int   = 16

    gaussian_kernel: tuple = (3, 3)
    gaussian_sigma:  float = 1.0

    depth_min:  float = 0.0
    depth_max:  float = 2.5
    normalize:  bool  = True
    out_min:    float = 0.0
    out_max:    float = 1.0

    # Blind-spot mask (disabled by default)
    blind_up:    int  = 0
    blind_down:  int  = 0
    blind_left:  int  = 0
    blind_right: int  = 0

    @property
    def out_height(self) -> int:
        return self.sim_height - self.crop_up - self.crop_down   # 36-18-0 = 18

    @property
    def out_width(self) -> int:
        return self.sim_width - self.crop_left - self.crop_right  # 64-16-16 = 32


CFG = DepthProcessorConfig()


def img_process(img: np.ndarray) -> np.ndarray:
    """Full depth-image processing pipeline. Returns float32 array normalised to [0, 1]."""
    img = img.copy()
    img = cv2.resize(img, (CFG.sim_width, CFG.sim_height), interpolation=cv2.INTER_NEAREST)
    # 1. Crop
    h, w = img.shape
    top    = CFG.crop_up
    bottom = h - CFG.crop_down  if CFG.crop_down  > 0 else h
    left   = CFG.crop_left
    right  = w - CFG.crop_right if CFG.crop_right > 0 else w
    img = img[top:bottom, left:right]

    # 2. Gaussian noise
    noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
    img = img.astype(np.float32) + noise

    # 3. Blind-spot mask
    if any([CFG.blind_up, CFG.blind_down, CFG.blind_left, CFG.blind_right]):
        rh, rw = img.shape
        if CFG.blind_up    > 0: img[:CFG.blind_up, :]         = 0.0
        if CFG.blind_down  > 0: img[rh - CFG.blind_down:, :]  = 0.0
        if CFG.blind_left  > 0: img[:, :CFG.blind_left]       = 0.0
        if CFG.blind_right > 0: img[:, rw - CFG.blind_right:] = 0.0

    # 4. Inpaint invalid/near pixels (< 0.2 m)
    invalid_mask = (img < 0.2).astype(np.uint8)
    if invalid_mask.any():
        img = cv2.inpaint(img, invalid_mask, inpaintRadius=3, flags=cv2.INPAINT_NS)

    # 5. Gaussian blur
    img = cv2.GaussianBlur(img, CFG.gaussian_kernel, CFG.gaussian_sigma, CFG.gaussian_sigma)

    # 6. Clip + normalize to [out_min, out_max]
    img = np.clip(img, CFG.depth_min, CFG.depth_max)
    if CFG.normalize and (CFG.depth_max - CFG.depth_min) > 1e-6:
        img = (img - CFG.depth_min) / (CFG.depth_max - CFG.depth_min)
        img = img * (CFG.out_max - CFG.out_min) + CFG.out_min

    return img.astype(np.float32)

# ---------------------------------------------------------------------------
# Threads
# ---------------------------------------------------------------------------

def SimulationThread():
    global mj_data, mj_model, unitree_bridge

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    unitree_bridge = unitree
    bridge_ready.set()

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        if config.ENABLE_ELASTIC_BAND:
            if elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        env_map.update(mj_model, mj_data, pelvis_link_id)
        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        env_map.add_visual_markers(viewer.user_scn)
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


def DepthRenderThread():
    renderer = mujoco.Renderer(
        mj_model, height=config.CAMERA_HEIGHT, width=config.CAMERA_WIDTH
    )
    renderer.enable_depth_rendering()

    bridge_ready.wait()  # wait until UnitreeSdk2Bridge is initialised

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()
        renderer.update_scene(mj_data, camera=config.CAMERA_NAME)
        locker.release()

        depth = renderer.render()

        # Clip invalid values before processing
        depth[depth < 0] = 0.0

        # Full processing pipeline → (out_height, out_width), normalised [0, 1]
        processed = img_process(depth)

        # Debug print
        cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
        valid_pixels = depth[depth > 0]
        # print(
        #     f"depth center={depth[cy, cx]:.4f}  "
        #     f"range=[{valid_pixels.min() if len(valid_pixels) else 0:.3f}, "
        #     f"{depth.max():.3f}]"
        # )

        # Publish raw + processed point clouds via DDS bridge
        unitree_bridge.PublishCameraData(depth, processed)
        cv2.imshow("depth image", (depth/depth.max()*255).astype(np.uint8))
        # OpenCV visualisation
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite("depth_raw.png",
                        (np.clip(depth, 0, config.CAMERA_MAX_DEPTH) / config.CAMERA_MAX_DEPTH * 255).astype(np.uint8))
            cv2.imwrite("depth_processed.png", (processed * 255).astype(np.uint8))

        time_until_next_frame = config.CAMERA_DT - (time.perf_counter() - step_start)
        if time_until_next_frame > 0:
            time.sleep(time_until_next_frame)

    cv2.destroyAllWindows()
    renderer.close()


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)
    depth_thread = Thread(target=DepthRenderThread, daemon=True)

    viewer_thread.start()
    sim_thread.start()
    if config.ENABLE_DEPTH_RENDER:
        depth_thread.start()
