import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config
# ---- 1. 引入高程图类 ----
from elevation_map import MujocoElevationMap

locker = threading.Lock()
unitree_bridge = None

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

# Set all robot hinge joint positions to 0 (skip freejoint bodies like cubes in the scene)
# for i in range(mj_model.njnt):
#     if mj_model.jnt_type[i] == mujoco.mjtJoint.mjJNT_HINGE:
#         mj_data.qpos[mj_model.jnt_qposadr[i]] = 0
# mujoco.mj_forward(mj_model, mj_data)

# ---- 2. 初始化高程图实例 ----
env_map = MujocoElevationMap(size_x=1.6, size_y=1.0, resolution=0.05, ray_start_height=0.1)
torso_link_id = mj_model.body("torso_link").id  # 获取 G1 的 torso_link 节点 ID 作为基准点
# pelvis_link_id = mj_model.body("pelvis").id  # 获取 G1 的 pelvis 节点 ID 作为基准点
# secondary_imu_id = mj_model.site("secondary_imu").id  # 获取 secondary_imu site ID 作为射线基准点


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


def SimulationThread():
    global mj_data, mj_model, unitree_bridge

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    unitree_bridge = unitree

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
        
        # ---- 3. 在物理步进前，更新计算高程图数据 ----
        # env_map.update(mj_model, mj_data, pelvis_link_id)
        env_map.update(mj_model, mj_data, torso_link_id)
        # env_map.update(mj_model, mj_data, secondary_imu_id)

        # ---- 3b. 通过 DDS 发布高程图射线距离数据 ----
        unitree_bridge.PublishElevationMapDist(env_map.get_dist())

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
        
        # ---- 4. 在视觉同步前，将红点注入到渲染队列中 ----
        env_map.add_visual_markers(viewer.user_scn)
        
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()