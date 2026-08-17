import time
import sys
import numpy as np

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
from unitree_sdk2py.utils.crc import CRC

G1_NUM_MOTOR = 29

# Per-joint PD gains from official G1 example
# Order: L_hip_pitch, L_hip_roll, L_hip_yaw, L_knee, L_ankle_pitch, L_ankle_roll,
#        R_hip_pitch, R_hip_roll, R_hip_yaw, R_knee, R_ankle_pitch, R_ankle_roll,
#        waist_yaw, waist_roll, waist_pitch,
#        L_shoulder_pitch, L_shoulder_roll, L_shoulder_yaw, L_elbow, L_wrist_roll, L_wrist_pitch, L_wrist_yaw,
#        R_shoulder_pitch, R_shoulder_roll, R_shoulder_yaw, R_elbow, R_wrist_roll, R_wrist_pitch, R_wrist_yaw
Kp = [
    60, 60, 60, 100, 40, 40,      # left leg
    60, 60, 60, 100, 40, 40,      # right leg
    60, 40, 40,                   # waist
    40, 40, 40, 40, 40, 40, 40,   # left arm
    40, 40, 40, 40, 40, 40, 40,   # right arm
]

Kd = [
    1, 1, 1, 2, 1, 1,    # left leg
    1, 1, 1, 2, 1, 1,    # right leg
    1, 1, 1,             # waist
    1, 1, 1, 1, 1, 1, 1, # left arm
    1, 1, 1, 1, 1, 1, 1, # right arm
]

dt = 0.002
runing_time = 0.0
crc = CRC()

input("Press enter to start")

if __name__ == '__main__':

    if len(sys.argv) < 2:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, sys.argv[1])

    pub = ChannelPublisher("rt/lowcmd", LowCmd_)
    pub.Init()

    cmd = unitree_hg_msg_dds__LowCmd_()
    cmd.mode_pr = 0
    cmd.mode_machine = 0
    for i in range(G1_NUM_MOTOR):
        cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
        cmd.motor_cmd[i].q = 0.0
        cmd.motor_cmd[i].kp = 0.0
        cmd.motor_cmd[i].dq = 0.0
        cmd.motor_cmd[i].kd = 0.0
        cmd.motor_cmd[i].tau = 0.0

    while True:
        step_start = time.perf_counter()

        runing_time += dt

        # Ramp up gains over 3 seconds to avoid sudden torque
        ratio = np.clip(runing_time / 3.0, 0.0, 1.0)

        for i in range(G1_NUM_MOTOR):
            cmd.motor_cmd[i].q = 0.0
            cmd.motor_cmd[i].dq = 0.0
            cmd.motor_cmd[i].kp = ratio * Kp[i]
            cmd.motor_cmd[i].kd = ratio * Kd[i]
            cmd.motor_cmd[i].tau = 0.0

        cmd.crc = crc.Crc(cmd)
        pub.Write(cmd)

        time_until_next_step = dt - (time.perf_counter() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
