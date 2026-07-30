import numpy as np

from ..base import BasicSdk2Bridge
from unitree_sdk2py.utils.crc import CRC

# Dex3-1 per-joint torque limits (thumb roots need more torque than the
# other finger joints). From the Dex3-1 URDF actuatorfrcrange, same order as
# UnitreeSdk2Bridge._DEX3_LEFT_JOINTS / _DEX3_RIGHT_JOINTS.
DEX3_TORQUE_LIMIT = np.array([2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4])


class UnitreeSdk2Bridge(BasicSdk2Bridge):
    """Unitree SDK2Py bridge implementation."""

    def _init_sdk_components(self):
        """Initialize Unitree SDK-specific components."""
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
        
        robot_type = self.robot.ROBOT_TYPE
        
        # Initialize based on robot type
        if "g1" in robot_type or "h1-2" in robot_type:
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
            from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
        elif robot_type == "h1" or robot_type == "go2":
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
            from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default
            from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_

            self.low_cmd = unitree_go_msg_dds__LowCmd_()
        else:
            # Raise an error if robot_type is not valid
            raise ValueError(f"Invalid robot type '{robot_type}'. Expected 'g1', 'h1', or 'go2'.")
        
        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher("rt/lowstate", LowState_)
        self.low_state_puber.Init()

        self.low_cmd_suber = ChannelSubscriber("rt/lowcmd", LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 1)

        # Initialize crc
        self.crc = CRC()

        # Initialize wireless controller for Unitree
        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher("rt/wirelesscontroller", WirelessController_)
        self.wireless_controller_puber.Init()

        # Dex3-1 hand bridge: separate DDS channel from the policy's rt/lowcmd,
        # mirroring the real robot (see xr_teleoperate robot_hand_unitree.py).
        # Only wired up if the MJCF actually has finger actuators beyond the
        # policy's NUM_MOTORS (see BasicSdk2Bridge.num_hand_motor).
        if self.num_hand_motor > 0:
            self._init_dex3_hand_bridge()

    # Order per Unitree Dex3-1 docs ("Sort by message structure"), matching
    # xr_teleoperate/teleop/robot_control/robot_hand_unitree.py's
    # Dex3_1_Left_JointIndex / Dex3_1_Right_JointIndex — this is the order
    # HandCmd_/HandState_ motor_cmd/motor_state arrays use on the wire.
    # NOTE: left and right are NOT mirror-symmetric here — left is
    # thumb/middle/index, right is thumb/index/middle. Swapping this breaks
    # ctrl_dual_hand's per-joint indexing on the real controller and would
    # send middle-finger commands to the index finger (and vice versa) on
    # the right hand only.
    _DEX3_LEFT_JOINTS = [
        "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
        "left_hand_middle_0_joint", "left_hand_middle_1_joint",
        "left_hand_index_0_joint", "left_hand_index_1_joint",
    ]
    _DEX3_RIGHT_JOINTS = [
        "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
        "right_hand_index_0_joint", "right_hand_index_1_joint",
        "right_hand_middle_0_joint", "right_hand_middle_1_joint",
    ]

    def _init_dex3_hand_bridge(self):
        from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_, unitree_hg_msg_dds__HandState_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_, HandState_

        if self.num_hand_motor != 14:
            raise ValueError(
                f"Dex3 bridge expects 14 finger actuators (7 left + 7 right), got {self.num_hand_motor}. "
                "Check the MJCF actuator block."
            )

        # Name -> MJCF address maps for the 14 finger joints, same rationale
        # as BasicSdk2Bridge.motor_{qpos,dof,actuator}_adr: don't assume a
        # contiguous block, resolve every joint/actuator by name.
        actuator_joint_id = self.mj_model.actuator_trnid[:, 0]

        def _build_maps(joint_names):
            joint_ids = [self.mj_model.joint(name).id for name in joint_names]
            qpos_adr = np.array([self.mj_model.jnt_qposadr[jid] for jid in joint_ids])
            dof_adr = np.array([self.mj_model.jnt_dofadr[jid] for jid in joint_ids])
            actuator_adr = np.array(
                [int(np.where(actuator_joint_id == jid)[0][0]) for jid in joint_ids]
            )
            return qpos_adr, dof_adr, actuator_adr

        self.dex3_left_qpos_adr, self.dex3_left_dof_adr, self.dex3_left_actuator_adr = _build_maps(
            self._DEX3_LEFT_JOINTS
        )
        self.dex3_right_qpos_adr, self.dex3_right_dof_adr, self.dex3_right_actuator_adr = _build_maps(
            self._DEX3_RIGHT_JOINTS
        )

        self.dex3_left_cmd = unitree_hg_msg_dds__HandCmd_()
        self.dex3_right_cmd = unitree_hg_msg_dds__HandCmd_()
        self.dex3_left_state = unitree_hg_msg_dds__HandState_()
        self.dex3_right_state = unitree_hg_msg_dds__HandState_()

        self.dex3_left_cmd_suber = ChannelSubscriber("rt/dex3/left/cmd", HandCmd_)
        self.dex3_left_cmd_suber.Init(self._Dex3LeftCmdHandler, 1)
        self.dex3_right_cmd_suber = ChannelSubscriber("rt/dex3/right/cmd", HandCmd_)
        self.dex3_right_cmd_suber.Init(self._Dex3RightCmdHandler, 1)

        self.dex3_left_state_puber = ChannelPublisher("rt/dex3/left/state", HandState_)
        self.dex3_left_state_puber.Init()
        self.dex3_right_state_puber = ChannelPublisher("rt/dex3/right/state", HandState_)
        self.dex3_right_state_puber.Init()

    def _Dex3LeftCmdHandler(self, msg):
        if msg:
            self.dex3_left_cmd = msg

    def _Dex3RightCmdHandler(self, msg):
        if msg:
            self.dex3_right_cmd = msg

    def LowCmdHandler(self, msg):
        """Handle Unitree low-level command messages."""
        if msg:
            self.low_cmd = msg

    def PublishLowState(self):
        """Publish Unitree low-level state."""
        if self.mj_data is None:
            return

        sensor = self.mj_data.sensordata
        qpos = self.mj_data.qpos
        qvel = self.mj_data.qvel
        qacc = self.mj_data.qacc
        actuator_force = self.mj_data.actuator_force
        num_motors = self.num_motor
        imu = self.low_state.imu_state

        motor_state = self.low_state.motor_state
        if self.use_sensor:
            for i in range(num_motors):
                m = motor_state[i]
                m.q = sensor[i]
                m.dq = sensor[i + num_motors]
                m.tau_est = sensor[i + 2 * num_motors]
        else:
            for i in range(num_motors):
                m = motor_state[i]
                m.q = qpos[self.motor_qpos_adr[i]]
                m.dq = qvel[self.motor_dof_adr[i]]
                m.ddq = qacc[self.motor_dof_adr[i]]
                m.tau_est = actuator_force[self.motor_actuator_adr[i]]

        if self.use_sensor and self.have_frame_sensor_:
            imu.quaternion[:] = sensor[self.dim_motor_sensor + 0 : self.dim_motor_sensor + 4]
            imu.gyroscope[:] = sensor[self.dim_motor_sensor + 4 : self.dim_motor_sensor + 7]
        else:
            imu.quaternion[:] = qpos[3:7]
            imu.gyroscope[:] = qvel[3:6]

        if self.use_sensor and self.have_frame_sensor_:
            imu.accelerometer[:] = sensor[self.dim_motor_sensor + 7 : self.dim_motor_sensor + 10]
        else:
            imu.accelerometer[:] = qacc[0:3]

        self.low_state.tick = int(self.mj_data.time * 1e3)
        self.low_state.crc = self.crc.Crc(self.low_state)
        self.low_state_puber.Write(self.low_state)

        if self.num_hand_motor > 0:
            self._publish_dex3_hand_state(qpos, qvel, actuator_force)

    def _publish_dex3_hand_state(self, qpos, qvel, actuator_force):
        for state, qpos_adr, dof_adr, act_adr in (
            (self.dex3_left_state, self.dex3_left_qpos_adr, self.dex3_left_dof_adr, self.dex3_left_actuator_adr),
            (self.dex3_right_state, self.dex3_right_qpos_adr, self.dex3_right_dof_adr, self.dex3_right_actuator_adr),
        ):
            for i in range(7):
                m = state.motor_state[i]
                m.q = qpos[qpos_adr[i]]
                m.dq = qvel[dof_adr[i]]
                m.tau_est = actuator_force[act_adr[i]]
        self.dex3_left_state_puber.Write(self.dex3_left_state)
        self.dex3_right_state_puber.Write(self.dex3_right_state)

    def compute_hand_torques(self, mj_data):
        """PD torque for the 14 Dex3 finger actuators, driven by rt/dex3/{left,right}/cmd.

        Returns (left_torques(7), right_torques(7)). Before any command has
        arrived, kp/kd/tau default to 0 (same convention as low_cmd), so
        fingers stay passive/zero-torque until xr_teleoperate starts publishing.
        """
        qpos = mj_data.qpos
        qvel = mj_data.qvel

        def _pd(cmd, qpos_adr, dof_adr):
            motor_cmd = list(cmd.motor_cmd)
            tau = np.zeros(7)
            for i in range(7):
                tau[i] = (
                    motor_cmd[i].tau
                    + motor_cmd[i].kp * (motor_cmd[i].q - qpos[qpos_adr[i]])
                    + motor_cmd[i].kd * (motor_cmd[i].dq - qvel[dof_adr[i]])
                )
            return np.clip(tau, -DEX3_TORQUE_LIMIT, DEX3_TORQUE_LIMIT)

        left = _pd(self.dex3_left_cmd, self.dex3_left_qpos_adr, self.dex3_left_dof_adr)
        right = _pd(self.dex3_right_cmd, self.dex3_right_qpos_adr, self.dex3_right_dof_adr)
        return left, right
