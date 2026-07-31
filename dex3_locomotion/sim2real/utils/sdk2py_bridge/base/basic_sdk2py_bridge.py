import sys
from abc import ABC, abstractmethod

import glfw
import mujoco
import numpy as np
import pygame
from loguru import logger
from termcolor import colored

from sim2real.utils.robot import Robot


class BasicSdk2Bridge(ABC):
    """Abstract base class for SDK2Py bridge implementations."""
    
    def __init__(self, mj_model, mj_data, robot_config):
        self.robot = Robot(robot_config)
        self.sdk_type = robot_config.get("SDK_TYPE", "unitree")
        self.motor_type = robot_config.get("MOTOR_TYPE", "serial")

        # Initialize the model and data
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.free_base = robot_config["FREE_BASE"]
        base_dof = 6 if self.free_base else 0

        # num_motor comes from NUM_MOTORS (config), NOT mj_model.nu - base_dof:
        # once Dex3 finger actuators are appended to the MJCF, nu grows beyond
        # the policy's motor count, and the "nu - 6" shortcut would silently
        # absorb the finger actuators into num_motor too.
        self.num_motor = self.robot.NUM_MOTORS

        self.torques = np.zeros(self.num_motor)
        self.torque_limit = np.array(self.robot.MOTOR_EFFORT_LIMIT_LIST)

        # Address maps for the policy's NUM_MOTORS body joints/actuators.
        #
        # IMPORTANT: config["dof_names"] is NOT the physical motor ordering —
        # it's a separate list used elsewhere only to look up upper/lower-body
        # index subsets (see base_policy.py's upper_dof_indices). Its order
        # (e.g. hip_yaw, hip_roll, hip_pitch) does not match the actual
        # motor_cmd[i]/DEFAULT_DOF_ANGLES[i]/MOTOR_KP[i] ordering, which
        # follows the MJCF <actuator> declaration order (hip_pitch, hip_roll,
        # hip_yaw, ...). Using dof_names here silently swapped hip
        # pitch/roll/yaw (and others) for every joint, sending each motor's
        # PD target to the wrong physical joint — this caused the whole robot
        # to seize up/thrash the instant the policy started applying torques.
        #
        # The policy's own motor index IS the MJCF actuator declaration
        # order (actuators 0..NUM_MOTORS-1, right after the 6 free-base
        # actuators) — that part is untouched by adding Dex3 fingers, since
        # fingers are appended as new actuators after it. So motor_actuator_adr
        # is just the trivial contiguous range; only qpos/qvel need per-joint
        # lookup, because body-tree DFS order (which qpos/qvel follow) does
        # shift once Dex3 fingers are attached as children of the wrist
        # bodies (everything after the *first* wrist, i.e. the whole right
        # arm, moves later in qpos/qvel by num_hand_motor slots).
        self.motor_actuator_adr = base_dof + np.arange(self.num_motor)
        actuator_joint_id = self.mj_model.actuator_trnid[:, 0]
        motor_joint_ids = [int(actuator_joint_id[a]) for a in self.motor_actuator_adr]
        self.motor_qpos_adr = np.array([self.mj_model.jnt_qposadr[jid] for jid in motor_joint_ids])
        self.motor_dof_adr = np.array([self.mj_model.jnt_dofadr[jid] for jid in motor_joint_ids])

        # Dex3 finger actuators (if any) are additional actuators appended
        # after the policy's body-motor block, driven by a separate
        # bridge/channel (see UnitreeSdk2Bridge._init_dex3_hand_bridge) and
        # resolved by joint NAME there.
        self.num_hand_motor = self.mj_model.nu - base_dof - self.num_motor

        self.use_sensor = self.robot.USE_SENSOR  # True: use sensor data; False: use ground truth data
        # Check if the robot is using sensor data
        self.have_imu_ = False
        self.have_frame_sensor_ = False
        if self.use_sensor:
            MOTOR_SENSOR_NUM = 3
            self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
            # Check sensor
            for i in range(self.dim_motor_sensor, self.mj_model.nsensor):
                name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
                if name == "imu_quat":
                    self.have_imu_ = True
                if name == "frame_pos":
                    self.have_frame_sensor_ = True

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }
        self.joystick = None

        # Initialize SDK-specific components
        self._init_sdk_components()

    @abstractmethod
    def _init_sdk_components(self):
        """Initialize SDK-specific components. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def LowCmdHandler(self, msg):
        """Handle low-level command messages. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def PublishLowState(self):
        """Publish low-level state. Must be implemented by subclasses."""
        pass

    def PublishWirelessController(self):
        """Publish wireless controller data."""
        if self.joystick is not None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(self.button_id["RB"])
            key_state[self.key_map["L1"]] = self.joystick.get_button(self.button_id["LB"])
            key_state[self.key_map["start"]] = self.joystick.get_button(self.button_id["START"])
            key_state[self.key_map["select"]] = self.joystick.get_button(self.button_id["SELECT"])
            key_state[self.key_map["R2"]] = self.joystick.get_axis(self.axis_id["RT"]) > 0
            key_state[self.key_map["L2"]] = self.joystick.get_axis(self.axis_id["LT"]) > 0
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            if hasattr(self, 'wireless_controller'):
                self.wireless_controller.keys = key_value
                self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
                self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
                self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
                self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

                self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox"):
        """Setup joystick/gamepad."""
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > 0:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
        else:
            print("No gamepad detected. Exiting program.")
            sys.exit()

        if js_type == "xbox":
            if sys.platform.startswith("linux"):
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 3,  # Right stick axis x
                    "RY": 4,  # Right stick axis y
                    "LT": 2,  # Left trigger
                    "RT": 5,  # Right trigger
                    "DX": 6,  # Directional pad x
                    "DY": 7,  # Directional pad y
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 4,
                    "RB": 5,
                    "SELECT": 6,
                    "START": 7,
                    "XBOX": 8,
                    "LSB": 9,
                    "RSB": 10,
                }
            elif sys.platform == "darwin":
                self.axis_id = {
                    "LX": 0,  # Left stick axis x
                    "LY": 1,  # Left stick axis y
                    "RX": 2,  # Right stick axis x
                    "RY": 3,  # Right stick axis y
                    "LT": 4,  # Left trigger
                    "RT": 5,  # Right trigger
                }
                self.button_id = {
                    "X": 2,
                    "Y": 3,
                    "B": 1,
                    "A": 0,
                    "LB": 9,
                    "RB": 10,
                    "SELECT": 4,
                    "START": 6,
                    "XBOX": 5,
                    "LSB": 7,
                    "RSB": 8,
                    "DYU": 11,
                    "DYD": 12,
                    "DXL": 13,
                    "DXR": 14,
                }
            else:
                print("Unsupported OS. ")

        elif js_type == "switch":
            # Yuanhang: may differ for different OS, need to be checked
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")

    def PrintSceneInformation(self):
        """Print scene information for debugging."""
        print(" ")
        logger.info(colored("<<------------- Link ------------->>", "green"))
        for i in range(self.mj_model.nbody):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_BODY, i)
            if name:
                logger.info(f"link_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Joint ------------->>", "green"))
        for i in range(self.mj_model.njnt):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_JOINT, i)
            if name:
                logger.info(f"joint_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Actuator ------------->>", "green"))
        for i in range(self.mj_model.nu):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_ACTUATOR, i)
            if name:
                logger.info(f"actuator_index: {i}, name: {name}")
        print(" ")

        logger.info(colored("<<------------- Sensor ------------->>", "green"))
        index = 0
        for i in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, i)
            if name:
                logger.info(f"sensor_index: {index}, name: {name}, dim: {self.mj_model.sensor_dim[i]}")
            index = index + self.mj_model.sensor_dim[i]
        print(" ")


class ElasticBand:
    """
    ref: https://github.com/unitreerobotics/unitree_mujoco
    """

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True
        self.estimate = False
        self.apply_force = 0

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
        if key == glfw.KEY_SPACE:
            self.estimate = not self.estimate
        if key == glfw.KEY_0:
            self.apply_force -= 10
            self.apply_force = np.clip(self.apply_force, -70, 70)
