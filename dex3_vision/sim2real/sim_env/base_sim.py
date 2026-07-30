import argparse
import sys
import threading
import time
from threading import Thread

import mujoco
import mujoco.viewer
import numpy as np
import yaml
from loguru import logger
from loop_rate_limiters import RateLimiter

sys.path.append("../")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from sim2real.utils.robot import Robot

from sim2real.utils.sdk2py_bridge import ElasticBand, create_sdk2py_bridge


class BaseSimulator:
    def __init__(self, config, head_cam=False, head_cam_name="head_camera",
                 head_cam_width=640, head_cam_height=480):
        self.config = config
        self.head_cam = head_cam
        self.head_cam_name = head_cam_name
        self.head_cam_width = head_cam_width
        self.head_cam_height = head_cam_height
        self._head_cam_renderer = None
        self._head_cam_publisher = None

        self.init_config()
        self.init_scene()
        self.init_factory()
        self.init_robot_bridge()

        self.sim_thread = Thread(target=self.simulation_thread)

    def init_config(self):
        self.robot = Robot(self.config)
        self.sdk_type = self.config.get("SDK_TYPE", "unitree")
        self.num_dof = self.robot.NUM_JOINTS
        self.sim_dt = self.config["SIMULATE_DT"]
        self.viewer_dt = self.config["VIEWER_DT"]
        self.torques = np.zeros(self.num_dof)
        self.hand_torques = None
        self.logger = logger
        self.rate = RateLimiter(1 / self.config["SIMULATE_DT"])

    def init_factory(self):
        if self.sdk_type == "unitree":
            if self.config.get("INTERFACE", None):
                if sys.platform == "linux":
                    self.config["INTERFACE"] = "lo"
                elif sys.platform == "darwin":
                    self.config["INTERFACE"] = "lo0"
                else:
                    raise NotImplementedError("Only support Linux and MacOS.")
                ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
            else:
                ChannelFactoryInitialize(self.config["DOMAIN_ID"])
        elif self.sdk_type == "booster":
            from booster_robotics_sdk_python import ChannelFactory

            ChannelFactory.Instance().Init(self.config["DOMAIN_ID"])
        else:
            raise NotImplementedError(f"SDK type {self.sdk_type} is not supported yet")
        self.logger.info(str.format("SDK TYPE: {0}", self.sdk_type))

    def init_scene(self):
        print(self.config["ROBOT_SCENE"])
        self.mj_model = mujoco.MjModel.from_xml_path(self.config["ROBOT_SCENE"])
        self.mj_data = mujoco.MjData(self.mj_model)
        self.mj_model.opt.timestep = self.sim_dt

        if self.head_cam:
            cam_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_CAMERA, self.head_cam_name)
            if cam_id == -1:
                raise ValueError(
                    f"camera '{self.head_cam_name}' not found in {self.config['ROBOT_SCENE']}. "
                    "Set HEAD_CAMERA_NAME in the config if the scene uses a different camera name."
                )

        base_body_name = self.config.get("BASE_BODY_NAME", "pelvis")
        self.base_id = self.mj_model.body(base_body_name).id

        # Enable the elastic band
        if self.config["ENABLE_ELASTIC_BAND"]:
            self.elastic_band = ElasticBand()
            band_attached_link_name = self.config.get("BAND_ATTACHED_LINK", "torso_link")
            self.band_attached_link = self.mj_model.body(band_attached_link_name).id
            self.viewer = mujoco.viewer.launch_passive(
                self.mj_model, self.mj_data, key_callback=self.elastic_band.MujuocoKeyCallback
            )
        else:
            self.viewer = mujoco.viewer.launch_passive(self.mj_model, self.mj_data)

    def init_robot_bridge(self):
        self.robot_bridge = create_sdk2py_bridge(self.mj_model, self.mj_data, self.config)
        if self.config["USE_JOYSTICK"]:
            if sys.platform == "linux":  # TODO [Yuanhang]: add other joystick support
                if self.config["SDK_TYPE"] == "unitree":
                    self.robot_bridge.SetupJoystick(
                        device_id=self.config["JOYSTICK_DEVICE"], js_type=self.config["JOYSTICK_TYPE"]
                    )
                else:
                    self.logger.warning(f"Joystick is not supported for {self.config['SDK_TYPE']} yet.")
            else:
                self.logger.warning("Joystick is not supported on Windows or MacOS.")

    def compute_torques(self):
        if self.robot_bridge.low_cmd:
            motor_cmd = list(self.robot_bridge.low_cmd.motor_cmd)
            qpos_adr = self.robot_bridge.motor_qpos_adr
            dof_adr = self.robot_bridge.motor_dof_adr
            try:
                for i in range(self.robot_bridge.num_motor):
                    self.torques[i] = (
                        motor_cmd[i].tau
                        + motor_cmd[i].kp * (motor_cmd[i].q - self.mj_data.qpos[qpos_adr[i]])
                        + motor_cmd[i].kd * (motor_cmd[i].dq - self.mj_data.qvel[dof_adr[i]])
                    )
            except Exception as e:
                self.logger.error(str.format("Joint {0} not found in motor_cmd: {1}", i, e))
        # Set the torque limit
        self.torques = np.clip(self.torques, -self.robot_bridge.torque_limit, self.robot_bridge.torque_limit)

        self.hand_torques = None
        if self.robot_bridge.num_hand_motor > 0:
            self.hand_torques = self.robot_bridge.compute_hand_torques(self.mj_data)

    def sim_step(self):
        self.robot_bridge.PublishLowState()
        if self.robot_bridge.joystick:
            self.robot_bridge.PublishWirelessController()
        if self.config["ENABLE_ELASTIC_BAND"]:
            if self.elastic_band.enable:
                self.mj_data.xfrc_applied[self.band_attached_link, :3] = self.elastic_band.Advance(
                    self.mj_data.qpos[:3], self.mj_data.qvel[:3]
                )
        self.compute_torques()
        # Actuator (ctrl) indices follow MJCF <actuator> declaration order,
        # not body-tree/qpos order, so body motors stay addressable via
        # motor_actuator_adr even where qpos/qvel are not contiguous (e.g.
        # once Dex3 finger actuators are appended to the MJCF).
        ctrl = np.zeros(self.mj_model.nu)
        ctrl[self.robot_bridge.motor_actuator_adr] = self.torques
        if self.hand_torques is not None:
            ctrl[self.robot_bridge.dex3_left_actuator_adr] = self.hand_torques[0]
            ctrl[self.robot_bridge.dex3_right_actuator_adr] = self.hand_torques[1]
        self.mj_data.ctrl = ctrl
        mujoco.mj_step(self.mj_model, self.mj_data)

    def _publish_head_cam_frame(self):
        # Lazily created here (not in init_scene): the GL context a Renderer
        # opens must belong to the thread that will use it, and this
        # simulation thread — not the constructor's thread — is the one
        # that calls this method every step.
        if self._head_cam_renderer is None:
            self._head_cam_renderer = mujoco.Renderer(
                self.mj_model, height=self.head_cam_height, width=self.head_cam_width
            )
            from sim2real.utils.head_cam_shm import HeadCamPublisher
            self._head_cam_publisher = HeadCamPublisher(
                height=self.head_cam_height, width=self.head_cam_width
            )
            self.logger.info(f"Head camera publishing '{self.head_cam_name}' "
                              f"({self.head_cam_width}x{self.head_cam_height}) to shared memory")
        self._head_cam_renderer.update_scene(self.mj_data, camera=self.head_cam_name)
        frame_rgb = self._head_cam_renderer.render()
        self._head_cam_publisher.publish(frame_rgb)

    def simulation_thread(self):
        sim_cnt = 0
        start_time = time.time()
        while self.viewer.is_running():
            self.sim_step()
            if sim_cnt % (self.viewer_dt / self.sim_dt) == 0:
                self.viewer.sync()
                if self.head_cam:
                    self._publish_head_cam_frame()
            # Get FPS
            sim_cnt += 1
            if sim_cnt % 100 == 0:
                end_time = time.time()
                self.logger.info(str.format("FPS: {0:.2f}", 100 / (end_time - start_time)))
                start_time = end_time
            self.rate.sleep()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof.yaml", help="config file")
    parser.add_argument("--head_cam", action="store_true",
                         help="Render the head camera each viewer-sync tick and publish it to "
                              "shared memory for loco_manip_xr.py to forward to the headset.")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    simulation = BaseSimulator(
        config,
        head_cam=args.head_cam,
        head_cam_name=config.get("HEAD_CAMERA_NAME", "head_camera"),
    )
    simulation.sim_thread.start()
