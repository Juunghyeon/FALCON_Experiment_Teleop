"""XR (Quest 2 / Quest 3 / Pico 4 / Vision Pro) teleop bridge for LocoManipPolicy.

Reuses the existing joystick-driven upper-body IK path (EE_left/right_{x,y,z},
update_waypoints(), G1_29_ArmIK_NoWrists via use_upper_body_controller) —
the only new part is refreshing those same variables from televuer's
wrist-pose stream instead of joystick deltas, gated by a deadman switch
(trigger or grip held on both controllers).

Requires config: use_upper_body_controller: true

Usage:
    python rl_policy/loco_manip/loco_manip_xr.py \
        --config=config/g1/g1_29dof_falcon.yaml \
        --model_path=models/falcon/g1_29dof.onnx \
        --cert_file=/path/to/cert.pem --key_file=/path/to/key.pem

Add first-person video from the robot's own head camera with
    --display_mode immersive --head_cam --head_cam_source robot \
    --img_server_ip <PC2 IP>
(or --head_cam_source sim for the Mujoco head camera; see head_cam_source.py).
"""

import argparse
import sys
import time

import numpy as np
import yaml
from termcolor import colored

sys.path.append("../")
sys.path.append("./rl_policy")

from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy
from sim2real.utils.head_cam_source import (
    DEFAULT_IMG_SERVER_IP,
    DEFAULT_IMG_SERVER_REQUEST_PORT,
)

# Workspace clamp for the EE targets (robot/waist frame, meters).
WORKSPACE_X = (0.10, 0.55)
WORKSPACE_Y_LEFT = (-0.05, 0.45)
WORKSPACE_Y_RIGHT = (-0.45, 0.05)
WORKSPACE_Z = (-0.25, 0.45)

# televuer's raw controller trigger/squeeze API: triggerValue is inverted
# (10.0 = not pressed, 0.0 = fully pressed) so that it lines up with the hand
# pinch convention. Booleans (left_ctrl_trigger / left_ctrl_squeeze) already
# reflect a reasonable "actively pressed" threshold from televuer itself.
STALE_TIMEOUT_S = 0.5


class LocoManipPolicyXR(LocoManipPolicy):
    def __init__(self, config, model_path, cert_file, key_file, engage_mode="any",
                 display_mode="pass-through", motion_scale=1.0, xr_mode="controller",
                 xr_debug=False, head_cam=False, head_cam_source="sim",
                 head_cam_width=640, head_cam_height=480,
                 head_cam_transport="auto",
                 img_server_ip=DEFAULT_IMG_SERVER_IP,
                 img_server_port=DEFAULT_IMG_SERVER_REQUEST_PORT,
                 dex3=False, rl_rate=50, policy_action_scale=0.25):
        if not config.get("use_upper_body_controller", False):
            raise ValueError(
                "loco_manip_xr.py requires use_upper_body_controller: true in the config "
                "(this is what builds the G1_29_ArmIK_NoWrists + waypoints_left/right path)."
            )
        if head_cam and display_mode not in ("immersive", "ego"):
            raise ValueError("--head_cam requires --display_mode immersive (or ego), got "
                              f"'{display_mode}'.")
        self.engage_mode = engage_mode
        self.motion_scale = motion_scale
        self.xr_debug = xr_debug
        self.head_cam = head_cam
        self.xr_mode = xr_mode
        self.dex3 = dex3

        super().__init__(config=config, model_path=model_path, rl_rate=rl_rate,
                          policy_action_scale=policy_action_scale)

        # The camera has to be connected before the XR display is built: the
        # real robot's image server is what tells us the frame geometry
        # (resolution, mono vs. binocular) the headset has to be set up for.
        if head_cam:
            self._init_head_cam(head_cam_source, head_cam_width, head_cam_height,
                                head_cam_transport, img_server_ip, img_server_port)
            img_shape = self._head_cam_shape
            binocular = self._head_cam_binocular
        else:
            self._use_webrtc = False
            img_shape = (head_cam_height, head_cam_width)
            binocular = False

        self._init_xr(cert_file, key_file, display_mode, xr_mode, img_shape, binocular)
        if dex3:
            self._init_dex3()

        # Deadman-switch engage state.
        self._engaged = False
        self._engage_offset_left = np.zeros(3)
        self._engage_offset_right = np.zeros(3)
        self._last_xr_time = 0.0
        self._xr_debug_count = 0

    def _init_xr(self, cert_file, key_file, display_mode, xr_mode, img_shape, binocular):
        from televuer import TeleVuerWrapper

        # immersive/ego display modes require an image transport. zmq (in-process
        # shared memory -> vuer's own zmq bridge) is what render_to_xr() feeds,
        # and is the only option for the sim camera. With the real robot the
        # image server can also expose a WebRTC stream, which the headset pulls
        # straight from the robot — no frames pass through this process at all.
        needs_image_transport = display_mode in ("immersive", "ego")
        use_webrtc = needs_image_transport and self._use_webrtc

        self.tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=(xr_mode == "hand"),
            binocular=binocular,
            img_shape=img_shape,
            display_fps=30.0,
            display_mode=display_mode,
            zmq=needs_image_transport and not use_webrtc,
            webrtc=use_webrtc,
            webrtc_url=self._head_cam_webrtc_url if use_webrtc else None,
            cert_file=cert_file,
            key_file=key_file,
            return_hand_rot_data=False,
        )
        self.logger.info(colored(
            f"XR bridge ready (display_mode={display_mode}, xr_mode={xr_mode}"
            + (f", video={'webrtc' if use_webrtc else 'zmq'} {img_shape[1]}x{img_shape[0]}"
               f"{' stereo' if binocular else ''}" if needs_image_transport and self.head_cam else "")
            + "). Open the headset browser and Enter VR.", "cyan"
        ))

    # ------------------------------------------------------------------
    # Dex3 fingers: simple open/close (not full dex_retargeting-based
    # per-finger mimicry) driven by hand-tracking pinch/squeeze, since only
    # "open vs. closed" grasping was requested. Joint order/limits below
    # match g1_29dof_dex3_freebase.xml (see the [thumb0,1,2,middle0,1,index0,1]
    # ordering used by the sim2sim DDS bridge in unitree_sdk2py_bridge.py).
    # thumb_0 (ab/adduction) is left at 0 in both poses — closing it too
    # tends to make the thumb collide with the palm/fingers in sim.
    _DEX3_OPEN_LEFT = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _DEX3_CLOSE_LEFT = np.array([0.0, 1.0472, 1.74533, -1.5708, -1.74533, -1.5708, -1.74533])
    _DEX3_OPEN_RIGHT = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    _DEX3_CLOSE_RIGHT = np.array([0.0, -1.0472, -1.74533, 1.5708, 1.74533, 1.5708, 1.74533])

    def _init_dex3(self):
        from unitree_sdk2py.core.channel import ChannelPublisher
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__HandCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import HandCmd_

        self._dex3_left_puber = ChannelPublisher("rt/dex3/left/cmd", HandCmd_)
        self._dex3_left_puber.Init()
        self._dex3_right_puber = ChannelPublisher("rt/dex3/right/cmd", HandCmd_)
        self._dex3_right_puber.Init()
        self._dex3_left_cmd = unitree_hg_msg_dds__HandCmd_()
        self._dex3_right_cmd = unitree_hg_msg_dds__HandCmd_()
        for cmd in (self._dex3_left_cmd, self._dex3_right_cmd):
            for i in range(7):
                # RIS mode byte the real Dex3 firmware expects, matching
                # xr_teleoperate's robot_hand_unitree.py _RIS_Mode: id in bits
                # 0-3, status(0x01=enable) in bits 4-6, timeout in bit 7.
                # Without this the real hand ignores the command (sim2sim's
                # mujoco bridge doesn't check it, so this only bites on real
                # hardware).
                cmd.motor_cmd[i].mode = (i & 0x0F) | ((0x01 & 0x07) << 4)
                cmd.motor_cmd[i].kp = 1.5
                cmd.motor_cmd[i].kd = 0.2
        self.logger.info(colored("Dex3 finger open/close bridge ready", "cyan"))

    def _update_dex3(self, tele_data):
        if self.xr_mode == "hand":
            # pinchValue: ~15 (open) -> 0 (fully pinched). Normalize to a
            # 0 (open) .. 1 (closed) grasp fraction and linearly blend
            # between the open/close poses so the fingers track the pinch
            # gesture continuously rather than snapping between two states.
            left_frac = float(np.clip(1.0 - tele_data.left_hand_pinchValue / 15.0, 0.0, 1.0))
            right_frac = float(np.clip(1.0 - tele_data.right_hand_pinchValue / 15.0, 0.0, 1.0))
        else:
            # Controller mode: no continuous grasp signal, just grip driven
            # by the trigger — simple open/close, no per-finger mimicry.
            # triggerValue follows the same 10.0 (released) -> 0.0 (fully
            # pressed) convention as hand mode's pinchValue.
            left_frac = float(np.clip(1.0 - tele_data.left_ctrl_triggerValue / 10.0, 0.0, 1.0))
            right_frac = float(np.clip(1.0 - tele_data.right_ctrl_triggerValue / 10.0, 0.0, 1.0))

        left_q = self._DEX3_OPEN_LEFT + left_frac * (self._DEX3_CLOSE_LEFT - self._DEX3_OPEN_LEFT)
        right_q = self._DEX3_OPEN_RIGHT + right_frac * (self._DEX3_CLOSE_RIGHT - self._DEX3_OPEN_RIGHT)

        for i in range(7):
            self._dex3_left_cmd.motor_cmd[i].q = float(left_q[i])
            self._dex3_right_cmd.motor_cmd[i].q = float(right_q[i])
        self._dex3_left_puber.Write(self._dex3_left_cmd)
        self._dex3_right_puber.Write(self._dex3_right_cmd)

    def _init_head_cam(self, source, width, height, transport, img_server_ip, img_server_port):
        from sim2real.utils.head_cam_source import make_head_cam_source

        self._head_cam_connected = False
        self._head_cam_src = None
        self._use_webrtc = False
        self._head_cam_webrtc_url = None
        self._head_cam_shape = (height, width)
        self._head_cam_binocular = False

        try:
            self._head_cam_src = make_head_cam_source(
                source, width=width, height=height,
                img_server_ip=img_server_ip, img_server_port=img_server_port,
            )
        except Exception as e:
            hint = (
                "Start sim_env with --head_cam, e.g.:\n"
                "  python sim_env/loco_manip.py --config=... --head_cam"
                if source == "sim" else
                f"Is teleimager's image_server running on the robot's PC2 ({img_server_ip})?\n"
                "  (on PC2)  python -m teleimager.image_server"
            )
            self.logger.error(f"head camera ({source}) unavailable: {e}\n{hint}")
            return

        self._head_cam_connected = True
        self._head_cam_shape = self._head_cam_src.img_shape
        self._head_cam_binocular = self._head_cam_src.binocular

        if transport == "webrtc" and not self._head_cam_src.webrtc_enabled:
            raise ValueError(
                "--head_cam_transport webrtc requested but the image server does not "
                "have enable_webrtc set for head_camera."
            )
        self._use_webrtc = self._head_cam_src.webrtc_enabled and transport in ("auto", "webrtc")
        self._head_cam_webrtc_url = self._head_cam_src.webrtc_url

        self.logger.info(colored(
            f"head camera stream connected (source={source}, "
            f"{self._head_cam_shape[1]}x{self._head_cam_shape[0]}, "
            f"{'binocular' if self._head_cam_binocular else 'monocular'}, "
            f"video={'webrtc' if self._use_webrtc else 'zmq'})", "green"
        ))

        # Config negotiation succeeding does not prove video is flowing: with the
        # image server unreachable, teleimager falls back to a cached cam_config
        # yaml and the client comes up looking healthy but empty.
        if not self._use_webrtc and hasattr(self._head_cam_src, "wait_for_frame"):
            if not self._head_cam_src.wait_for_frame(timeout=2.0):
                self.logger.warning(colored(
                    f"connected to the camera config but no frames from {img_server_ip} yet — "
                    "the config may have come from a cached cam_config yaml rather than a live "
                    "server. Check that image_server is running on PC2 and the ZMQ port is "
                    "reachable. Continuing; the headset will stay black until frames arrive.",
                    "yellow"
                ))

    def _forward_head_cam_frame(self):
        # With WebRTC the headset pulls video from the image server itself, so
        # there is nothing for us to forward.
        if not self._head_cam_connected or self._use_webrtc:
            return
        # Sources hand back BGR because televuer's render_to_xr() always applies
        # COLOR_BGR2RGB before display.
        frame_bgr = self._head_cam_src.read_bgr()
        if frame_bgr is None:
            return
        self.tv_wrapper.render_to_xr(frame_bgr)

    # ------------------------------------------------------------------
    # Deadman switch
    # ------------------------------------------------------------------

    def _is_engaged(self, tele_data):
        if self.engage_mode == "always":
            return True
        if self.xr_mode == "hand":
            # No controller trigger/squeeze buttons in hand-tracking mode,
            # and using a hand gesture (e.g. fist) as the deadman switch
            # would collide with using that same gesture to close the Dex3
            # fingers (--dex3). So hand mode always tracks; --engage is only
            # meaningful with --xr_mode controller.
            return True
        left = right = False
        if self.engage_mode in ("any", "trigger"):
            left = left or bool(tele_data.left_ctrl_trigger)
            right = right or bool(tele_data.right_ctrl_trigger)
        if self.engage_mode in ("any", "squeeze"):
            left = left or bool(tele_data.left_ctrl_squeeze)
            right = right or bool(tele_data.right_ctrl_squeeze)
        return left and right

    # ------------------------------------------------------------------
    # Per-step XR -> EE target update
    # ------------------------------------------------------------------

    def _update_from_xr(self):
        try:
            tele_data = self.tv_wrapper.get_tele_data()
        except Exception as e:
            self.logger.error(f"XR get_tele_data failed: {e}")
            return
        if tele_data is None:
            return

        now = time.time()
        if not tele_data.motion_data_ready:
            if self.xr_debug:
                self._xr_debug_count += 1
                if self._xr_debug_count % 50 == 1:
                    print("[xr] waiting for headset motion data (Enter VR on the headset yet?)")
            return
        self._last_xr_time = now

        engaged_now = self._is_engaged(tele_data)

        if self.xr_mode == "hand":
            # WebXR hand-tracking skeleton: joint 0 is the wrist landmark.
            left_wrist = tele_data.left_hand_pos[0]
            right_wrist = tele_data.right_hand_pos[0]
        else:
            left_wrist = tele_data.left_wrist_pose[:3, 3]
            right_wrist = tele_data.right_wrist_pose[:3, 3]

        if engaged_now and not self._engaged:
            # Rising edge: zero at the current hand position so the arm
            # doesn't jump to the raw wrist coordinate on engage.
            self._engage_offset_left = np.array(
                [self.EE_left_x, self.EE_left_y, self.EE_left_z]
            ) - left_wrist * self.motion_scale
            self._engage_offset_right = np.array(
                [self.EE_right_x, self.EE_right_y, self.EE_right_z]
            ) - right_wrist * self.motion_scale
            self.logger.info(colored("XR tracking engaged", "green"))
        elif not engaged_now and self._engaged:
            self.logger.info(colored("XR tracking released", "yellow"))

        self._engaged = engaged_now

        if self._engaged:
            target_left = left_wrist * self.motion_scale + self._engage_offset_left
            target_right = right_wrist * self.motion_scale + self._engage_offset_right

            self.EE_left_x = float(np.clip(target_left[0], *WORKSPACE_X))
            self.EE_left_y = float(np.clip(target_left[1], *WORKSPACE_Y_LEFT))
            self.EE_left_z = float(np.clip(target_left[2], *WORKSPACE_Z))

            self.EE_right_x = float(np.clip(target_right[0], *WORKSPACE_X))
            self.EE_right_y = float(np.clip(target_right[1], *WORKSPACE_Y_RIGHT))
            self.EE_right_z = float(np.clip(target_right[2], *WORKSPACE_Z))

            self.update_waypoints()

        if self.dex3:
            self._update_dex3(tele_data)

        if self.xr_debug:
            self._xr_debug_count += 1
            if self._xr_debug_count % 50 == 1:
                if self.xr_mode == "hand":
                    print(
                        f"[xr] ready={int(tele_data.motion_data_ready)} "
                        f"L[pinch={tele_data.left_hand_pinchValue:.1f} sqz={int(tele_data.left_hand_squeeze)}] "
                        f"R[pinch={tele_data.right_hand_pinchValue:.1f} sqz={int(tele_data.right_hand_squeeze)}] "
                        f"engaged={int(self._engaged)} "
                        f"wristL={np.round(left_wrist, 3)} wristR={np.round(right_wrist, 3)}"
                    )
                else:
                    print(
                        f"[xr] ready={int(tele_data.motion_data_ready)} "
                        f"L[trig={int(tele_data.left_ctrl_trigger)}/{tele_data.left_ctrl_triggerValue:.1f} "
                        f"sqz={int(tele_data.left_ctrl_squeeze)}/{tele_data.left_ctrl_squeezeValue:.2f} "
                        f"A={int(tele_data.left_ctrl_aButton)} B={int(tele_data.left_ctrl_bButton)} "
                        f"stk={np.round(tele_data.left_ctrl_thumbstickValue, 2)}] "
                        f"R[trig={int(tele_data.right_ctrl_trigger)}/{tele_data.right_ctrl_triggerValue:.1f} "
                        f"sqz={int(tele_data.right_ctrl_squeeze)}/{tele_data.right_ctrl_squeezeValue:.2f} "
                        f"A={int(tele_data.right_ctrl_aButton)} B={int(tele_data.right_ctrl_bButton)}] "
                        f"engaged={int(self._engaged)} "
                        f"wristL={np.round(left_wrist, 3)} wristR={np.round(right_wrist, 3)}"
                    )

        # Controller-only buttons (start/stop/stand-walk/thumbstick locomotion):
        # televuer doesn't allocate ctrl_* shared state at all in hand-tracking
        # mode (see TeleVuer.__init__), so tele_data.right_ctrl_bButton etc.
        # would raise AttributeError there. In hand mode the policy just runs
        # continuously with locomotion held at zero; start/stop/init aren't
        # exposed via hand gestures (yet).
        if self.xr_mode == "controller":
            self._handle_xr_buttons(tele_data)
        else:
            self.lin_vel_command[:] = 0.0
            self.ang_vel_command[:] = 0.0

    def _handle_xr_buttons(self, tele_data):
        # Right B: emergency stop (edge-triggered so holding it doesn't spam).
        if tele_data.right_ctrl_bButton and not getattr(self, "_prev_right_b", False):
            self._handle_stop_policy()
            self.logger.info(colored("XR emergency stop (right B)", "red"))
        self._prev_right_b = tele_data.right_ctrl_bButton

        # Right A: start/resume policy.
        if tele_data.right_ctrl_aButton and not getattr(self, "_prev_right_a", False):
            self._handle_start_policy()
        self._prev_right_a = tele_data.right_ctrl_aButton

        # Left X: init/ready pose.
        if tele_data.left_ctrl_aButton and not getattr(self, "_prev_left_x", False):
            self._handle_init_state()
        self._prev_left_x = tele_data.left_ctrl_aButton

        # Left Y: toggle stand <-> walk.
        if tele_data.left_ctrl_bButton and not getattr(self, "_prev_left_y", False):
            self.stand_command[0, 0] = 0 if self.stand_command[0, 0] else 1
            self.logger.info(colored(f"stand_command={self.stand_command[0, 0]}", "cyan"))
        self._prev_left_y = tele_data.left_ctrl_bButton

        if self.stand_command[0, 0]:
            lx, ly = tele_data.left_ctrl_thumbstickValue
            rx, _ = tele_data.right_ctrl_thumbstickValue
            self.lin_vel_command[0, 0] = -float(ly) * 0.5
            self.lin_vel_command[0, 1] = -float(lx) * 0.5
            self.ang_vel_command[0, 0] = -float(rx) * 0.5
        else:
            self.lin_vel_command[:] = 0.0
            self.ang_vel_command[:] = 0.0

    # ------------------------------------------------------------------

    def policy_action(self):
        # Stale watchdog: if XR data stops arriving, zero locomotion commands
        # and drop the engage latch, so that a reconnect is always treated as
        # a fresh rising edge (re-zeroing the arm offset) instead of jumping
        # to wherever the operator's real wrist ended up during the dropout.
        if time.time() - self._last_xr_time > STALE_TIMEOUT_S and self._last_xr_time > 0:
            self.lin_vel_command[:] = 0.0
            self.ang_vel_command[:] = 0.0
            if self._engaged:
                self.logger.info(colored("XR tracking released (stale data)", "yellow"))
            self._engaged = False

        self._update_from_xr()
        if self.head_cam:
            self._forward_head_cam_frame()
        super().policy_action()

    def run(self):
        try:
            while True:
                self.policy_action()
                self.rate.sleep()
        except KeyboardInterrupt:
            pass
        finally:
            try:
                self.tv_wrapper.close()
            except Exception:
                pass
            if self.head_cam and self._head_cam_connected:
                try:
                    self._head_cam_src.close()
                except Exception:
                    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot (XR teleop)")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_falcon.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    parser.add_argument("--cert_file", type=str, required=True, help="path to SSL cert.pem")
    parser.add_argument("--key_file", type=str, required=True, help="path to SSL key.pem")
    parser.add_argument("--engage", type=str, default="any",
                         choices=["any", "trigger", "squeeze", "always"],
                         help="Deadman switch mode. 'always' skips the deadman switch — testing only.")
    parser.add_argument("--display_mode", type=str, default="pass-through",
                         choices=["pass-through", "immersive", "ego"])
    parser.add_argument("--motion_scale", type=float, default=1.0)
    parser.add_argument("--xr_mode", type=str, default="controller", choices=["controller", "hand"])
    parser.add_argument("--xr_debug", action="store_true")
    parser.add_argument("--head_cam", action="store_true",
                         help="Show the robot's head camera feed in the headset. "
                              "Requires --display_mode immersive (or ego).")
    parser.add_argument("--head_cam_source", type=str, default="sim", choices=["sim", "robot"],
                         help="'sim': Mujoco head camera (sim_env must run with --head_cam). "
                              "'robot': the real G1's head camera, served by teleimager's "
                              "image_server on PC2.")
    parser.add_argument("--head_cam_transport", type=str, default="auto",
                         choices=["auto", "zmq", "webrtc"],
                         help="Video transport for --head_cam_source robot. 'auto' prefers "
                              "WebRTC (headset pulls video straight from the robot) when the "
                              "image server offers it, else ZMQ via this process.")
    parser.add_argument("--img_server_ip", type=str, default=DEFAULT_IMG_SERVER_IP,
                         help="IP of the robot PC2 running teleimager's image_server.")
    parser.add_argument("--img_server_port", type=int, default=DEFAULT_IMG_SERVER_REQUEST_PORT,
                         help="teleimager camera-config request port.")
    parser.add_argument("--head_cam_width", type=int, default=640,
                         help="Sim head camera width (robot resolution comes from the server).")
    parser.add_argument("--head_cam_height", type=int, default=480,
                         help="Sim head camera height (robot resolution comes from the server).")
    parser.add_argument("--dex3", action="store_true",
                         help="Drive Dex3 finger open/close (via rt/dex3/{left,right}/cmd). "
                              "In --xr_mode hand: continuous pinch distance. "
                              "In --xr_mode controller: trigger-driven open/close.")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = LocoManipPolicyXR(
        config=config,
        model_path=model_path,
        cert_file=args.cert_file,
        key_file=args.key_file,
        engage_mode=args.engage,
        display_mode=args.display_mode,
        motion_scale=args.motion_scale,
        xr_mode=args.xr_mode,
        xr_debug=args.xr_debug,
        head_cam=args.head_cam,
        head_cam_source=args.head_cam_source,
        head_cam_width=args.head_cam_width,
        head_cam_height=args.head_cam_height,
        head_cam_transport=args.head_cam_transport,
        img_server_ip=args.img_server_ip,
        img_server_port=args.img_server_port,
        dex3=args.dex3,
        rl_rate=50,
        policy_action_scale=0.25,
    )
    policy.run()
