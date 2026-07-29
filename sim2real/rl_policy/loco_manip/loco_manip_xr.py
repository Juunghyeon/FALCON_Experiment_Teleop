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
                 xr_debug=False, head_cam=False, head_cam_width=640, head_cam_height=480,
                 rl_rate=50, policy_action_scale=0.25):
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

        super().__init__(config=config, model_path=model_path, rl_rate=rl_rate,
                          policy_action_scale=policy_action_scale)

        self._init_xr(cert_file, key_file, display_mode, xr_mode, head_cam_width, head_cam_height)
        if head_cam:
            self._init_head_cam(head_cam_width, head_cam_height)

        # Deadman-switch engage state.
        self._engaged = False
        self._engage_offset_left = np.zeros(3)
        self._engage_offset_right = np.zeros(3)
        self._last_xr_time = 0.0
        self._xr_debug_count = 0

    def _init_xr(self, cert_file, key_file, display_mode, xr_mode, img_width, img_height):
        from televuer import TeleVuerWrapper

        # immersive/ego display modes require an image transport; zmq (in-process
        # shared memory -> vuer's own zmq bridge) is what render_to_xr() feeds.
        # webrtc is the other option but needs a separate signaling server we
        # don't have, so zmq is the only one that works with our setup.
        needs_image_transport = display_mode in ("immersive", "ego")

        self.tv_wrapper = TeleVuerWrapper(
            use_hand_tracking=(xr_mode == "hand"),
            binocular=False,
            img_shape=(img_height, img_width),
            display_fps=30.0,
            display_mode=display_mode,
            zmq=needs_image_transport,
            cert_file=cert_file,
            key_file=key_file,
            return_hand_rot_data=False,
        )
        self.logger.info(colored(
            f"XR bridge ready (display_mode={display_mode}, xr_mode={xr_mode}). "
            "Open the headset browser and Enter VR.", "cyan"
        ))

    def _init_head_cam(self, width, height):
        from sim2real.utils.head_cam_shm import HeadCamSubscriber

        self._head_cam_width = width
        self._head_cam_height = height
        self._head_cam_connected = False
        try:
            self._head_cam_sub = HeadCamSubscriber(connect_timeout=10.0)
            self._head_cam_connected = True
            self.logger.info(colored("head camera stream connected", "green"))
        except TimeoutError as e:
            self.logger.error(
                f"{e}\nStart sim_env with --head_cam, e.g.:\n"
                "  python sim_env/loco_manip.py --config=... --head_cam"
            )
            self._head_cam_sub = None

    def _forward_head_cam_frame(self):
        if not self._head_cam_connected:
            return
        frame_rgb = self._head_cam_sub.read()
        if frame_rgb is None:
            return
        # televuer's render_to_xr() always applies COLOR_BGR2RGB before display,
        # so a genuinely-RGB source frame must be pre-flipped to BGR here for the
        # final result seen in the headset to come out RGB again.
        self.tv_wrapper.render_to_xr(frame_rgb[:, :, ::-1])

    # ------------------------------------------------------------------
    # Deadman switch
    # ------------------------------------------------------------------

    def _is_engaged(self, tele_data):
        if self.engage_mode == "always":
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

        if self.xr_debug:
            self._xr_debug_count += 1
            if self._xr_debug_count % 50 == 1:
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

        self._handle_xr_buttons(tele_data)

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
            self.lin_vel_command[0, 0] = float(ly) * 0.5
            self.lin_vel_command[0, 1] = -float(lx) * 0.5
            self.ang_vel_command[0, 0] = -float(rx) * 0.5
        else:
            self.lin_vel_command[:] = 0.0
            self.ang_vel_command[:] = 0.0

    # ------------------------------------------------------------------

    def policy_action(self):
        # Stale watchdog: if XR data stops arriving, zero locomotion
        # commands but leave the arm target (and everything else) alone.
        if time.time() - self._last_xr_time > STALE_TIMEOUT_S and self._last_xr_time > 0:
            self.lin_vel_command[:] = 0.0
            self.ang_vel_command[:] = 0.0

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
                         help="Forward the sim_env head camera feed to the headset. "
                              "Requires sim_env to be started with --head_cam, and "
                              "--display_mode immersive (or ego).")
    parser.add_argument("--head_cam_width", type=int, default=640)
    parser.add_argument("--head_cam_height", type=int, default=480)
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
        head_cam_width=args.head_cam_width,
        head_cam_height=args.head_cam_height,
        rl_rate=50,
        policy_action_scale=0.25,
    )
    policy.run()
