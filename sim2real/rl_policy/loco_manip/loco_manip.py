import os
import sys
import time

import numpy as np
import argparse
import yaml

sys.path.append("../")
sys.path.append("./rl_policy")

import pinocchio as pin
from sim2real.rl_policy.dec_loco.dec_loco import DecLocomotionPolicy

from termcolor import colored
from sim2real.utils.arm_ik.robot_arm_ik_g1_23dof import G1_29_ArmIK_NoWrists
from sim2real.utils.motion_player import MotionPlayer


class LocoManipPolicy(DecLocomotionPolicy):
    def __init__(
        self, config, model_path, rl_rate=50, policy_action_scale=0.25
    ):
        # ---- Balance descriptor obs injection --------------------------------
        # Must happen BEFORE super().__init__() calls _init_obs_config(), which
        # reads obs_dict / obs_dims / obs_scales from config to build buffers.
        #
        # use_balance_descriptor: true  → inject balance_descriptor (dim=3) into
        #   actor_obs_bd (separate key, history=1). Concatenated with actor_obs at inference.
        # use_balance_descriptor: false → no BD obs (matches original FALCON, dim=575).
        self._use_balance_descriptor = bool(config.get("use_balance_descriptor", False))

        actor_obs = config.get("obs_dict", {}).get("actor_obs", [])
        obs_dict = config.get("obs_dict", {})
        history_dict = config.get("history_length_dict", {})
        if self._use_balance_descriptor:
            if "balance_descriptor" in actor_obs:
                actor_obs.remove("balance_descriptor")
            obs_dict["actor_obs_bd"] = ["balance_descriptor"]
            history_dict["actor_obs_bd"] = 1
            if "balance_descriptor" not in config.get("obs_dims", {}):
                config.setdefault("obs_dims", {})["balance_descriptor"] = 3
            if "balance_descriptor" not in config.get("obs_scales", {}):
                config.setdefault("obs_scales", {})["balance_descriptor"] = 1.0
        else:
            if "balance_descriptor" in actor_obs:
                actor_obs.remove("balance_descriptor")
            obs_dict.pop("actor_obs_bd", None)
            history_dict.pop("actor_obs_bd", None)
            config.get("obs_dims", {}).pop("balance_descriptor", None)
            config.get("obs_scales", {}).pop("balance_descriptor", None)

        if self._use_balance_descriptor:
            bd_cfg = config.get("balance_descriptor", {})
            self._bd_nominal_type     = int(bd_cfg.get("bd_nominal_type", 3))
            self._bd_d_ref            = float(bd_cfg.get("bd_d_ref", 0.10))
            self._bd_d_ref_ds         = float(bd_cfg.get("bd_d_ref_ds", 0.12))
            self._bd_d_ref_ss         = float(bd_cfg.get("bd_d_ref_ss", 0.05))
            self._bd_delta_safe       = float(bd_cfg.get("bd_delta_safe", 1.5))
            self._bd_m_safe_ds        = float(bd_cfg.get("bd_m_safe_ds", -0.40))
            self._bd_m_safe_ss        = float(bd_cfg.get("bd_m_safe_ss", -0.85))

            self._bd_h_ns_prev        = np.zeros(6)
            self._bd_h_ns_smooth      = np.zeros(6)
            self._bd_h_dot_ns_smooth  = np.zeros(6)
            self._bd_initialized      = False
            self._bd_prev_contact     = (1, 1)
            self._bd_contact_freeze_steps = 0

            self._bd_ema_alpha        = float(bd_cfg.get("bd_ema_alpha", 0.15))
            self._bd_ema_alpha_k      = float(bd_cfg.get("bd_ema_alpha_k", self._bd_ema_alpha))
            self._bd_landing_freeze_n = int(bd_cfg.get("bd_landing_freeze_n", 3))
            self._bd_h_dot_l_clamp    = float(bd_cfg.get("bd_h_dot_l_clamp", 350.0))
            self._bd_h_dot_k_clamp    = float(bd_cfg.get("bd_h_dot_k_clamp", 50.0))
            self._bd_base_z_nominal   = float(bd_cfg.get("bd_base_z_nominal", 0.75))

            self._bd_delta_lp_alpha   = float(bd_cfg.get("bd_delta_lp_alpha", 0.0))
            self._bd_delta_fall_alpha = float(bd_cfg.get("bd_delta_fall_alpha", 0.0))
            self._bd_delta_bar_D_lp   = 0.0
            self._bd_delta_p_D_lp     = np.zeros(2)
            self._bd_delta_p_D_ds     = np.zeros(2)

            self._bd_delta_p_dead     = float(bd_cfg.get("bd_delta_p_dead", 0.0))
            self._bd_m_bar_dead       = float(bd_cfg.get("bd_m_bar_dead", 0.0))
            self._bd_u_D_gate_dbar    = float(bd_cfg.get("bd_u_D_gate_dbar", 0.0))
            self._bd_pitch_safe_gain  = float(bd_cfg.get("bd_pitch_safe_gain", 0.0))
            self._bd_pitch_dot_thresh = float(bd_cfg.get("bd_pitch_dot_thresh", 0.1))
            self._bd_prev_pitch_rad   = 0.0
            self._bd_crouch_drop      = float(bd_cfg.get("bd_crouch_drop", 0.0))
            self._bd_force_zero_obs   = bool(bd_cfg.get("bd_force_zero_obs", False))
            self._bd_dp_mag_thresh    = float(bd_cfg.get("bd_dp_mag_thresh", 0.02))
            self._bd_obs_ema_alpha    = float(bd_cfg.get("bd_obs_ema_alpha", 1.0))
            self._bd_obs_ema          = np.zeros(3, dtype=np.float32)

            self._bd_mj_model = None
            self._bd_mj_data  = None
            self._bd_policy_dt = 1.0 / 50.0

            print(
                f"[BalanceDescriptor] enabled  nominal_type={self._bd_nominal_type}  "
                f"d_ref={self._bd_d_ref}  delta_safe={self._bd_delta_safe}"
            )
        else:
            print("[BalanceDescriptor] disabled — obs dim matches original FALCON (575)")

        self._last_bd_obs = np.zeros(3, dtype=np.float32)
        # ----------------------------------------------------------------------

        super().__init__(config, model_path, rl_rate, policy_action_scale)

        if self._use_balance_descriptor:
            self._init_bd_mujoco()

        self.ref_upper_dof_pos = np.zeros((1, self.num_upper_dofs))
        self.ref_upper_dof_pos += self.default_dof_angles[self.upper_dof_indices]
        self.residual_upper_body_action = self.config.get("residual_upper_body_action", False)

        self.upper_body_controller = None
        if self.config.get("use_upper_body_controller", False):
            self.init_upper_body_controller()

        self.motion_player = None
        if self.config.get("use_motion_player", False):
            self._init_motion_player()
        self._motion_player_fps_scale_step = float(
            self.config.get("motion_player_fps_scale_step", 0.25)
        )

        # Blend state for easing ref_upper_dof_pos back to the default pose
        # when the motion player is toggled OFF (see _handle_motion_player_toggle).
        self._upper_blend_src = None
        self._upper_blend_target = None
        self._upper_blend_count = 0
        self._upper_blend_steps = int(self.config.get("motion_player_blend_steps", 20))

    def get_current_obs_buffer_dict(self, robot_state_data):
        current_obs_dict = super().get_current_obs_buffer_dict(robot_state_data)
        current_obs_dict["actions"] = self.last_policy_action
        if self._use_balance_descriptor:
            current_obs_dict["balance_descriptor"] = self._get_balance_descriptor(robot_state_data)
            if self._bd_crouch_drop > 0.0:
                threat_ratio = min(self._bd_delta_bar_D_lp / max(self._bd_delta_safe, 1e-6), 1.0)
                h_nominal = float(self.config.get("DESIRED_BASE_HEIGHT", 0.75))
                self.base_height_command[0, 0] = h_nominal - self._bd_crouch_drop * threat_ratio
        else:
            current_obs_dict["balance_descriptor"] = np.zeros((1, 3), dtype=np.float32)
        current_obs_dict["command_base_height"] = self.base_height_command
        return current_obs_dict

    def _init_bd_mujoco(self):
        """Load policy-side MuJoCo model for BD computation."""
        import mujoco

        scene_path = self.config.get("ROBOT_SCENE", None)
        if scene_path is None:
            print("[BalanceDescriptor] ROBOT_SCENE not set — BD will return zeros")
            return

        if not os.path.isabs(scene_path):
            sim2real_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '../..'))
            candidates = [
                os.path.join(sim2real_dir, scene_path),
                os.path.join(os.getcwd(), scene_path),
            ]
            for c in candidates:
                if os.path.isfile(c):
                    scene_path = os.path.normpath(c)
                    break
            else:
                scene_path = os.path.normpath(candidates[0])

        print(f"[BD_INIT] scene_path={scene_path}  exists={os.path.isfile(scene_path)}")

        try:
            self._bd_mj_model = mujoco.MjModel.from_xml_path(scene_path)
            self._bd_mj_data  = mujoco.MjData(self._bd_mj_model)
            self._bd_policy_dt = 1.0 / float(self.config.get("rl_rate", 50))
            self._bd_h_ns_prev       = np.zeros(6)
            self._bd_h_ns_smooth     = np.zeros(6)
            self._bd_h_dot_ns_smooth = np.zeros(6)
            self._bd_initialized     = False
            self._bd_delta_bar_D_lp  = 0.0
            self._bd_delta_p_D_lp    = np.zeros(2)
            self._bd_delta_p_D_ds    = np.zeros(2)

            from sim2real.utils.cmm_balance import _subtree_body_ids, FOOT_ROOT_LEFT, FOOT_ROOT_RIGHT
            self._bd_left_ids  = _subtree_body_ids(self._bd_mj_model, FOOT_ROOT_LEFT)
            self._bd_right_ids = _subtree_body_ids(self._bd_mj_model, FOOT_ROOT_RIGHT)

            print(f"[BalanceDescriptor] MuJoCo loaded: {scene_path}")
        except Exception as e:
            print(f"[BalanceDescriptor] MuJoCo load failed ({e}) — BD will return zeros")
            self._bd_mj_model = None
            self._bd_mj_data  = None

    def _get_balance_descriptor(self, robot_state_data):
        """Compute BD obs (1, 3) = [u_D_x, u_D_y, delta_bar_norm]."""
        if self._bd_mj_model is None:
            if not getattr(self, '_bd_none_logged', False):
                self._bd_none_logged = True
                print("[BD] MuJoCo model not loaded — BD disabled")
            return np.zeros((1, 3), dtype=np.float32)

        try:
            import mujoco
            from sim2real.utils.cmm_balance import (
                compute_nonself_momentum,
                compute_support_polygon,
                compute_balance_descriptor,
                compute_directional_margin,
            )

            model = self._bd_mj_model
            data  = self._bd_mj_data
            N     = self.num_dofs
            rsd   = robot_state_data[0]

            data.qpos[:3]    = rsd[0:3]
            if data.qpos[2] == 0.0:
                data.qpos[2] = self._bd_base_z_nominal
            data.qpos[3:7]   = rsd[3:7]
            data.qpos[7:7+N] = rsd[7:7+N]
            data.qvel[:3]    = rsd[7+N : 7+N+3]
            data.qvel[3:6]   = rsd[7+N+3 : 7+N+6]
            data.qvel[6:6+N] = rsd[7+N+6 : 7+2*N+6]

            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_fwdVelocity(model, data)
            mujoco.mj_subtreeVel(model, data)

            left_body_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link")
            right_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link")
            foot_z_thresh = 0.08
            l_c = int(data.xpos[left_body_id, 2]  < foot_z_thresh) if left_body_id  != -1 else 1
            r_c = int(data.xpos[right_body_id, 2] < foot_z_thresh) if right_body_id != -1 else 1

            cur_contact = (l_c, r_c)
            prev_l, prev_r = self._bd_prev_contact
            landing = (l_c > prev_l) or (r_c > prev_r)
            was_frozen = self._bd_contact_freeze_steps > 0
            if landing:
                self._bd_contact_freeze_steps = self._bd_landing_freeze_n
            elif self._bd_contact_freeze_steps > 0:
                self._bd_contact_freeze_steps -= 1
            just_unfroze = was_frozen and (self._bd_contact_freeze_steps == 0)
            self._bd_prev_contact = cur_contact

            h_ns, com = compute_nonself_momentum(model, data)
            dt    = self._bd_policy_dt
            alpha = self._bd_ema_alpha

            if not self._bd_initialized:
                self._bd_h_ns_prev       = h_ns.copy()
                self._bd_h_ns_smooth     = h_ns.copy()
                self._bd_h_dot_ns_smooth = np.zeros(6)
                self._bd_initialized     = True
                self._bd_prev_contact    = cur_contact
                return np.zeros((1, 3), dtype=np.float32)

            alpha_k = self._bd_ema_alpha_k
            h_ns_smooth = np.concatenate([
                alpha   * h_ns[:3] + (1.0 - alpha)   * self._bd_h_ns_smooth[:3],
                alpha_k * h_ns[3:] + (1.0 - alpha_k) * self._bd_h_ns_smooth[3:],
            ])

            if self._bd_contact_freeze_steps > 0:
                h_dot_ns = self._bd_h_dot_ns_smooth.copy()
                self._bd_h_ns_smooth = h_ns_smooth.copy()
            elif just_unfroze:
                h_dot_ns = np.zeros(6)
                self._bd_h_ns_smooth = h_ns_smooth.copy()
            else:
                h_dot_ns = (h_ns_smooth - self._bd_h_ns_smooth) / max(dt, 1e-6)
                self._bd_h_ns_smooth = h_ns_smooth.copy()

            self._bd_h_ns_prev       = h_ns.copy()
            self._bd_h_dot_ns_smooth = h_dot_ns.copy()

            lc = self._bd_h_dot_l_clamp
            kc = self._bd_h_dot_k_clamp
            h_dot_ns[:3] = np.clip(h_dot_ns[:3], -lc, lc)
            h_dot_ns[3:] = np.clip(h_dot_ns[3:], -kc, kc)

            support_pts = compute_support_polygon(model, data)
            total_fz = float(model.body_subtreemass[1]) * 9.81

            if support_pts is not None and len(support_pts) >= 2:
                if l_c and r_c:
                    l_pos = data.xpos[left_body_id, :2] if left_body_id  != -1 else support_pts[:4].mean(0)
                    r_pos = data.xpos[right_body_id, :2] if right_body_id != -1 else support_pts[4:].mean(0)
                    p_zmp = (l_pos + r_pos) / 2.0
                elif l_c:
                    p_zmp = data.xpos[left_body_id,  :2] if left_body_id  != -1 else support_pts[:4].mean(0)
                elif r_c:
                    p_zmp = data.xpos[right_body_id, :2] if right_body_id != -1 else support_pts[4:].mean(0)
                else:
                    p_zmp = com[:2].copy()
            else:
                p_zmp = com[:2].copy()

            is_ds = (l_c == 1 and r_c == 1)
            d_ref = self._bd_d_ref

            desc = compute_balance_descriptor(
                h_dot_ns, com[2], total_fz, p_zmp, support_pts, d_ref=d_ref
            )

            dp_max = self._bd_d_ref * 10.0
            delta_p_D_raw   = np.clip(desc["delta_p_D"], -dp_max, dp_max)
            delta_bar_D_raw = float(max(0.0, desc["delta_bar_D"]))

            fall_a = self._bd_delta_fall_alpha
            if fall_a > 0.0:
                if delta_bar_D_raw >= self._bd_delta_bar_D_lp:
                    self._bd_delta_bar_D_lp = delta_bar_D_raw
                else:
                    self._bd_delta_bar_D_lp = (fall_a * delta_bar_D_raw
                                               + (1.0 - fall_a) * self._bd_delta_bar_D_lp)
                delta_bar_D = self._bd_delta_bar_D_lp
                delta_p_D   = delta_p_D_raw
            else:
                lp = self._bd_delta_lp_alpha
                if lp > 0.0:
                    self._bd_delta_bar_D_lp = lp * delta_bar_D_raw + (1.0 - lp) * self._bd_delta_bar_D_lp
                    self._bd_delta_p_D_lp   = lp * delta_p_D_raw   + (1.0 - lp) * self._bd_delta_p_D_lp
                    delta_bar_D = self._bd_delta_bar_D_lp
                    delta_p_D   = self._bd_delta_p_D_lp
                else:
                    delta_bar_D = delta_bar_D_raw
                    delta_p_D   = delta_p_D_raw

            if is_ds or delta_bar_D_raw > 0.3:
                self._bd_delta_p_D_ds = delta_p_D.copy()
            else:
                delta_p_D = self._bd_delta_p_D_ds.copy()

            dp_dead = self._bd_delta_p_dead
            if dp_dead > 0.0 and np.linalg.norm(delta_p_D) < dp_dead:
                delta_p_D = np.zeros(2)

            if self._bd_nominal_type == 3 and support_pts is not None and len(support_pts) >= 3:
                p_nom = support_pts.mean(axis=0)
                d_dir_nom = compute_directional_margin(p_nom, support_pts, desc["u_D"])
                m_bar_D = float(np.clip((desc["d_dir"] - d_dir_nom) / max(self._bd_d_ref, 1e-6), -5.0, 5.0))
            else:
                m_bar_D = desc["m_bar_D"]

            m_bar_dead = self._bd_m_bar_dead
            if m_bar_dead > 0.0 and abs(m_bar_D) < m_bar_dead:
                m_bar_D = 0.0

            qw, qx, qy, qz = rsd[3], rsd[4], rsd[5], rsd[6]
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            cy, sy = np.cos(yaw), np.sin(yaw)
            dpx_body =  cy * delta_p_D[0] + sy * delta_p_D[1]
            dpy_body = -sy * delta_p_D[0] + cy * delta_p_D[1]

            dp_mag    = float(np.sqrt(dpx_body**2 + dpy_body**2))
            dp_thresh = self._bd_dp_mag_thresh
            if dp_mag > dp_thresh:
                u_D_x = dpx_body / dp_mag
                u_D_y = dpy_body / dp_mag
            else:
                u_D_x = 0.0
                u_D_y = 0.0

            if self._bd_u_D_gate_dbar > 0.0 and delta_bar_D < self._bd_u_D_gate_dbar:
                u_D_x = 0.0
                u_D_y = 0.0

            if getattr(self, '_bd_force_zero_obs', False):
                u_D_x, u_D_y = 0.0, 0.0

            valid = float(dp_mag > dp_thresh)
            delta_bar_norm = float(np.clip(delta_bar_D / max(self._bd_delta_safe, 1e-6), 0.0, 1.0)) * valid

            raw_obs = np.array([u_D_x, u_D_y, delta_bar_norm], dtype=np.float32)
            a = self._bd_obs_ema_alpha
            if a < 1.0:
                self._bd_obs_ema = a * raw_obs + (1.0 - a) * self._bd_obs_ema
                result = self._bd_obs_ema.reshape(1, 3)
            else:
                result = raw_obs.reshape(1, 3)
            self._last_bd_obs = result[0]

            self._bd_debug_count = getattr(self, '_bd_debug_count', 0) + 1
            if self._bd_debug_count % 50 == 1:
                print(f"[BD] u_D=({u_D_x:.3f},{u_D_y:.3f}) dbar_norm={delta_bar_norm:.3f} "
                      f"delta_bar={delta_bar_D:.4f} dp_mag={dp_mag:.4f}")

            return result

        except Exception as e:
            import traceback
            if not getattr(self, '_bd_exception_logged', False):
                print(f"[BalanceDescriptor] exception: {e}")
                traceback.print_exc()
                self._bd_exception_logged = True
            self._last_bd_obs = np.zeros(3, dtype=np.float32)
            return np.zeros((1, 3), dtype=np.float32)

    def set_mujoco_refs(self, mj_model, mj_data, sim_dt=0.005):
        """Wire external MuJoCo refs when policy and simulator share a process."""
        self._bd_mj_model        = mj_model
        self._bd_mj_data         = mj_data
        self._bd_policy_dt       = float(sim_dt)
        self._bd_h_ns_prev       = np.zeros(6)
        self._bd_h_ns_smooth     = np.zeros(6)
        self._bd_h_dot_ns_smooth = np.zeros(6)
        self._bd_initialized     = False
        self._bd_delta_bar_D_lp  = 0.0
        self._bd_delta_p_D_lp    = np.zeros(2)
        self._bd_obs_ema         = np.zeros(3, dtype=np.float32)
        print(f"[BalanceDescriptor] external MuJoCo refs wired (dt={sim_dt}s)")

    def init_upper_body_controller(self):
        if self.config["ROBOT_TYPE"] == "g1_29dof":
            self.upper_body_controller = G1_29_ArmIK_NoWrists(
                Unit_Test=False, Visualization=False, robot_config=self.config
            )
        else:
            self.logger.error("Unsupported robot type: %s", self.config["ROBOT_TYPE"])
        self.waypoint_index = 0
        self.speed_factor = 0.05
        self.base_z_offset = 0.8
        self.degrees = 0
        self.theta = np.radians(self.degrees)
        self.EE_left_R = np.array(
            [[np.cos(-self.theta), -np.sin(-self.theta), 0],
             [np.sin(-self.theta),  np.cos(-self.theta), 0],
             [0, 0, 1]]
        )
        self.EE_right_R = np.array(
            [[np.cos(self.theta), -np.sin(self.theta), 0],
             [np.sin(self.theta),  np.cos(self.theta), 0],
             [0, 0, 1]]
        )
        self.EE_left_x  = 0.30
        self.EE_right_x = 0.30
        self.EE_left_y  = 0.13
        self.EE_right_y = -0.13
        self.EE_left_z  = 0.08
        self.EE_right_z = 0.08
        self.update_waypoints()
        self.EE_efrc_L = np.array([0, 0, 0, 0, 0, 0])
        self.EE_efrc_R = np.array([0, 0, 0, 0, 0, 0])
        self.upper_body_controller.set_initial_poses(
            self.waypoints_left[0].translation,
            self.waypoints_right[0].translation,
            self.waypoints_left[0].rotation,
            self.waypoints_right[0].rotation,
        )

    def rl_inference(self, robot_state_data):
        obs = self.prepare_obs_for_rl(robot_state_data)
        policy_action = self.policy(obs)
        policy_action = np.clip(policy_action, -100, 100)

        self.last_policy_action = policy_action.copy()
        scaled_policy_action = policy_action * self.policy_action_scale

        if self.residual_upper_body_action:
            scaled_policy_action[:, self.upper_dof_indices] += (
                self.ref_upper_dof_pos - self.default_dof_angles[self.upper_dof_indices]
            )

        self._infer_debug_count = getattr(self, '_infer_debug_count', 0) + 1
        if self._infer_debug_count % 100 == 1:
            print(f"[INFER] step={self._infer_debug_count}  obs_dim={obs['actor_obs'].shape}  BD={self._last_bd_obs}")

        return scaled_policy_action

    def policy_action(self):
        cmd_q   = np.zeros(self.num_dofs)
        cmd_dq  = np.zeros(self.num_dofs)
        cmd_tau = np.zeros(self.num_dofs)

        robot_state_data = self.state_processor.robot_state_data

        if self.upper_body_controller:
            upper_body_qpos, _ = self.upper_body_controller.get_q_tau(
                self.waypoints_left[0],
                self.waypoints_right[0],
                self.EE_efrc_L,
                self.EE_efrc_R,
            )
            arm_reduced_joint_indices = [0, 1, 2, 3, 7, 8, 9, 10]
            for i, idx in enumerate(arm_reduced_joint_indices):
                self.ref_upper_dof_pos[0, idx] = upper_body_qpos[i]
            for idx in [19, 20, 21, 26, 27, 28]:
                self.ref_upper_dof_pos[0, idx - 15] = 0.0

        if self.motion_player is not None and self.motion_player.enabled:
            self.ref_upper_dof_pos[:] = self.motion_player.step()
        elif self._upper_blend_count > 0:
            t = 1.0 - self._upper_blend_count / self._upper_blend_steps
            self.ref_upper_dof_pos[0] = (
                (1.0 - t) * self._upper_blend_src + t * self._upper_blend_target
            )
            self._upper_blend_count -= 1

        scaled_policy_action = self.rl_inference(robot_state_data)

        if self.get_ready_state:
            q_target = self.get_init_target(robot_state_data)
            self.init_count = min(self.init_count, 500)
        elif not self.use_policy_action:
            q_target = robot_state_data[:, 7 : 7 + self.num_dofs]
        else:
            q_target = scaled_policy_action + self.default_dof_angles

        if self.motor_pos_lower_limit_list and self.motor_pos_upper_limit_list:
            q_target[0] = np.clip(q_target[0], self.motor_pos_lower_limit_list, self.motor_pos_upper_limit_list)

        cmd_q = q_target[0]
        self.command_sender.send_command(cmd_q, cmd_dq, cmd_tau, robot_state_data[0, 7 : 7 + self.num_dofs])

    def update_waypoints(self):
        self.waypoints_left = [
            pin.SE3(self.EE_left_R.astype(np.float64), np.array([self.EE_left_x, self.EE_left_y, self.EE_left_z]))
        ]
        self.waypoints_right = [
            pin.SE3(self.EE_right_R.astype(np.float64), np.array([self.EE_right_x, self.EE_right_y, self.EE_right_z]))
        ]

    def handle_keyboard_button(self, keycode):
        super().handle_keyboard_button(keycode)
        if keycode == ",":
            self.waist_dofs_command[:, 0] -= 0.2
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode == ".":
            self.waist_dofs_command[:, 0] += 0.2
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif keycode == "m":
            self.lin_vel_command[:, 0] = 0.5
            self.stand_command[0, 0] = 1
            self.logger.info(colored(f"lin_vel_command: {self.lin_vel_command}", "green"))
        elif keycode in ["1", "2"]:
            self._handle_base_height_control(keycode)
        elif keycode == "t":
            self._handle_motion_player_toggle()
        elif keycode == "f":
            self._handle_motion_player_speed(+self._motion_player_fps_scale_step)
        elif keycode == "g":
            self._handle_motion_player_speed(-self._motion_player_fps_scale_step)
        elif keycode == "n":
            self._handle_motion_player_next()

    def _handle_start_policy(self):
        """Toggle policy on; second press → stand in place."""
        if self.use_policy_action:
            self.lin_vel_command[0, 0] = 0.0
            self.lin_vel_command[0, 1] = 0.0
            self.ang_vel_command[0, 0] = 0.0
            self.stand_command[0, 0] = 0
            self.logger.info(colored("Stand in place (velocity zeroed)", "blue"))
        else:
            super()._handle_start_policy()

    def handle_joystick_button(self, cur_key):
        super().handle_joystick_button(cur_key)
        if cur_key in ["B+up", "B+down"]:
            self._handle_joystick_base_height_control(cur_key)
        elif cur_key == "Y+up":
            self.waist_dofs_command[:, 2] -= 0.1
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "Y+down":
            self.waist_dofs_command[:, 2] += 0.1
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "select+left":
            self.waist_dofs_command[:, 0] -= 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+right":
            self.waist_dofs_command[:, 0] += 0.1
            self.logger.info(colored(f"waist yaw: {self.waist_dofs_command[:, 0]}", "green"))
        elif cur_key == "select+up":
            self.waist_dofs_command[:, 2] -= 0.05
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "select+down":
            self.waist_dofs_command[:, 2] += 0.05
            self.logger.info(colored(f"waist pitch: {self.waist_dofs_command[:, 2]}", "green"))
        elif cur_key == "A+B":
            self.command_sender.kp_level = 1.0
            self.logger.info(colored(f"kp level reset: {self.command_sender.kp_level}", "green"))
        elif cur_key == "L1":
            self._handle_motion_player_toggle()
        elif cur_key == "R1+up":
            self._handle_motion_player_next()
        elif self.config.get("use_upper_body_controller", False):
            self._handle_joystick_upper_body(cur_key)

    def _handle_joystick_upper_body(self, cur_key):
        if cur_key == "R1+up":
            self.EE_left_x += 0.05; self.EE_right_x += 0.05
        elif cur_key == "R1+down":
            self.EE_left_x -= 0.05; self.EE_right_x -= 0.05
        elif cur_key == "R1+left":
            self.EE_left_y += 0.02; self.EE_right_y -= 0.02
        elif cur_key == "R1+right":
            self.EE_left_y -= 0.02; self.EE_right_y += 0.02
        elif cur_key == "X+up":
            self.EE_left_z += 0.05; self.EE_right_z += 0.05
        elif cur_key == "X+down":
            self.EE_left_z -= 0.05; self.EE_right_z -= 0.05
        elif cur_key in ["X+left", "X+right"]:
            self.degrees += -5 if cur_key == "X+left" else 5
            self.theta = np.radians(self.degrees)
            self.EE_left_R = np.array(
                [[np.cos(-self.theta), -np.sin(-self.theta), 0],
                 [np.sin(-self.theta),  np.cos(-self.theta), 0],
                 [0, 0, 1]]
            )
            self.EE_right_R = np.array(
                [[np.cos(self.theta), -np.sin(self.theta), 0],
                 [np.sin(self.theta),  np.cos(self.theta), 0],
                 [0, 0, 1]]
            )
        else:
            return
        self.update_waypoints()

    def _handle_base_height_control(self, keycode):
        if keycode == "1":
            self.base_height_command[0, 0] += 0.1
        elif keycode == "2":
            self.base_height_command[0, 0] -= 0.1

    def _handle_joystick_base_height_control(self, cur_key):
        if cur_key == "B+up":
            self.base_height_command[0, 0] += 0.1
        elif cur_key == "B+down":
            self.base_height_command[0, 0] -= 0.1

    def _init_motion_player(self):
        import traceback
        try:
            self.motion_player = MotionPlayer.from_config(self.config)
            self.motion_player.enabled = False
            self.logger.info(colored(
                f"MotionPlayer initialised (inactive): {self.motion_player}", "cyan"
            ))
        except Exception as e:
            self.logger.error(f"MotionPlayer init failed: {e}\n{traceback.format_exc()}")
            self.motion_player = None

    def _handle_motion_player_toggle(self):
        if self.motion_player is None:
            self.logger.warning("MotionPlayer not initialised — check use_motion_player and motion_player_pkl in config")
            return
        if not self.motion_player.enabled:
            self.motion_player.reset(
                motion_idx=0, random_start=False, _blend=True,
                init_pose=self.ref_upper_dof_pos[0],
            )
            self.motion_player.enabled = True
            self._upper_blend_count = 0
            self.logger.info(colored(
                f"MotionPlayer ON  motion='{self.motion_player.current_motion_name}'  "
                f"fps_scale={self.motion_player.fps_scale:.2f}", "cyan"
            ))
        else:
            self.motion_player.enabled = False
            self._upper_blend_src = self.ref_upper_dof_pos[0].copy()
            self._upper_blend_target = self.default_dof_angles[self.upper_dof_indices].copy()
            self._upper_blend_count = self._upper_blend_steps
            self.logger.info(colored("MotionPlayer OFF — easing back to default upper pose", "cyan"))

    def _handle_motion_player_speed(self, delta):
        if self.motion_player is None:
            return
        self.motion_player.adjust_fps_scale(delta)
        self.logger.info(colored(
            f"MotionPlayer fps_scale={self.motion_player.fps_scale:.2f}  "
            f"(motion speed {self.motion_player.fps_scale:.1f}×)", "cyan"
        ))

    def _handle_motion_player_next(self):
        if self.motion_player is None:
            return
        next_idx = (self.motion_player._motion_idx + 1) % self.motion_player._n_motions
        self.motion_player.reset(motion_idx=next_idx, random_start=False)
        self.logger.info(colored(
            f"MotionPlayer → motion #{self.motion_player._motion_idx}  "
            f"'{self.motion_player.current_motion_name}'", "cyan"
        ))

    def _print_control_status(self):
        super()._print_control_status()
        print(f"Base height command: {self.base_height_command}")
        print(f"Waist dofs command: {self.waist_dofs_command}")
        if self._use_balance_descriptor:
            print(f"BD obs: {self._last_bd_obs}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot")
    parser.add_argument("--config", type=str, default="config/g1/g1_29dof_falcon.yaml", help="config file")
    parser.add_argument("--model_path", type=str, help="path to the ONNX model file")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    model_path = args.model_path if args.model_path else config.get("model_path")
    if not model_path:
        raise ValueError("model_path must be provided either via --model_path argument or in config file")

    policy = LocoManipPolicy(
        config=config, model_path=model_path, rl_rate=50, policy_action_scale=0.25
    )
    policy.run()
