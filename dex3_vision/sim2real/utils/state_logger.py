"""
Thread-safe state logger for real-robot deployment (NPZ output).

Ported from the sim2sim StateLogger (FALCON training repo,
sim2real/utils/state_logger.py). The sim version reads MuJoCo mj_model /
mj_data directly (ground-truth ZMP, CoM, contact forces from the physics
engine). On real hardware there is no such ground truth, so this logger
instead reads:
  - robot_state_data straight from the policy's state_processor (IMU quat,
    joint q/qvel, base ang_vel; base_pos is not observable and stays zero)
  - the policy's own balance-descriptor MuJoCo shadow model
    (policy._bd_mj_model / policy._bd_mj_data), if BD is enabled, using the
    same cmm_balance helpers the training env uses — this is the same
    "policy-side MuJoCo" already computing _last_bd_obs, just re-read here
    for ZMP/CoM/contact bookkeeping instead of re-deriving it.
  - user commands (lin_vel/ang_vel/base_height/stand/waist) and motion
    player state directly off the policy object (single-process, no IPC
    needed unlike the two-process sim2sim setup).

Logged quantities per timestep:
    time            float    wall-clock time since logging start (s)
    base_quat       (4,)     IMU orientation quaternion (w, x, y, z)
    q               (nq,)    joint angles
    dq              (nq,)    joint velocities
    base_ang_vel    (3,)     base angular velocity (rad/s), from IMU gyro
    action          (nq,)    last scaled policy action (post-clip, pre-default-offset)
    com_pos         (3,)     whole-body CoM position (BD shadow model; zeros if BD disabled)
    p_zmp           (2,)     ZMP position from BD shadow model contacts
    l_foot_contact  int      left  foot contact state (0 or 1, BD shadow model)
    r_foot_contact  int      right foot contact state (0 or 1, BD shadow model)
    support_polygon (8, 2)   foot-corner world positions (BD shadow model)
    delta_p_D       (2,)     disturbance-induced ZMP shift (m)
    u_D             (2,)     disturbance direction unit vector
    d_dir           float    raw directional support margin from ZMP (m)
    delta_bar_D     float    normalised disturbance magnitude  ||delta_p_D|| / d_ref
    m_bar_D         float    normalised directional support margin
    policy_bd_obs   (3,)     BD obs actually fed to the policy (policy._last_bd_obs)

    --- Command fields ---
    cmd_lin_vel     (2,)     linear velocity command [vx, vy] (m/s)
    cmd_ang_vel     float    angular velocity command [yaw rate] (rad/s)
    cmd_base_height float    base height command (m)
    cmd_stand       int8     stand mode flag (0=walk, 1=stand)
    cmd_waist       (3,)     waist joint command [yaw, roll, pitch] (rad)
    motion_idx      int16    index of currently playing motion (-1 = none/disabled)
                             name lookup: meta['motion_names'][motion_idx]
"""

import json
import threading
import time
import numpy as np
from datetime import datetime
from pathlib import Path

# Number of support-polygon points (4 corners x 2 feet), matches cmm_balance layout.
_N_POLY_PTS = 8


class StateLogger:
    """
    Usage:
        logger = StateLogger(log_dir='logs', prefix='real_log')
        logger.wire_policy(policy)
        logger.start()
        # each control step, after policy.policy_action():
        logger.log()
        logger.end()   # writes compressed NPZ
    """

    def __init__(self, log_dir='logs', prefix='real_log', d_ref=0.1):
        self.log_dir  = Path(log_dir)
        self.prefix   = prefix
        self.lock     = threading.Lock()
        self.active   = False
        self._policy  = None
        self.start_time = None
        self._out_path  = None

        self.d_ref = d_ref

        # BD shadow-model foot subtree ids (built lazily once policy is wired and
        # its _bd_mj_model exists; mirrors policy._bd_left_ids/_bd_right_ids).
        self._foot_left_ids  = None
        self._foot_right_ids = None

        # Data buffers
        self._times          = []
        self._base_quat      = []
        self._q              = []
        self._dq             = []
        self._base_ang_vel   = []
        self._action         = []
        self._com_pos        = []
        self._p_zmp          = []
        self._l_foot_contact = []
        self._r_foot_contact = []
        self._support_polygon = []
        self._delta_p_D      = []
        self._u_D            = []
        self._d_dir          = []
        self._delta_bar_D    = []
        self._m_bar_D        = []
        self._policy_bd_obs  = []

        # Command logging
        self._cmd_lin_vel     = []
        self._cmd_ang_vel     = []
        self._cmd_base_height = []
        self._cmd_stand       = []
        self._cmd_waist       = []

        # Motion player logging
        self._motion_idx      = []
        self._motion_names    = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_dir(self):
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def _reset_buffers(self):
        self._times.clear()
        self._base_quat.clear()
        self._q.clear()
        self._dq.clear()
        self._base_ang_vel.clear()
        self._action.clear()
        self._com_pos.clear()
        self._p_zmp.clear()
        self._l_foot_contact.clear()
        self._r_foot_contact.clear()
        self._support_polygon.clear()
        self._delta_p_D.clear()
        self._u_D.clear()
        self._d_dir.clear()
        self._delta_bar_D.clear()
        self._m_bar_D.clear()
        self._policy_bd_obs.clear()
        self._cmd_lin_vel.clear()
        self._cmd_ang_vel.clear()
        self._cmd_base_height.clear()
        self._cmd_stand.clear()
        self._cmd_waist.clear()
        self._motion_idx.clear()
        self._motion_names.clear()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def wire_policy(self, policy) -> None:
        """Wire the policy instance so log() can read robot_state_data, commands,
        motion player state, and (if enabled) the BD shadow MuJoCo model.

        Call once after the policy is initialised:
            logger.wire_policy(policy)
        """
        self._policy = policy

    def start(self):
        """Open a new log file and begin recording."""
        with self.lock:
            if self.active:
                return
            self._ensure_dir()
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._out_path  = str(self.log_dir / f"{self.prefix}_{ts}.npz")
            self.start_time = time.time()
            self.active     = True
            self._reset_buffers()

    def log(self):
        """Record one timestep of state / BD / command data from the wired policy."""
        if not self.active:
            return

        policy = self._policy
        if policy is None:
            return

        t_now = time.time() - (self.start_time or time.time())

        # --- robot state (from state_processor, IMU-based; base_pos not observable) ---
        base_quat    = np.array([1., 0., 0., 0.])
        q            = np.zeros(0)
        dq           = np.zeros(0)
        base_ang_vel = np.zeros(3)
        try:
            rsd = policy.state_processor.robot_state_data
            if rsd is not None:
                N  = policy.num_dofs
                r0 = rsd[0]
                base_quat    = r0[3:7].copy()
                q            = r0[7:7 + N].copy()
                base_ang_vel = r0[7 + N + 3: 7 + N + 6].copy()
                dq           = r0[7 + N + 6: 7 + 2 * N + 6].copy()
        except Exception:
            pass

        # --- last policy action ---
        action = np.zeros(0)
        try:
            action = np.asarray(policy.last_policy_action).flatten().copy()
        except Exception:
            pass

        # --- BD shadow-model quantities (ZMP/CoM/contacts/support polygon) ---
        com          = np.zeros(3)
        p_zmp        = np.zeros(2)
        l_contact    = 0
        r_contact    = 0
        support_pts  = np.zeros((_N_POLY_PTS, 2))
        delta_p_D    = np.zeros(2)
        u_D          = np.zeros(2)
        d_dir        = 0.0
        delta_bar_D  = 0.0
        m_bar_D      = 0.0
        try:
            model = getattr(policy, '_bd_mj_model', None)
            data  = getattr(policy, '_bd_mj_data', None)
            if model is not None and data is not None:
                import mujoco
                from sim2real.utils import cmm_balance

                mujoco.mj_subtreeVel(model, data)
                com = data.subtree_com[1].copy()

                if self._foot_left_ids is None:
                    self._foot_left_ids  = cmm_balance._subtree_body_ids(model, cmm_balance.FOOT_ROOT_LEFT)
                    self._foot_right_ids = cmm_balance._subtree_body_ids(model, cmm_balance.FOOT_ROOT_RIGHT)
                l_contact, r_contact = cmm_balance.compute_foot_contacts(
                    model, data, left_ids=self._foot_left_ids, right_ids=self._foot_right_ids)

                pts = cmm_balance.compute_support_polygon(model, data)
                if pts is not None and pts.shape == (_N_POLY_PTS, 2):
                    support_pts = pts

                pz, _fz = cmm_balance.compute_zmp_from_contacts(model, data)
                if pz is not None:
                    p_zmp = pz
                else:
                    p_zmp = com[:2].copy()

                # Reuse the BD state the policy already computed this step
                # (avoids re-deriving finite-difference momentum rate here).
                delta_p_D = np.asarray(getattr(policy, '_bd_delta_p_D_ds', np.zeros(2)), dtype=float)
                dp_mag = float(np.linalg.norm(delta_p_D))
                u_D = (delta_p_D / dp_mag) if dp_mag > 1e-6 else np.zeros(2)
                d_dir = cmm_balance.compute_directional_margin(p_zmp, pts, u_D) if pts is not None else 0.0
                d_ref = getattr(policy, '_bd_d_ref', self.d_ref)
                delta_bar_D = dp_mag / max(d_ref, 1e-6)
                m_bar_D = d_dir / max(d_ref, 1e-6)
        except Exception:
            pass

        # --- policy BD obs actually fed to the network ---
        policy_bd_obs = np.zeros(3, dtype=np.float32)
        try:
            obs = getattr(policy, '_last_bd_obs', None)
            if obs is not None:
                policy_bd_obs = np.asarray(obs, dtype=np.float32).flatten()[:3]
        except Exception:
            pass

        # --- commands ---
        cmd_lin_vel     = np.zeros(2)
        cmd_ang_vel     = 0.0
        cmd_base_height = 0.0
        cmd_stand       = 0
        cmd_waist       = np.zeros(3)
        try:
            cmd_lin_vel     = np.asarray(policy.lin_vel_command[0], dtype=float)
            cmd_ang_vel     = float(policy.ang_vel_command[0, 0])
            cmd_base_height = float(policy.base_height_command[0, 0])
            cmd_stand       = int(policy.stand_command[0, 0])
            cmd_waist       = np.asarray(policy.waist_dofs_command[0], dtype=float)
        except Exception:
            pass

        # --- motion player ---
        cur_motion_idx = -1
        try:
            mp = getattr(policy, 'motion_player', None)
            if mp is not None and mp.enabled:
                name = mp.current_motion_name
                if name:
                    if name not in self._motion_names:
                        self._motion_names.append(name)
                    cur_motion_idx = self._motion_names.index(name)
        except Exception:
            pass

        # --- append to buffers ---
        with self.lock:
            self._times.append(t_now)
            self._base_quat.append(base_quat)
            self._q.append(q)
            self._dq.append(dq)
            self._base_ang_vel.append(base_ang_vel)
            self._action.append(action)
            self._com_pos.append(com)
            self._p_zmp.append(p_zmp)
            self._l_foot_contact.append(l_contact)
            self._r_foot_contact.append(r_contact)
            self._support_polygon.append(support_pts)
            self._delta_p_D.append(delta_p_D)
            self._u_D.append(u_D)
            self._d_dir.append(d_dir)
            self._delta_bar_D.append(delta_bar_D)
            self._m_bar_D.append(m_bar_D)
            self._policy_bd_obs.append(policy_bd_obs)
            self._cmd_lin_vel.append(cmd_lin_vel)
            self._cmd_ang_vel.append(cmd_ang_vel)
            self._cmd_base_height.append(cmd_base_height)
            self._cmd_stand.append(cmd_stand)
            self._cmd_waist.append(cmd_waist)
            self._motion_idx.append(cur_motion_idx)

    def end(self):
        """Stop logging and save the NPZ file."""
        with self.lock:
            if not self.active:
                return
            try:
                def _vstack(lst, w):
                    return np.vstack(lst) if lst and lst[0].size > 0 else np.zeros((0, w))

                meta = {
                    'started_at': (datetime.fromtimestamp(self.start_time).isoformat()
                                   if self.start_time else None),
                    'ended_at': datetime.now().isoformat(),
                    'motion_names': list(self._motion_names),
                }
                np.savez_compressed(
                    self._out_path,
                    time            = np.array(self._times),
                    base_quat       = _vstack(self._base_quat, 4),
                    q               = _vstack(self._q, 0),
                    dq              = _vstack(self._dq, 0),
                    base_ang_vel    = _vstack(self._base_ang_vel, 3),
                    action          = _vstack(self._action, 0),
                    com_pos         = _vstack(self._com_pos, 3),
                    p_zmp           = _vstack(self._p_zmp, 2),
                    l_foot_contact  = np.array(self._l_foot_contact, dtype=np.int8),
                    r_foot_contact  = np.array(self._r_foot_contact, dtype=np.int8),
                    support_polygon = np.stack(self._support_polygon) if self._support_polygon else np.zeros((0, _N_POLY_PTS, 2)),
                    delta_p_D       = _vstack(self._delta_p_D, 2),
                    u_D             = _vstack(self._u_D, 2),
                    d_dir           = np.array(self._d_dir),
                    delta_bar_D     = np.array(self._delta_bar_D),
                    m_bar_D         = np.array(self._m_bar_D),
                    policy_bd_obs   = _vstack(self._policy_bd_obs, 3),
                    cmd_lin_vel     = _vstack(self._cmd_lin_vel, 2),
                    cmd_ang_vel     = np.array(self._cmd_ang_vel, dtype=np.float32),
                    cmd_base_height = np.array(self._cmd_base_height, dtype=np.float32),
                    cmd_stand       = np.array(self._cmd_stand, dtype=np.int8),
                    cmd_waist       = _vstack(self._cmd_waist, 3),
                    motion_idx      = np.array(self._motion_idx, dtype=np.int16),
                    meta            = json.dumps(meta),
                )
                print(f"StateLogger: saved {len(self._times)} steps -> {self._out_path}")
            except Exception as e:
                print(f"StateLogger.end() failed: {e}")

            self._reset_buffers()
            self._out_path  = None
            self.active     = False
            self.start_time = None

    def is_active(self):
        return self.active


__all__ = ['StateLogger']
