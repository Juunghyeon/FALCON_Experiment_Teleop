"""
Lightweight AMASS/ACCAD motion player for sim2real evaluation.

Reads dof sequences from the training pkl file, advances playback frame-by-frame
at each control step, and returns ref_upper_dof_pos (1, N_upper) that feeds
directly into the policy observation — exactly replicating what the Isaac Gym
training environment does via MotionLibRobot.

No Isaac Gym / hydra dependencies.

Usage:
    player = MotionPlayer.from_config(config)
    player.reset()
    while True:
        ref_upper = player.step()          # (1, N_upper) float32
        policy.ref_upper_dof_pos = ref_upper
        ...
    player.set_fps_scale(2.0)              # play at 2× speed
"""

import numpy as np
from pathlib import Path


class MotionPlayer:
    """
    Frame-level motion player backed by a pkl motion file.

    Parameters
    ----------
    pkl_path : str | Path
        Path to the motion pkl file (e.g. accad_all.pkl).
        Each entry must have keys:
            'dof'  : np.ndarray  (T, N_dof) — joint angles in robot dof order
            'fps'  : int | float — native capture framerate (typically 30)
    all_dof_names : list[str]
        Full list of robot DOF names in the same order as pkl['dof'].
        Used to locate upper-body joints.
    upper_dof_names : list[str]
        Subset of all_dof_names that belong to the upper body.
    fps_scale : float
        Initial playback speed multiplier.  1.0 = original speed.
        2.0 = 2× faster (useful for fast-motion evaluation).
    control_rate : float
        Policy control frequency in Hz (default 50 Hz).
    loop : bool
        If True, advance to the next motion sequentially when current one ends.
    """

    def __init__(
        self,
        pkl_path,
        all_dof_names,
        upper_dof_names,
        fps_scale=1.0,
        control_rate=50.0,
        loop=True,
        blend_steps=50,
    ):
        import joblib

        pkl_path = Path(pkl_path)
        if not pkl_path.exists():
            raise FileNotFoundError(f"Motion pkl not found: {pkl_path}")

        data = joblib.load(str(pkl_path))
        # data is a dict {key: {'dof': ..., 'fps': ...}}
        self._motion_names = list(data.keys())
        self._motions = list(data.values())
        self._n_motions = len(self._motions)
        if self._n_motions == 0:
            raise ValueError("pkl file contains no motions")

        # upper-body DOF indices within the full dof array
        self._upper_ids = [all_dof_names.index(n) for n in upper_dof_names]
        self._n_upper = len(upper_dof_names)

        self.fps_scale = float(fps_scale)
        self._fps_scale_min = 0.1
        self._fps_scale_max = 5.0
        self._control_dt = 1.0 / float(control_rate)
        self.loop = loop

        # Cross-fade blend on motion transition (steps at control_rate)
        self._blend_steps = max(1, int(blend_steps))

        # Runtime state (initialised by reset())
        self._frames = None        # (T, N_upper) current motion
        self._native_fps = 30.0
        self._motion_len = 0.0     # duration in real seconds (unscaled)
        self._motion_time = 0.0    # elapsed real seconds
        self._motion_idx = -1
        self.enabled = False

        # Cross-fade state
        self._blend_src   = None   # (N_upper,) pose at blend start
        self._blend_count = 0      # steps remaining in blend (0 = inactive)
        self._last_ref    = None   # (N_upper,) last returned pose (for blend src)

        # Optional callback: called with (motion_idx, motion_name) on auto-advance
        self.on_motion_change = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config):
        """Construct from a sim2real YAML config dict.

        Required config keys:
            motion_player_pkl       : path to pkl file
            dof_names               : full list of DOF names
            dof_names_upper_body    : upper-body DOF names
        Optional:
            motion_player_fps_scale : float (default 1.0)
            rl_rate                 : int   (default 50)
        """
        pkl_path = config["motion_player_pkl"]
        # If relative, try resolving from the repo root (two levels up from this file)
        p = Path(pkl_path)
        if not p.is_absolute() and not p.exists():
            repo_root = Path(__file__).resolve().parents[2]  # sim2real/utils/ → repo root
            candidate = repo_root / p
            if candidate.exists():
                pkl_path = str(candidate)
        all_dof_names = config["dof_names"]
        upper_dof_names = config["dof_names_upper_body"]
        fps_scale = float(config.get("motion_player_fps_scale", 1.0))
        control_rate = float(config.get("rl_rate", 50))
        blend_steps = int(config.get("motion_player_blend_steps", 20))
        return cls(
            pkl_path=pkl_path,
            all_dof_names=all_dof_names,
            upper_dof_names=upper_dof_names,
            fps_scale=fps_scale,
            control_rate=control_rate,
            blend_steps=blend_steps,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, motion_idx=None, random_start=True, _blend=True, init_pose=None):
        """Load a motion and reset the time counter.

        Parameters
        ----------
        motion_idx : int | None
            Index into the motion list.  None = random.
        random_start : bool
            If True, start at a random position within the motion.
        _blend : bool
            If True and a previous pose exists, start a cross-fade blend
            from the last output pose to the new motion's first frame.
        init_pose : np.ndarray | None
            (N_upper,) pose to blend from when there is no previous
            playback pose yet (i.e. the very first reset). Lets the first
            motion activation ease in from the robot's current pose
            instead of snapping straight to frame 0.
        """
        if motion_idx is None:
            motion_idx = int(np.random.randint(self._n_motions))
        self._motion_idx = motion_idx

        mot = self._motions[self._motion_idx]
        dof_all = np.asarray(mot["dof"], dtype=np.float32)   # (T, N_dof)
        new_frames = dof_all[:, self._upper_ids]              # (T, N_upper)
        self._native_fps = float(mot.get("fps", 30))
        T = new_frames.shape[0]
        self._motion_len = (T - 1) / self._native_fps         # real seconds

        # Capture current output pose as blend source before switching frames.
        # A caller-supplied init_pose is the authoritative current pose (e.g.
        # loco_manip tracks ref_upper_dof_pos even while this player is off
        # and a return-to-default blend is running) so it takes priority over
        # our own possibly-stale _last_ref.
        if _blend and self._blend_steps > 0:
            if init_pose is not None:
                self._blend_src = np.asarray(init_pose, dtype=np.float32).copy()
            elif self._last_ref is not None:
                self._blend_src = self._last_ref.copy()
            elif self._frames is not None:
                self._blend_src = self._frames[0].copy()
            else:
                self._blend_src = None
            self._blend_count = self._blend_steps if self._blend_src is not None else 0
        else:
            self._blend_src   = None
            self._blend_count = 0

        self._frames = new_frames

        if random_start and self._motion_len > 0:
            self._motion_time = np.random.uniform(0.0, self._motion_len)
        else:
            self._motion_time = 0.0

    def step(self):
        """Advance one control step and return ref_upper_dof_pos.

        Returns
        -------
        np.ndarray  shape (1, N_upper), dtype float32
        """
        if self._frames is None:
            # First call after toggle ON: use deterministic start (motion 0, frame 0)
            self.reset(motion_idx=0, random_start=False, _blend=False)

        T = self._frames.shape[0]

        # frame position accounting for playback speed
        frame_f = self._motion_time * self._native_fps * self.fps_scale
        frame_f = np.clip(frame_f, 0.0, T - 1 - 1e-6)

        f0 = int(frame_f)
        f1 = min(f0 + 1, T - 1)
        alpha = frame_f - f0
        ref = (1.0 - alpha) * self._frames[f0] + alpha * self._frames[f1]

        # Cross-fade blend: linearly interpolate from src pose to new motion
        if self._blend_count > 0:
            t = 1.0 - self._blend_count / self._blend_steps   # 0→1 over blend window
            ref = (1.0 - t) * self._blend_src + t * ref
            self._blend_count -= 1

        # advance real time
        self._motion_time += self._control_dt

        # check whether the motion has ended (in effective / scaled time)
        effective_duration = self._motion_len / self.fps_scale if self.fps_scale > 0 else self._motion_len
        if self._motion_time >= effective_duration:
            if self.loop:
                next_idx = (self._motion_idx + 1) % self._n_motions
                self.reset(motion_idx=next_idx, random_start=False, _blend=True)
                if self.on_motion_change is not None:
                    try:
                        self.on_motion_change(self._motion_idx, self.current_motion_name)
                    except Exception:
                        pass
            else:
                self._motion_time = effective_duration

        self._last_ref = ref.copy()
        return ref.reshape(1, self._n_upper).astype(np.float32)

    def set_fps_scale(self, scale):
        """Set playback speed multiplier (clamped to [0.1, 5.0])."""
        self.fps_scale = float(np.clip(scale, self._fps_scale_min, self._fps_scale_max))

    def adjust_fps_scale(self, delta):
        """Increment / decrement fps_scale by delta."""
        self.set_fps_scale(self.fps_scale + delta)

    @property
    def current_motion_name(self):
        """Return the pkl key (name) of the currently loaded motion, or '' if not loaded."""
        if self._motion_idx < 0 or self._motion_idx >= len(self._motion_names):
            return ""
        return self._motion_names[self._motion_idx]

    def toggle(self):
        """Toggle motion playback on/off."""
        self.enabled = not self.enabled
        return self.enabled

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self):
        return (
            f"MotionPlayer(n_motions={self._n_motions}, "
            f"n_upper={self._n_upper}, fps_scale={self.fps_scale:.2f}, "
            f"enabled={self.enabled})"
        )
