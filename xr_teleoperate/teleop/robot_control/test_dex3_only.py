"""Minimal Dex3 finger teleop test with first-person view.

Drives rt/dex3/{left,right}/cmd from Quest hand-tracking data via
Dex3_1_Controller. Uses immersive display mode + the sim_env head_cam
shared-memory feed (see sim2real/utils/head_cam_shm.py) instead of
teleimager, since pass-through mode was unreliable on this headset/browser
and this repo's sim_env doesn't run teleimager's image_server.

Requires sim_env to be running with --head_cam.
"""
import argparse
import sys
import time
from multiprocessing import Array, Lock

from televuer import TeleVuerWrapper
from unitree_sdk2py.core.channel import ChannelFactoryInitialize

from robot_hand_unitree import Dex3_1_Controller

import logging_mp
logger_mp = logging_mp.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cert_file", type=str, required=True)
    parser.add_argument("--key_file", type=str, required=True)
    parser.add_argument("--domain_id", type=int, default=0, help="0 for real robot, 1 for simulation isolation")
    parser.add_argument("--interface", type=str, default="lo", help="sim2sim: lo")
    parser.add_argument("--sim2real_dir", type=str,
                         default="/home/jh/Humanoid/FALCON_BD_real/FALCON_experiment/dex3/sim2real",
                         help="Path to sim2real/ (for head_cam_shm.py) matching the running sim_env.")
    args = parser.parse_args()

    sys.path.append(args.sim2real_dir)
    from utils.head_cam_shm import HeadCamSubscriber

    ChannelFactoryInitialize(args.domain_id, args.interface)

    head_cam = HeadCamSubscriber()
    logger_mp.info("Connected to head_cam shared memory, waiting for first frame...")
    first_frame = None
    deadline = time.time() + 5.0
    while first_frame is None and time.time() < deadline:
        first_frame = head_cam.read()
        time.sleep(0.05)
    if first_frame is None:
        raise TimeoutError("No head_cam frame published yet (is sim_env's viewer running/synced?)")
    img_shape = first_frame.shape[:2]
    logger_mp.info(f"head_cam frame shape: {img_shape}")

    tv_wrapper = TeleVuerWrapper(
        use_hand_tracking=True,
        binocular=False,
        img_shape=img_shape,
        display_mode="immersive",
        zmq=True,
        cert_file=args.cert_file,
        key_file=args.key_file,
        return_hand_rot_data=False,
    )

    left_hand_pos_array = Array('d', 75, lock=True)
    right_hand_pos_array = Array('d', 75, lock=True)
    dual_hand_data_lock = Lock()
    dual_hand_state_array = Array('d', 14, lock=False)
    dual_hand_action_array = Array('d', 14, lock=False)
    hand_ctrl = Dex3_1_Controller(
        left_hand_pos_array, right_hand_pos_array, dual_hand_data_lock,
        dual_hand_state_array, dual_hand_action_array,
    )

    logger_mp.info("Dex3-only test running. Open the Quest browser URL, allow hand tracking, and move your hands.")
    last_log = 0.0
    try:
        while True:
            frame = head_cam.read()
            # televuer's render thread does cv2.cvtColor(latest_frame, ...)
            # with no shape/emptiness check of its own and dies (silently,
            # leaving the headset stuck on the last frame) if handed
            # anything but a full HxWx3 array — so validate defensively
            # rather than trusting head_cam.read()'s None-on-torn-read
            # contract alone.
            if frame is not None and frame.size > 0 and frame.ndim == 3:
                # render_to_xr() internally does BGR2RGB; feed it the mirror
                # image so it comes out correct on the headset (verified in
                # the earlier arm-teleop head_cam work). copy() so we never
                # hand the renderer a view into shared memory that the
                # publisher could overwrite mid-render.
                tv_wrapper.render_to_xr(frame[:, :, ::-1].copy())

            tele_data = tv_wrapper.get_tele_data()
            now = time.time()
            if now - last_log > 1.0:
                logger_mp.info(f"motion_data_ready={tele_data.motion_data_ready}")
                last_log = now
            if tele_data.motion_data_ready:
                with left_hand_pos_array.get_lock():
                    left_hand_pos_array[:] = tele_data.left_hand_pos.flatten()
                with right_hand_pos_array.get_lock():
                    right_hand_pos_array[:] = tele_data.right_hand_pos.flatten()
            time.sleep(0.01)
    except KeyboardInterrupt:
        pass
