"""Quest 2 <-> robot head-camera vision-only preview. No robot control at all.

Unlike rl_policy/loco_manip/loco_manip_xr.py, this script never imports
LocoManipPolicy / DDS / unitree_sdk2py, so it can never send a single command
to the real robot's motors. It only does two things:

    1. TeleVuerWrapper: opens the same HTTPS/vuer server Quest 2 connects to
       (immersive display mode, zmq image transport).
    2. RobotHeadCamSource (or SimHeadCamSource): pulls frames from PC2's
       teleimager image_server (or the Mujoco sim's shared memory) and
       forwards them to the headset via render_to_xr().

WebRTC is intentionally not offered here: Quest 2 (WiFi-only) has no network
path to PC2 (ethernet-only robot-internal network, no IP forwarding on the
operator PC), so only the ZMQ relay through this process actually reaches the
headset. See dex3_vision/G1_HEAD_CAM.md for the full network explanation.

Usage:
    python vision_preview_xr.py \
        --cert_file=$HOME/.config/xr_teleoperate/cert.pem \
        --key_file=$HOME/.config/xr_teleoperate/key.pem \
        --head_cam_source robot --img_server_ip 192.168.123.164
"""

import argparse
import sys
import time

sys.path.append("../")

from sim2real.utils.head_cam_source import (
    DEFAULT_IMG_SERVER_IP,
    DEFAULT_IMG_SERVER_REQUEST_PORT,
    make_head_cam_source,
)


def main():
    parser = argparse.ArgumentParser(description="Vision-only Quest 2 preview (no robot control)")
    parser.add_argument("--cert_file", type=str, required=True, help="path to SSL cert.pem")
    parser.add_argument("--key_file", type=str, required=True, help="path to SSL key.pem")
    parser.add_argument("--head_cam_source", type=str, default="sim", choices=["sim", "robot"],
                         help="'sim': Mujoco head camera (sim_env must run with --head_cam). "
                              "'robot': the real G1's head camera, served by teleimager's "
                              "image_server on PC2.")
    parser.add_argument("--head_cam_width", type=int, default=640,
                         help="Sim head camera width (robot resolution comes from the server).")
    parser.add_argument("--head_cam_height", type=int, default=480,
                         help="Sim head camera height (robot resolution comes from the server).")
    parser.add_argument("--img_server_ip", type=str, default=DEFAULT_IMG_SERVER_IP,
                         help="IP of the robot PC2 running teleimager's image_server.")
    parser.add_argument("--img_server_port", type=int, default=DEFAULT_IMG_SERVER_REQUEST_PORT,
                         help="teleimager camera-config request port.")
    parser.add_argument("--fps", type=float, default=30.0, help="Target loop/display rate.")
    args = parser.parse_args()

    print(f"[vision-preview] connecting to head_cam_source={args.head_cam_source} ...")
    src = make_head_cam_source(
        args.head_cam_source, width=args.head_cam_width, height=args.head_cam_height,
        img_server_ip=args.img_server_ip, img_server_port=args.img_server_port,
    )
    print(f"[vision-preview] camera connected: shape={src.img_shape} binocular={src.binocular}")

    from televuer import TeleVuerWrapper

    tv_wrapper = TeleVuerWrapper(
        use_hand_tracking=False,
        binocular=src.binocular,
        img_shape=src.img_shape,
        display_fps=30.0,
        display_mode="immersive",
        zmq=True,
        webrtc=False,
        cert_file=args.cert_file,
        key_file=args.key_file,
        return_hand_rot_data=False,
    )
    print("[vision-preview] XR bridge ready. Open the headset browser and Enter VR.")
    print("[vision-preview] No robot control code is loaded — this process cannot move the robot.")

    period = 1.0 / args.fps
    frame_count = 0
    try:
        while True:
            t0 = time.time()
            frame_bgr = src.read_bgr()
            if frame_bgr is not None:
                tv_wrapper.render_to_xr(frame_bgr)
                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"[vision-preview] forwarded {frame_count} frames")
            dt = time.time() - t0
            if dt < period:
                time.sleep(period - dt)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            tv_wrapper.close()
        except Exception:
            pass
        try:
            src.close()
        except Exception:
            pass
        print("[vision-preview] closed.")


if __name__ == "__main__":
    main()
