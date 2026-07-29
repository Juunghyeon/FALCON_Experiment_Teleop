"""Head-camera frame sources for the XR bridge.

Two interchangeable sources feed the same consumer (loco_manip_xr.py ->
televuer render_to_xr):

    sim   - Mujoco offscreen render, handed over through POSIX shared memory
            by sim_env (see head_cam_shm.py).
    robot - the real G1's head camera, served by teleimager's image_server
            running on the robot's PC2 and pulled over ZMQ by ImageClient.

Both expose the same API:

    source.img_shape   (height, width) of a full frame as televuer wants it
                       (for a binocular camera, width spans BOTH eyes)
    source.binocular   whether the frame is a side-by-side stereo pair
    source.read_bgr()  latest frame as HxWx3 uint8 BGR, or None if nothing
                       new is available yet
    source.close()

BGR is the common currency because televuer's render_to_xr() unconditionally
applies COLOR_BGR2RGB before display: whatever we hand it is interpreted as
BGR, so a genuinely-RGB source (the sim) has to flip on the way out.

The robot source additionally reports whether the image server offers a WebRTC
stream. If it does, televuer can pull the video directly from the server and
render_to_xr()/read_bgr() are never used — that path skips the
JPEG-decode-and-re-encode round trip through this process entirely.
"""

import time

import numpy as np

DEFAULT_IMG_SERVER_IP = "192.168.123.164"
DEFAULT_IMG_SERVER_REQUEST_PORT = 60000


class SimHeadCamSource:
    """Mujoco head camera published into shared memory by sim_env --head_cam."""

    binocular = False
    webrtc_enabled = False
    webrtc_url = None

    def __init__(self, width=640, height=480, connect_timeout=10.0):
        from sim2real.utils.head_cam_shm import HeadCamSubscriber

        self.img_shape = (height, width)
        self._sub = HeadCamSubscriber(connect_timeout=connect_timeout)

    def read_bgr(self):
        frame_rgb = self._sub.read()
        if frame_rgb is None:
            return None
        return frame_rgb[:, :, ::-1]

    def close(self):
        self._sub.close()


class RobotHeadCamSource:
    """Real G1 head camera, served by teleimager's image_server on PC2.

    The frame geometry (resolution, mono vs. binocular) is not configured here:
    it is whatever cam_config_server.yaml on the robot says, fetched from the
    server at connect time so the headset display is always set up to match.
    """

    def __init__(self, host=DEFAULT_IMG_SERVER_IP, request_port=DEFAULT_IMG_SERVER_REQUEST_PORT):
        try:
            from teleimager.image_client import ImageClient
        except ImportError as e:
            raise ImportError(
                f"{e}\nInstall the teleimager client into this env:\n"
                "  cd xr_teleoperate/teleop/teleimager && pip install -e . --no-deps"
            ) from e

        self._client = ImageClient(host=host, request_port=request_port, request_bgr=True)
        cam_config = self._client.get_cam_config()
        head = cam_config["head_camera"]

        if not head.get("enable_zmq") and not head.get("enable_webrtc"):
            raise RuntimeError(
                f"head_camera on the image server at {host} has neither ZMQ nor WebRTC "
                "enabled — set enable_zmq: true in cam_config_server.yaml on PC2."
            )

        self.img_shape = tuple(head["image_shape"])  # (height, width), width spans both eyes
        self.binocular = bool(head.get("binocular", False))
        self.fps = head.get("fps", 30)
        self.zmq_enabled = bool(head.get("enable_zmq"))
        self.webrtc_enabled = bool(head.get("enable_webrtc"))
        self.webrtc_url = (
            f"https://{host}:{head['webrtc_port']}/offer" if self.webrtc_enabled else None
        )
        self._last_frame_id = None

    def read_bgr(self):
        if not self.zmq_enabled:
            return None
        frame = self._client.get_head_frame()
        bgr = frame.bgr if frame else None
        if bgr is None:
            return None
        # The subscriber hands back the same decoded array until a new frame
        # arrives; the policy loop (50 Hz) outruns the camera (30 Hz), so skip
        # the ones we have already pushed to the headset.
        if id(bgr) == self._last_frame_id:
            return None
        self._last_frame_id = id(bgr)
        return np.ascontiguousarray(bgr)

    def wait_for_frame(self, timeout=2.0):
        """True once a frame has actually arrived over ZMQ.

        Worth checking at startup: when the image server is unreachable,
        teleimager's requester quietly falls back to a cached/bundled
        cam_config yaml, so construction succeeds and only the absence of
        frames reveals that nothing is really connected.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.read_bgr() is not None:
                return True
            time.sleep(0.05)
        return False

    def close(self):
        self._client.close()


def make_head_cam_source(source, width=640, height=480,
                         img_server_ip=DEFAULT_IMG_SERVER_IP,
                         img_server_port=DEFAULT_IMG_SERVER_REQUEST_PORT):
    if source == "sim":
        return SimHeadCamSource(width=width, height=height)
    if source == "robot":
        return RobotHeadCamSource(host=img_server_ip, request_port=img_server_port)
    raise ValueError(f"unknown head cam source '{source}' (expected 'sim' or 'robot')")


def _preview():
    """Standalone check: python -m sim2real.utils.head_cam_source --source robot

    Opens the camera stream and shows it in an OpenCV window, so the camera
    link can be verified on its own before bringing up the policy and headset.
    """
    import argparse

    import cv2

    parser = argparse.ArgumentParser(description="Preview the head camera stream")
    parser.add_argument("--source", default="robot", choices=["sim", "robot"])
    parser.add_argument("--img_server_ip", default=DEFAULT_IMG_SERVER_IP)
    parser.add_argument("--img_server_port", type=int, default=DEFAULT_IMG_SERVER_REQUEST_PORT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    src = make_head_cam_source(args.source, width=args.width, height=args.height,
                               img_server_ip=args.img_server_ip,
                               img_server_port=args.img_server_port)
    print(f"connected: shape={src.img_shape} binocular={src.binocular} "
          f"webrtc={src.webrtc_enabled} url={src.webrtc_url}")
    try:
        while True:
            frame = src.read_bgr()
            if frame is not None:
                cv2.imshow("head camera (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        src.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    _preview()
