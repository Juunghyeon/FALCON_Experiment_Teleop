# Quest 2 connectivity smoke test — no robot / camera server required.
#
# Verifies only that: the HTTPS/WebXR page comes up, Quest 2 can connect,
# and controller (or hand) tracking data streams back to this process.
# Uses display_mode="pass-through" so no image source (zmq/webrtc) is needed.
#
# Usage:
#   cd xr_teleoperate/teleop/televuer/example   (or anywhere on PYTHONPATH with televuer installed)
#   python test_quest2_connection.py --input-mode controller   # start here, hand tracking second
#   python test_quest2_connection.py --input-mode hand

import argparse
import time

from televuer import TeleVuerWrapper
import logging_mp
logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-mode", choices=["hand", "controller"], default="controller")
    args = parser.parse_args()

    use_hand_track = args.input_mode == "hand"

    tv_wrapper = TeleVuerWrapper(
        use_hand_tracking=use_hand_track,
        binocular=True,
        img_shape=(480, 1280),
        display_mode="pass-through",   # no robot camera image needed
        zmq=False,
        webrtc=False,
    )

    print("\n" + "=" * 70)
    print("Quest 2 should now be on the same Wi-Fi as this PC.")
    print("On Quest 2 (inside the headset), open the browser and go to:")
    print("  https://<this-PC-IP>:8012/?ws=wss://<this-PC-IP>:8012")
    print("Accept the self-signed certificate warning, then click")
    print('the "pass-through" / "Enter VR" button in the bottom-left corner.')
    print("=" * 70 + "\n")

    input("Press Enter here once you've entered VR on the Quest 2...")

    try:
        while True:
            start_time = time.time()
            tele_data = tv_wrapper.get_tele_data()

            print("-" * 60)
            print(f"head_pose:\n{tele_data.head_pose}")
            print(f"left_wrist_pose:\n{tele_data.left_wrist_pose}")
            print(f"right_wrist_pose:\n{tele_data.right_wrist_pose}")
            print(f"motion_data_ready: {tele_data.motion_data_ready}")

            if use_hand_track:
                print(f"left_hand_pos shape: {tele_data.left_hand_pos.shape}")
                print(f"right_hand_pos shape: {tele_data.right_hand_pos.shape}")
                print(f"left_pinch: {tele_data.left_hand_pinchValue:.2f}  right_pinch: {tele_data.right_hand_pinchValue:.2f}")
            else:
                print(f"left_trigger: {tele_data.left_ctrl_triggerValue:.2f}  right_trigger: {tele_data.right_ctrl_triggerValue:.2f}")
                print(f"left_thumbstick: {tele_data.left_ctrl_thumbstickValue}")
                print(f"right_thumbstick: {tele_data.right_ctrl_thumbstickValue}")
                print(f"left_A: {tele_data.left_ctrl_aButton}  left_B: {tele_data.left_ctrl_bButton}")

            elapsed = time.time() - start_time
            time.sleep(max(0, 0.2 - elapsed))  # ~5 Hz print rate, easy to read

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        tv_wrapper.close()


if __name__ == "__main__":
    main()
