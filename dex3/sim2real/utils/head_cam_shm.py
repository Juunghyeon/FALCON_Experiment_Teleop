"""Shared-memory hand-off for the Mujoco head-camera frame between the
sim_env process (renders) and the policy/XR-bridge process (reads and
forwards to the headset via televuer).

Named POSIX shared memory (multiprocessing.shared_memory) rather than a
socket/pipe: same-machine, low-latency, no serialization needed.

Block layout (name: "falcon_head_cam"):
    bytes  0.. 3   uint32  height
    bytes  4.. 7   uint32  width
    bytes  8..15   uint64  sequence number (seqlock)
    bytes 16..     uint8   RGB frame, height * width * 3 bytes

The sequence number is a seqlock: the publisher sets it odd while writing
and even once the frame is complete, so a reader can detect (and retry on)
a torn read without any OS-level locking.
"""

import struct
import time
from multiprocessing import shared_memory

import numpy as np

SHM_NAME = "falcon_head_cam"
HEADER_FMT = "<IIQ"  # height, width, seq
HEADER_SIZE = struct.calcsize(HEADER_FMT)


class HeadCamPublisher:
    def __init__(self, height, width, shm_name=SHM_NAME):
        self.height = height
        self.width = width
        self._seq = 0
        size = HEADER_SIZE + height * width * 3

        # Clean up a stale block left behind by a crashed previous run.
        try:
            stale = shared_memory.SharedMemory(name=shm_name)
            stale.close()
            stale.unlink()
        except FileNotFoundError:
            pass

        self.shm = shared_memory.SharedMemory(name=shm_name, create=True, size=size)
        self._buf = self.shm.buf

    def publish(self, frame_rgb):
        """frame_rgb: HxWx3 uint8 array, must match the height/width given at construction."""
        assert frame_rgb.shape == (self.height, self.width, 3), (
            f"frame shape {frame_rgb.shape} != expected {(self.height, self.width, 3)}"
        )
        self._seq += 1
        struct.pack_into(HEADER_FMT, self._buf, 0, self.height, self.width, self._seq)  # odd: writing
        self._buf[HEADER_SIZE:HEADER_SIZE + frame_rgb.nbytes] = frame_rgb.tobytes()
        self._seq += 1
        struct.pack_into(HEADER_FMT, self._buf, 0, self.height, self.width, self._seq)  # even: done

    def close(self):
        try:
            self.shm.close()
            self.shm.unlink()
        except Exception:
            pass


class HeadCamSubscriber:
    def __init__(self, shm_name=SHM_NAME, connect_timeout=5.0):
        deadline = time.time() + connect_timeout
        self.shm = None
        while time.time() < deadline:
            try:
                self.shm = shared_memory.SharedMemory(name=shm_name)
                break
            except FileNotFoundError:
                time.sleep(0.1)
        if self.shm is None:
            raise TimeoutError(
                f"head cam shared memory '{shm_name}' not found after {connect_timeout}s "
                "(is sim_env running with --head_cam?)"
            )
        self._buf = self.shm.buf

        # multiprocessing's resource_tracker auto-unlinks every SharedMemory
        # block a process has opened once that process exits — including
        # blocks it only read, never created. Without this, a subscriber
        # process exiting (e.g. loco_manip_xr.py shutting down, or even a
        # short-lived debug script) would delete the publisher's (sim_env's)
        # block out from under it. Only the publisher should ever unlink.
        try:
            from multiprocessing import resource_tracker
            resource_tracker.unregister(self.shm._name, "shared_memory")
        except Exception:
            pass

    def read(self):
        """Returns an HxWx3 uint8 RGB frame, or None if no frame has been published yet
        or a torn read was detected (caller should just try again next tick)."""
        height, width, seq1 = struct.unpack_from(HEADER_FMT, self._buf, 0)
        if seq1 == 0 or seq1 % 2 == 1:
            return None  # never published yet, or publisher mid-write
        frame_bytes = bytes(self._buf[HEADER_SIZE:HEADER_SIZE + height * width * 3])
        _, _, seq2 = struct.unpack_from(HEADER_FMT, self._buf, 0)
        if seq1 != seq2:
            return None  # torn read, publisher wrote during our copy
        return np.frombuffer(frame_bytes, dtype=np.uint8).reshape(height, width, 3)

    def close(self):
        try:
            self.shm.close()
        except Exception:
            pass
