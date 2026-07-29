# Publishes dual-arm IK solutions (14 dof: left arm 7 + right arm 7) to the
# FALCON sim2real policy process over DDS, so the policy can use them as
# ref_upper_dof_pos instead of the pkl-based MotionPlayer.
#
# NOTE: while this is used, the local arm_ctrl (robot_arm.py) must NOT
# publish to rt/lowcmd / rt/arm_sdk — only the sim2real policy process may
# write low-level commands. See FALCON_experiment/sim2real/README.md.

import json

from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.idl.std_msgs.msg.dds_ import String_

import logging_mp
logger_mp = logging_mp.getLogger(__name__)


class TeleopUpperRefPublisher:
    def __init__(self, topic="rt/teleop_upper_ref"):
        self._publisher = ChannelPublisher(topic, String_)
        self._publisher.Init()
        logger_mp.info(f"[TeleopUpperRefPublisher] publishing on {topic}")

    def publish(self, sol_q):
        msg = String_(data=json.dumps({"q": [float(x) for x in sol_q]}))
        self._publisher.Write(msg)
