import sys
import time
sys.path.append("../")
sys.path.append("./")
import yaml
import numpy as np
from sim2real.rl_policy.loco_manip.loco_manip import LocoManipPolicy

with open("config/g1/g1_finalcheck.yaml") as f:
    config = yaml.safe_load(f)

policy = LocoManipPolicy(config=config, model_path="models/falcon/g1_29dof.onnx", rl_rate=50, policy_action_scale=0.25)

while policy.state_processor.robot_state_data is None:
    time.sleep(0.02)

policy._handle_init_state()
for step in range(150):
    policy.policy_action()
    policy.rate.sleep()

policy._handle_start_policy()

def quat_to_upright(q):
    w,x,y,z = q
    return 1 - 2*(x*x + y*y)

for step in range(500):
    policy.policy_action()
    policy.rate.sleep()
    if step % 20 == 0:
        rsd = policy.state_processor.robot_state_data
        upright = quat_to_upright(rsd[0,3:7])
        qvel = rsd[0, 7+29+6:7+29+6+29]
        print(f"step={step:4d} upright={upright:.3f} |qvel|norm={np.linalg.norm(qvel):.3f}" + ("  <<< FALLING" if upright < 0.7 else ""))
