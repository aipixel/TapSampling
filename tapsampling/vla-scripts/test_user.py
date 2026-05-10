import requests

import numpy as np
a = np.zeros((200, 200, 3), dtype=np.uint8)
b = np.zeros((200, 200, 3), dtype=np.uint8)
c = np.zeros((4, 10, 7))

payload = {
    "obs": {'rgb_obs': {'rgb_static': a.tolist(), 'rgb_gripper': b.tolist()}},
    "instruction": "move to target",
    "step": 5,
    "query_actions": c.tolist()
}
resp = requests.post("http://127.0.0.1:8002/step", json=payload, timeout=60)
print(resp.json())