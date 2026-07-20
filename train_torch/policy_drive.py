# PyTorch policy driving the env (no learning yet). Confirms a torch brain can play,
# and sets up exactly what REINFORCE needs: sampled actions + their log-probabilities.
# Run from project root:  python train_torch/policy_drive.py
import subprocess, json, os
import torch, torch.nn as nn
from torch.distributions import Categorical

N_OBS, N_ACT = 12, 12

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_OBS, 16)
        self.fc2 = nn.Linear(16, N_ACT + 1)   # 12 action logits + 1 troop-fraction logit
    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        out = self.fc2(h)
        return out[..., :N_ACT], out[..., N_ACT]

net = PolicyNet()

def choose(obs):
    x = torch.tensor(obs, dtype=torch.float32)
    action_logits, troop_logit = net(x)
    dist = Categorical(logits=action_logits)
    action = dist.sample()
    troop = torch.sigmoid(troop_logit)
    logp = dist.log_prob(action)              # kept for REINFORCE (part 3)
    return int(action.item()), float(troop.item()), logp

def tsx_cmd():
    if os.environ.get("TSX"): return os.environ["TSX"].split()
    local = os.path.join("node_modules", ".bin", "tsx.cmd" if os.name == "nt" else "tsx")
    return [local] if os.path.exists(local) else ["npx", "tsx"]
ENV = os.environ.get("ENVFILE", "src/env/env_server.ts")
proc = subprocess.Popen(tsx_cmd() + [ENV], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rpc(msg):
    proc.stdin.write(json.dumps(msg) + "\n"); proc.stdin.flush()
    return json.loads(proc.stdout.readline())

for ep in range(3):
    o = rpc({"cmd": "reset", "seed": ep})["obs"]; total = 0.0; steps = 0; done = False
    while not done and steps < 2000:
        a, tr, _ = choose(o)
        r = rpc({"cmd": "step", "action": a, "troop": tr})
        o = r["obs"]; total += r["reward"]; done = r["done"]; steps += 1
    print(f"episode {ep}: {steps} steps, reward {total:.3f}")
proc.stdin.close(); proc.terminate()
print("pytorch policy drove the env OK")
