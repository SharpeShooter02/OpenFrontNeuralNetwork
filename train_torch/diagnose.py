# Diagnostic: load the trained torch policy, play a few episodes, and print WHAT the agent
# actually does — action histogram + a decoded trajectory (land/troops/gold/cities/etc).
# Run from project root:  python train_torch/diagnose.py
import subprocess, json, os
import torch, torch.nn as nn
from torch.distributions import Categorical

N_OBS, N_ACT = 12, 12
ACTIONS = ["expandEmpty","attackWeak","attackStrong","(unused3)","reqAlliance","buildCity",
           "buildDefense","buildSilo","buildSAM","nuke","boatAttack","buildPort"]

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_OBS, 16)
        self.fc2 = nn.Linear(16, N_ACT + 1)
        self.troop_log_std = nn.Parameter(torch.tensor(-0.5))
    def forward(self, x):
        h = torch.tanh(self.fc1(x)); out = self.fc2(h)
        return out[..., :N_ACT], out[..., N_ACT]

net = PolicyNet()
net.load_state_dict(torch.load(os.path.join("data", "torch_policy.pt")))
net.eval()

def choose(obs):
    x = torch.tensor(obs, dtype=torch.float32)
    logits, troop_mean = net(x)
    a = int(Categorical(logits=logits).sample().item())
    return a, float(torch.sigmoid(troop_mean).item())

def decode(o):
    return (f"land={o[0]*100:5.2f}% troops={o[1]*200000:8.0f} gold={o[2]*200000:8.0f} "
            f"enAlive={o[3]*60:2.0f} emptyAdj={int(o[4])} enNbr={o[5]*6:.0f} "
            f"ratio={o[6]:.2f} allies={o[7]*5:.0f} cities={o[9]*8:.1f} silo={int(o[10])} coast={int(o[11])}")

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
    o = rpc({"cmd": "reset", "seed": 1000 + ep})["obs"]
    hist = [0]*N_ACT; troops_used = []
    done, steps, total = False, 0, 0.0
    print(f"\n===== EPISODE {ep} (seed {1000+ep}) =====")
    print(f"  start: {decode(o)}")
    while not done and steps < 2000:
        a, tr = choose(o); hist[a] += 1; troops_used.append(tr)
        r = rpc({"cmd": "step", "action": a, "troop": tr})
        o = r["obs"]; done = r["done"]; total += r["reward"]; steps += 1
        if steps % 40 == 0 or done:
            print(f"  step {steps:4d} act={ACTIONS[a]:12s} tr={tr:.2f} | {decode(o)}")
    print(f"  END: {steps} steps, reward {total:+.3f}, avg troop-frac {sum(troops_used)/len(troops_used):.2f}")
    order = sorted(range(N_ACT), key=lambda i: -hist[i])
    print("  action histogram: " + ", ".join(f"{ACTIONS[i]}={hist[i]}" for i in order if hist[i] > 0))

proc.stdin.close(); proc.terminate()
