# Diagnostic: load the trained torch policy, play a few held-out episodes, and print WHAT the agent
# does — action histogram, decoded trajectory, and learned DIPLOMACY (accept/request/break counts).
# Run from project root:  python train_torch/diagnose.py
import subprocess, json, os
import torch, torch.nn as nn
from torch.distributions import Categorical

N_OBS, N_ACT, N_HID = 16, 13, 24
FCAND, KCAND, DIPLO = 7, 6, 3
ACTIONS = ["expandEmpty","attackWeak","attackStrong","diplomacy","wait","buildCity",
           "buildDefense","buildSilo","buildSAM","nuke","boatAttack","buildPort","buildFactory"]

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_OBS, N_HID)
        self.fc2 = nn.Linear(N_HID, N_ACT + 1)
        self.cand = nn.Sequential(nn.Linear(FCAND, 16), nn.Tanh(), nn.Linear(16, 1))
        self.troop_log_std = nn.Parameter(torch.tensor(-0.5))
    def forward(self, x):
        h = torch.tanh(self.fc1(x)); out = self.fc2(h)
        return out[..., :N_ACT], out[..., N_ACT]
    def score_cands(self, cands):
        return self.cand(cands).squeeze(-1)

net = PolicyNet()
net.load_state_dict(torch.load(os.path.join("data", "torch_policy.pt")), strict=False)  # ignore critic (vf.*) keys
net.eval()

def pad_cands(cands):
    c = torch.zeros(KCAND, FCAND); m = torch.zeros(KCAND)
    for i, row in enumerate(cands[:KCAND]):
        c[i] = torch.tensor(row, dtype=torch.float32); m[i] = 1.0
    return c, m

def choose(obs, cands):
    x = torch.tensor(obs, dtype=torch.float32)
    logits, troop_mean = net(x)
    a = int(Categorical(logits=logits).sample().item())
    cpad, cmask = pad_cands(cands)
    scores = net.score_cands(cpad).masked_fill(cmask == 0, -1e9)
    tgt = int(Categorical(logits=scores).sample().item())
    return a, float(torch.sigmoid(troop_mean).item()), tgt

def decode(o):
    return (f"land={o[0]*100:5.2f}% troops={o[1]*200000:8.0f} goldL={o[2]:.2f} "
            f"enNbr={o[5]*6:.0f} wkRatio={o[6]:.2f} stRatio={o[12]:.2f} pressure={o[13]:.2f} "
            f"allies={o[7]*5:.0f} allyBack={o[14]:.2f} offer={o[15]:.2f} cities={o[9]*8:.1f} silo={int(o[10])}")

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
    r0 = rpc({"cmd": "reset", "seed": 90001 + ep}); o = r0["obs"]; cands = r0["cands"]
    hist = [0]*N_ACT; troops_used = []; diplo = {"accept": 0, "request": 0, "break": 0}
    done, steps, total = False, 0, 0.0
    print(f"\n===== EPISODE {ep} (held-out seed {90001+ep}) =====")
    print(f"  start: {decode(o)}")
    while not done and steps < 2000:
        a, tr, tgt = choose(o, cands); hist[a] += 1; troops_used.append(tr)
        if a == DIPLO and 0 < tgt < len(cands):                     # infer diplomacy kind from candidate features
            c = cands[tgt]
            if c[3] == 1: diplo["accept"] += 1
            elif c[4] == 1: diplo["request"] += 1
            elif c[2] == 1: diplo["break"] += 1
        r = rpc({"cmd": "step", "action": a, "troop": tr, "target": tgt})
        o = r["obs"]; cands = r["cands"]; done = r["done"]; total += r["reward"]; steps += 1
        if steps % 40 == 0 or done:
            print(f"  step {steps:4d} act={ACTIONS[a]:12s} tr={tr:.2f} | {decode(o)}")
    print(f"  END: {steps} steps, reward {total:+.3f}, avg troop-frac {sum(troops_used)/len(troops_used):.2f}")
    order = sorted(range(N_ACT), key=lambda i: -hist[i])
    print("  action histogram: " + ", ".join(f"{ACTIONS[i]}={hist[i]}" for i in order if hist[i] > 0))
    print(f"  diplomacy: accept={diplo['accept']} request={diplo['request']} break={diplo['break']}")

proc.stdin.close(); proc.terminate()
