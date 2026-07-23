# Load the trained CNN policy (data/torch_cnn.pt) and record ONE game to viz/replay.js via the env's
# recorder, so you can watch the "eyes" agent. Run:  MAP=box SEED=90001 python train_torch/record_cnn.py
import subprocess, json, os
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical, Normal

N_OBS, N_ACT = 16, 13
FCAND, KCAND, DIPLO = 7, 6, 3
KP, FP, NTYPE = 8, 4, 4
PLACE_MAP = {6: 0, 7: 1, 8: 2, 12: 3}
GC, GW = 6, 32
SEED = int(os.environ.get("SEED", "90001"))

class CNNActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(GC, 16, 3, padding=1); self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.trunk = nn.Linear(32 + N_OBS, 32)
        self.act = nn.Linear(32, N_ACT + 1); self.vf = nn.Linear(32, 1)
        self.cand = nn.Sequential(nn.Linear(FCAND, 16), nn.Tanh(), nn.Linear(16, 1))
        self.place = nn.Sequential(nn.Linear(FP + NTYPE, 16), nn.Tanh(), nn.Linear(16, 1))
        self.troop_log_std = nn.Parameter(torch.tensor(-0.5))
    def forward(self, grid, scal):
        h = F.relu(self.conv1(grid)); h = F.max_pool2d(h, 2); h = F.relu(self.conv2(h))
        g = h.mean(dim=(-1, -2)); t = torch.tanh(self.trunk(torch.cat([g, scal], dim=-1)))
        out = self.act(t); return out[..., :N_ACT], out[..., N_ACT]
    def score_cands(self, c): return self.cand(c).squeeze(-1)
    def score_place(self, f): return self.place(f).squeeze(-1)

net = CNNActorCritic(); net.load_state_dict(torch.load(os.path.join("data", "torch_cnn.pt")), strict=False); net.eval()

def pad(rows, K, Fd):
    c = torch.zeros(K, Fd); m = torch.zeros(K)
    for i, row in enumerate(rows[:K]): c[i] = torch.tensor(row, dtype=torch.float32); m[i] = 1.0
    return c, m

def tsx_cmd():
    if os.environ.get("TSX"): return os.environ["TSX"].split()
    local = os.path.join("node_modules", ".bin", "tsx.cmd" if os.name == "nt" else "tsx")
    return [local] if os.path.exists(local) else ["npx", "tsx"]
p = subprocess.Popen(tsx_cmd() + ["src/env/env_server.ts"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rpc(m): p.stdin.write(json.dumps(m) + "\n"); p.stdin.flush(); return json.loads(p.stdout.readline())

r = rpc({"cmd": "reset", "seed": SEED, "spatial": True, "record": True})
o, cds, pts, sp = r["obs"], r["cands"], r["ptiles"], r["spatial"]
steps, total = 0, 0.0
while steps < 2000:
    grid = torch.tensor(sp, dtype=torch.float32).view(1, GC, GW, GW); scal = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
    cpad, cmask = pad(cds, KCAND, FCAND); ppad, pmask = pad(pts, KP, FP)
    with torch.no_grad():
        logits, tmean = net(grid, scal)
        a = int(Categorical(logits=logits[0]).sample())
        troop = float(torch.sigmoid(Normal(tmean[0], net.troop_log_std.exp()).sample()))
        tgt = int(Categorical(logits=net.score_cands(cpad).masked_fill(cmask == 0, -1e9)).sample())
        oh = torch.zeros(NTYPE)
        if a in PLACE_MAP: oh[PLACE_MAP[a]] = 1.0
        feats = torch.cat([ppad, oh.expand(KP, NTYPE)], dim=-1)
        ptg = int(Categorical(logits=net.score_place(feats).masked_fill(pmask == 0, -1e9)).sample())
    r = rpc({"cmd": "step", "action": a, "troop": troop, "target": tgt, "ptarget": ptg})
    o, cds, pts, sp = r["obs"], r["cands"], r["ptiles"], r.get("spatial", sp); total += r["reward"]; steps += 1
    if r["done"]: break
p.stdin.close(); p.terminate()
print(f"recorded seed {SEED}: {steps} steps, reward {total:+.3f} -> viz/replay.js (open viz/index.html)")
