# Scoreboard: run the current MLP policy (data/torch_policy.pt) over N held-out seeds and report
# peak land %, survival, and WIN-RATE (the project goal metric). A "win" = agent is the last player
# standing OR reaches >=80% of the map. Configure via env: MAP, DIFFICULTY, NUM_NATIONS, BOTS, SEEDS.
# e.g.  MAP=box_full DIFFICULTY=impossible SEEDS=8 python train_torch/winrate.py
import subprocess, json, os
import torch, torch.nn as nn
from torch.distributions import Categorical, Normal

N_OBS, N_ACT, N_HID = 16, 13, 24
FCAND, KCAND, DIPLO = 7, 6, 3
KP, FP, NTYPE = 8, 4, 4
PLACE_MAP = {6: 0, 7: 1, 8: 2, 12: 3}
SEEDS = int(os.environ.get("SEEDS", "8"))
VAL_BASE = 90001

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_OBS, N_HID); self.fc2 = nn.Linear(N_HID, N_ACT + 1)
        self.cand = nn.Sequential(nn.Linear(FCAND, 16), nn.Tanh(), nn.Linear(16, 1))
        self.place = nn.Sequential(nn.Linear(FP + NTYPE, 16), nn.Tanh(), nn.Linear(16, 1))
        self.troop_log_std = nn.Parameter(torch.tensor(-0.5))
    def forward(self, x):
        h = torch.tanh(self.fc1(x)); out = self.fc2(h); return out[:N_ACT], out[N_ACT]
    def sc(self, c): return self.cand(c).squeeze(-1)
    def sp(self, f): return self.place(f).squeeze(-1)

net = Net(); net.load_state_dict(torch.load(os.path.join("data", "torch_policy.pt")), strict=False); net.eval()

def pad(rows, K, Fd):
    c = torch.zeros(K, Fd); m = torch.zeros(K)
    for i, r in enumerate(rows[:K]): c[i] = torch.tensor(r, dtype=torch.float32); m[i] = 1.0
    return c, m

def tsx_cmd():
    if os.environ.get("TSX"): return os.environ["TSX"].split()
    local = os.path.join("node_modules", ".bin", "tsx.cmd" if os.name == "nt" else "tsx")
    return [local] if os.path.exists(local) else ["npx", "tsx"]
p = subprocess.Popen(tsx_cmd() + ["src/env/env_server.ts"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rpc(m): p.stdin.write(json.dumps(m) + "\n"); p.stdin.flush(); return json.loads(p.stdout.readline())

print(f"MAP={os.environ.get('MAP','box')} DIFFICULTY={os.environ.get('DIFFICULTY','medium')} "
      f"players={os.environ.get('NUM_NATIONS','?')}n+{os.environ.get('BOTS','?')}t over {SEEDS} seeds")
wins = 0
for s in range(SEEDS):
    r = rpc({"cmd": "reset", "seed": VAL_BASE + s}); o, cds, pts = r["obs"], r["cands"], r["ptiles"]
    peak, alive_end, steps = 0.0, False, 0
    while steps < int(os.environ.get("MAXST", "1500")):
        cpad, cmask = pad(cds, KCAND, FCAND); ppad, pmask = pad(pts, KP, FP)
        with torch.no_grad():
            logits, tmean = net(torch.tensor(o, dtype=torch.float32))
            a = int(Categorical(logits=logits).sample()); troop = float(torch.sigmoid(Normal(tmean, net.troop_log_std.exp()).sample()))
            tgt = int(Categorical(logits=net.sc(cpad).masked_fill(cmask == 0, -1e9)).sample())
            oh = torch.zeros(NTYPE)
            if a in PLACE_MAP: oh[PLACE_MAP[a]] = 1.0
            ptg = int(Categorical(logits=net.sp(torch.cat([ppad, oh.expand(KP, NTYPE)], -1)).masked_fill(pmask == 0, -1e9)).sample())
        r = rpc({"cmd": "step", "action": a, "troop": troop, "target": tgt, "ptarget": ptg})
        o = r["obs"]; cds = r["cands"]; pts = r["ptiles"]; peak = max(peak, o[0]); alive_end = o[0] > 0; steps += 1
        if r["done"]: break
    won = alive_end and peak >= 0.8            # last-standing shows as alive at a done with high share
    wins += won
    print(f"  seed {VAL_BASE+s}: peak land {peak*100:5.2f}%  {'WON' if won else ('survived' if alive_end else 'died')}  ({steps} steps)")
print(f"WIN-RATE: {wins}/{SEEDS} = {100*wins/SEEDS:.0f}%")
p.stdin.close(); p.terminate()
