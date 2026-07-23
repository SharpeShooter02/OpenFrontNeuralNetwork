# PPO with a CONVOLUTIONAL trunk — the agent's "eyes". Reads the map as a 6x32x32 image (mine/enemy/
# neutral/impassable/my+enemy structures) via conv layers, pools to a global feature, concatenates the
# 16 scalar obs, and feeds the existing heads (action / troop / value / diplomacy-candidate / placement).
# This fixes the "blind" gap: the policy sees the board instead of only aggregate scalars.
# Attacks stay player-directed (engine mechanic); spatial nuke/build targeting is a later refinement.
# Run:  MAP=box python train_torch/ppo_cnn.py   (env must be asked for spatial obs -> we send spatial:true)
import subprocess, json, os, sys
import torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import Categorical, Normal

N_OBS, N_ACT = 16, 13
FCAND, KCAND, DIPLO = 7, 6, 3
KP, FP, NTYPE = 8, 4, 4
PLACE_MAP = {6: 0, 7: 1, 8: 2, 12: 3}
GC, GW = 6, 32                    # spatial grid: channels, width/height
GAMMA, LAMBDA = 0.99, 0.95
CLIP, LR, WEIGHT_DECAY = 0.2, 1e-3, 1e-4
EPOCHS, MINIBATCH = 4, 256
ENT_COEF, VF_COEF = 0.01, 0.5
EPISODES = int(os.environ.get("EPISODES", "600"))
BATCH_EP = int(os.environ.get("BATCH", "10"))
POOL = int(os.environ.get("POOL", "32"))
TRAIN_SEEDS = [1000 + i for i in range(POOL)]
VAL_SEEDS = [90001, 90002, 90003]
MAX_STEPS = 2000
SAVE_PATH = os.path.join("data", "torch_cnn.pt")

class CNNActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(GC, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.trunk = nn.Linear(32 + N_OBS, 32)                    # global CNN feature (32) + scalar obs (16)
        self.act = nn.Linear(32, N_ACT + 1)                       # action logits + troop mean
        self.vf = nn.Linear(32, 1)
        self.cand = nn.Sequential(nn.Linear(FCAND, 16), nn.Tanh(), nn.Linear(16, 1))
        self.place = nn.Sequential(nn.Linear(FP + NTYPE, 16), nn.Tanh(), nn.Linear(16, 1))
        self.troop_log_std = nn.Parameter(torch.tensor(-0.5))
    def forward(self, grid, scal):                               # grid [B,GC,GW,GW], scal [B,N_OBS]
        h = F.relu(self.conv1(grid)); h = F.max_pool2d(h, 2); h = F.relu(self.conv2(h))
        g = h.mean(dim=(-1, -2))                                 # global average pool -> [B,32]
        t = torch.tanh(self.trunk(torch.cat([g, scal], dim=-1)))
        out = self.act(t)
        return out[..., :N_ACT], out[..., N_ACT], self.vf(t).squeeze(-1)
    def score_cands(self, cands): return self.cand(cands).squeeze(-1)
    def score_place(self, feats): return self.place(feats).squeeze(-1)

net = CNNActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def dists(logits, tmean): return Categorical(logits=logits), Normal(tmean, net.troop_log_std.exp())
def to_grid(spatial): return torch.tensor(spatial, dtype=torch.float32).view(1, GC, GW, GW)
def pad(rows, K, Fd):
    c = torch.zeros(K, Fd); m = torch.zeros(K)
    for i, row in enumerate(rows[:K]): c[i] = torch.tensor(row, dtype=torch.float32); m[i] = 1.0
    return c, m
def cand_dist(cpad, cmask): return Categorical(logits=net.score_cands(cpad).masked_fill(cmask == 0, -1e9))
def type_oh(actions):
    oh = torch.zeros(len(actions), NTYPE)
    for a, ti in PLACE_MAP.items(): oh[:, ti] = (actions == a).float()
    return oh
def place_dist(ppad, pmask, oh):
    feats = torch.cat([ppad, oh.unsqueeze(-2).expand(*ppad.shape[:-1], NTYPE)], dim=-1)
    return Categorical(logits=net.score_place(feats).masked_fill(pmask == 0, -1e9))
def uses_place(actions): return sum((actions == a).float() for a in PLACE_MAP)

def tsx_cmd():
    if os.environ.get("TSX"): return os.environ["TSX"].split()
    local = os.path.join("node_modules", ".bin", "tsx.cmd" if os.name == "nt" else "tsx")
    return [local] if os.path.exists(local) else ["npx", "tsx"]
proc = subprocess.Popen(tsx_cmd() + [os.environ.get("ENVFILE", "src/env/env_server.ts")],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rpc(msg):
    proc.stdin.write(json.dumps(msg) + "\n"); proc.stdin.flush()
    return json.loads(proc.stdout.readline())

def gae(rew, val):
    adv = [0.0]*len(rew); last = 0.0
    for t in reversed(range(len(rew))):
        nv = val[t+1] if t+1 < len(val) else 0.0
        last = (rew[t] + GAMMA*nv - val[t]) + GAMMA*LAMBDA*last; adv[t] = last
    return adv, [adv[t]+val[t] for t in range(len(rew))]

def rollout(seed):
    r0 = rpc({"cmd": "reset", "seed": seed, "spatial": True}); o = r0["obs"]; cds = r0["cands"]; pts = r0["ptiles"]; sp = r0["spatial"]
    G, S, A, TR, TG, PTG, CD, MK, PT, PM, LP, VAL, REW = [], [], [], [], [], [], [], [], [], [], [], [], []
    done, steps, total = False, 0, 0.0
    while not done and steps < MAX_STEPS:
        grid = to_grid(sp); scal = torch.tensor(o, dtype=torch.float32).unsqueeze(0)
        cpad, cmask = pad(cds, KCAND, FCAND); ppad, pmask = pad(pts, KP, FP)
        with torch.no_grad():
            logits, tmean, value = net(grid, scal)
            cat, tdist = dists(logits[0], tmean[0])
            a = cat.sample(); traw = tdist.sample(); tgt = cand_dist(cpad, cmask).sample()
            oh = type_oh(torch.tensor([int(a)]))[0]; ptg = place_dist(ppad, pmask, oh).sample()
            lp = cat.log_prob(a) + tdist.log_prob(traw)
            if int(a) == DIPLO: lp = lp + cand_dist(cpad, cmask).log_prob(tgt)
            if int(a) in PLACE_MAP: lp = lp + place_dist(ppad, pmask, oh).log_prob(ptg)
        r = rpc({"cmd": "step", "action": int(a), "troop": float(torch.sigmoid(traw)), "target": int(tgt), "ptarget": int(ptg)})
        G.append(sp); S.append(o); A.append(int(a)); TR.append(float(traw)); TG.append(int(tgt)); PTG.append(int(ptg))
        CD.append(cpad); MK.append(cmask); PT.append(ppad); PM.append(pmask); LP.append(float(lp)); VAL.append(float(value)); REW.append(r["reward"])
        o = r["obs"]; cds = r["cands"]; pts = r["ptiles"]; sp = r["spatial"]; done = r["done"]; total += r["reward"]; steps += 1
    adv, ret = gae(REW, VAL)
    return G, S, A, TR, TG, PTG, CD, MK, PT, PM, LP, adv, ret, total

def run_val():
    tot = 0.0
    for sd in VAL_SEEDS:
        r0 = rpc({"cmd": "reset", "seed": sd, "spatial": True}); o = r0["obs"]; cds = r0["cands"]; pts = r0["ptiles"]; sp = r0["spatial"]; done, steps, rep = False, 0, 0.0
        while not done and steps < MAX_STEPS:
            cpad, cmask = pad(cds, KCAND, FCAND); ppad, pmask = pad(pts, KP, FP)
            with torch.no_grad():
                logits, tmean, _ = net(to_grid(sp), torch.tensor(o, dtype=torch.float32).unsqueeze(0))
                cat, tdist = dists(logits[0], tmean[0])
                a = int(cat.sample()); tgt = int(cand_dist(cpad, cmask).sample())
                ptg = int(place_dist(ppad, pmask, type_oh(torch.tensor([a]))[0]).sample())
            r = rpc({"cmd": "step", "action": a, "troop": float(torch.sigmoid(tdist.sample())), "target": tgt, "ptarget": ptg})
            o = r["obs"]; cds = r["cands"]; pts = r["ptiles"]; sp = r["spatial"]; done = r["done"]; rep += r["reward"]; steps += 1
        tot += rep
    return tot / len(VAL_SEEDS)

recent = []
for upd in range(EPISODES // BATCH_EP):
    G, S, A, TR, TG, PTG, CD, MK, PT, PM, LP, ADV, RET = ([] for _ in range(13))
    for i in range(BATCH_EP):
        seed = TRAIN_SEEDS[(upd*BATCH_EP + i) % len(TRAIN_SEEDS)]
        g, s, a, tr, tg, ptg, cd, mk, pt, pm, lp, adv, ret, total = rollout(seed)
        G += g; S += s; A += a; TR += tr; TG += tg; PTG += ptg; CD += cd; MK += mk; PT += pt; PM += pm; LP += lp; ADV += adv; RET += ret
        recent.append(total); recent = recent[-20:]
    G = torch.tensor(G, dtype=torch.float32).view(-1, GC, GW, GW); S = torch.tensor(S, dtype=torch.float32)
    A = torch.tensor(A); TR = torch.tensor(TR, dtype=torch.float32); TG = torch.tensor(TG); PTG = torch.tensor(PTG)
    CD = torch.stack(CD); MK = torch.stack(MK); PT = torch.stack(PT); PM = torch.stack(PM); LP = torch.tensor(LP, dtype=torch.float32)
    ADV = torch.tensor(ADV, dtype=torch.float32); RET = torch.tensor(RET, dtype=torch.float32)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8); N = len(S)
    for _ in range(EPOCHS):
        idx = torch.randperm(N)
        for st in range(0, N, MINIBATCH):
            mb = idx[st:st+MINIBATCH]
            logits, tmean, value = net(G[mb], S[mb])
            cat, tdist = dists(logits, tmean)
            cdist = cand_dist(CD[mb], MK[mb]); pdist = place_dist(PT[mb], PM[mb], type_oh(A[mb]))
            new_lp = (cat.log_prob(A[mb]) + tdist.log_prob(TR[mb])
                      + (A[mb] == DIPLO).float()*cdist.log_prob(TG[mb]) + uses_place(A[mb])*pdist.log_prob(PTG[mb]))
            ratio = torch.exp(new_lp - LP[mb])
            pg = -torch.min(ratio*ADV[mb], torch.clamp(ratio, 1-CLIP, 1+CLIP)*ADV[mb]).mean()
            loss = pg + VF_COEF*((value - RET[mb])**2).mean() - ENT_COEF*cat.entropy().mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
    ma = sum(recent)/len(recent); val = run_val()
    print(f"update {upd:3d} (ep {(upd+1)*BATCH_EP:4d}): ma20 {ma:+.3f}, pg {pg.item():+.3f}, VAL {val:+.3f}"); sys.stdout.flush()

torch.save(net.state_dict(), SAVE_PATH)
proc.stdin.close(); proc.terminate()
print(f"saved CNN policy to {SAVE_PATH}")
