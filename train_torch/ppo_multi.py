# PPO with MULTI-ACTION + troop RESERVE (Phase 1: throughput + discipline).
# The nations' edge is doing many things at once with troop discipline, not vision (the CNN A/B
# confirmed vision didn't help). So: the action head is 13 independent BERNOULLIs -- the agent fires
# ANY SUBSET of actions each decision (attack AND build AND ally simultaneously). And a RESERVE head
# says what fraction of the army to keep back; the env splits the rest among fired attacks. Diplomacy
# and placement targets (candidate heads) apply only when their action fires (conditional log-probs).
# Run:  MAP=box DIFFICULTY=medium python train_torch/ppo_multi.py
import subprocess, json, os, sys
import torch, torch.nn as nn
from torch.distributions import Bernoulli, Normal, Categorical

N_OBS, N_ACT, N_HID = 16, 13, 32
FCAND, KCAND, DIPLO = 7, 6, 3
KP, FP = 8, 4
PLACE_ACTS = [6, 7, 8, 12]        # actions that place a structure (use the placement target)
GAMMA, LAMBDA, CLIP = 0.99, 0.95, 0.2
LR, WEIGHT_DECAY = 1.5e-3, 1e-4
EPOCHS, MINIBATCH = 4, 512
ENT_COEF, VF_COEF = 0.01, 0.5
EPISODES = int(os.environ.get("EPISODES", "1000"))
BATCH_EP = int(os.environ.get("BATCH", "10"))
POOL = int(os.environ.get("POOL", "32"))
TRAIN_SEEDS = [1000 + i for i in range(POOL)]
VAL_SEEDS = [90001, 90002, 90003]
MAX_STEPS = 2000
SAVE_PATH = os.path.join("data", "torch_multi.pt")

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_OBS, N_HID)
        self.act = nn.Linear(N_HID, N_ACT)                 # 13 Bernoulli logits (which actions to fire)
        self.res = nn.Linear(N_HID, 1)                     # reserve mean (fraction of army to keep)
        self.vf = nn.Linear(N_HID, 1)
        self.cand = nn.Sequential(nn.Linear(FCAND, 16), nn.Tanh(), nn.Linear(16, 1))
        self.place = nn.Sequential(nn.Linear(FP, 16), nn.Tanh(), nn.Linear(16, 1))
        self.res_log_std = nn.Parameter(torch.tensor(-0.5))
    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        return self.act(h), self.res(h).squeeze(-1), self.vf(h).squeeze(-1)
    def score_cands(self, c): return self.cand(c).squeeze(-1)
    def score_place(self, f): return self.place(f).squeeze(-1)

net = Net(); opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def pad(rows, K, Fd):
    c = torch.zeros(K, Fd); m = torch.zeros(K)
    for i, r in enumerate(rows[:K]): c[i] = torch.tensor(r, dtype=torch.float32); m[i] = 1.0
    return c, m
def cdist(cpad, cmask): return Categorical(logits=net.score_cands(cpad).masked_fill(cmask == 0, -1e9))
def pdist(ppad, pmask): return Categorical(logits=net.score_place(ppad).masked_fill(pmask == 0, -1e9))

def tsx_cmd():
    if os.environ.get("TSX"): return os.environ["TSX"].split()
    local = os.path.join("node_modules", ".bin", "tsx.cmd" if os.name == "nt" else "tsx")
    return [local] if os.path.exists(local) else ["npx", "tsx"]
proc = subprocess.Popen(tsx_cmd() + [os.environ.get("ENVFILE", "src/env/env_server.ts")],
                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rpc(m): proc.stdin.write(json.dumps(m) + "\n"); proc.stdin.flush(); return json.loads(proc.stdout.readline())

def gae(rew, val):
    adv = [0.0]*len(rew); last = 0.0
    for t in reversed(range(len(rew))):
        nv = val[t+1] if t+1 < len(val) else 0.0
        last = (rew[t] + GAMMA*nv - val[t]) + GAMMA*LAMBDA*last; adv[t] = last
    return adv, [adv[t]+val[t] for t in range(len(rew))]

def rollout(seed):
    r0 = rpc({"cmd": "reset", "seed": seed}); o, cds, pts = r0["obs"], r0["cands"], r0["ptiles"]
    O, ACT, RR, TG, PTG, CD, MK, PT, PM, LP, VAL, REW = ([] for _ in range(12))
    done, steps, total = False, 0, 0.0
    while not done and steps < MAX_STEPS:
        cpad, cmask = pad(cds, KCAND, FCAND); ppad, pmask = pad(pts, KP, FP)
        with torch.no_grad():
            alog, rmean, value = net(torch.tensor(o, dtype=torch.float32))
            adist = Bernoulli(logits=alog); acts = adist.sample()                 # [13] of 0/1
            rdist = Normal(rmean, net.res_log_std.exp()); rraw = rdist.sample(); reserve = float(torch.sigmoid(rraw))
            tgt = cdist(cpad, cmask).sample(); ptg = pdist(ppad, pmask).sample()
            lp = adist.log_prob(acts).sum() + rdist.log_prob(rraw)
            if acts[DIPLO] == 1: lp = lp + cdist(cpad, cmask).log_prob(tgt)
            if any(acts[a] == 1 for a in PLACE_ACTS): lp = lp + pdist(ppad, pmask).log_prob(ptg)
        r = rpc({"cmd": "step", "actions": [int(a) for a in acts], "reserve": reserve, "target": int(tgt), "ptarget": int(ptg)})
        O.append(o); ACT.append(acts); RR.append(float(rraw)); TG.append(int(tgt)); PTG.append(int(ptg))
        CD.append(cpad); MK.append(cmask); PT.append(ppad); PM.append(pmask); LP.append(float(lp)); VAL.append(float(value)); REW.append(r["reward"])
        o = r["obs"]; cds = r["cands"]; pts = r["ptiles"]; done = r["done"]; total += r["reward"]; steps += 1
    adv, ret = gae(REW, VAL)
    return O, ACT, RR, TG, PTG, CD, MK, PT, PM, LP, adv, ret, total

def run_val():
    tot = 0.0
    for sd in VAL_SEEDS:
        r0 = rpc({"cmd": "reset", "seed": sd}); o, cds, pts = r0["obs"], r0["cands"], r0["ptiles"]; done, steps, rep = False, 0, 0.0
        while not done and steps < MAX_STEPS:
            cpad, cmask = pad(cds, KCAND, FCAND); ppad, pmask = pad(pts, KP, FP)
            with torch.no_grad():
                alog, rmean, _ = net(torch.tensor(o, dtype=torch.float32))
                acts = Bernoulli(logits=alog).sample(); reserve = float(torch.sigmoid(Normal(rmean, net.res_log_std.exp()).sample()))
                tgt = int(cdist(cpad, cmask).sample()); ptg = int(pdist(ppad, pmask).sample())
            r = rpc({"cmd": "step", "actions": [int(a) for a in acts], "reserve": reserve, "target": tgt, "ptarget": ptg})
            o = r["obs"]; cds = r["cands"]; pts = r["ptiles"]; done = r["done"]; rep += r["reward"]; steps += 1
        tot += rep
    return tot / len(VAL_SEEDS)

recent = []
for upd in range(EPISODES // BATCH_EP):
    O, ACT, RR, TG, PTG, CD, MK, PT, PM, LP, ADV, RET = ([] for _ in range(12))
    for i in range(BATCH_EP):
        seed = TRAIN_SEEDS[(upd*BATCH_EP + i) % len(TRAIN_SEEDS)]
        o, ac, rr, tg, ptg, cd, mk, pt, pm, lp, adv, ret, total = rollout(seed)
        O += o; ACT += ac; RR += rr; TG += tg; PTG += ptg; CD += cd; MK += mk; PT += pt; PM += pm; LP += lp; ADV += adv; RET += ret
        recent.append(total); recent = recent[-20:]
    O = torch.tensor(O, dtype=torch.float32); ACT = torch.stack(ACT); RR = torch.tensor(RR, dtype=torch.float32)
    TG = torch.tensor(TG); PTG = torch.tensor(PTG); CD = torch.stack(CD); MK = torch.stack(MK); PT = torch.stack(PT); PM = torch.stack(PM)
    LP = torch.tensor(LP, dtype=torch.float32); ADV = torch.tensor(ADV, dtype=torch.float32); RET = torch.tensor(RET, dtype=torch.float32)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8); N = len(O)
    place_fired = (ACT[:, PLACE_ACTS].sum(-1) > 0).float()
    for _ in range(EPOCHS):
        idx = torch.randperm(N)
        for s in range(0, N, MINIBATCH):
            mb = idx[s:s+MINIBATCH]
            alog, rmean, value = net(O[mb])
            adist = Bernoulli(logits=alog); rdist = Normal(rmean, net.res_log_std.exp())
            cd_ = Categorical(logits=net.score_cands(CD[mb]).masked_fill(MK[mb] == 0, -1e9))
            pd_ = Categorical(logits=net.score_place(PT[mb]).masked_fill(PM[mb] == 0, -1e9))
            new_lp = (adist.log_prob(ACT[mb]).sum(-1) + rdist.log_prob(RR[mb])
                      + (ACT[mb][:, DIPLO]) * cd_.log_prob(TG[mb]) + place_fired[mb] * pd_.log_prob(PTG[mb]))
            ratio = torch.exp(new_lp - LP[mb])
            pg = -torch.min(ratio*ADV[mb], torch.clamp(ratio, 1-CLIP, 1+CLIP)*ADV[mb]).mean()
            loss = pg + VF_COEF*((value - RET[mb])**2).mean() - ENT_COEF*adist.entropy().sum(-1).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
    ma = sum(recent)/len(recent); val = run_val()
    # avg #actions fired/step this batch (throughput sanity)
    nact = ACT.sum(-1).mean().item()
    print(f"update {upd:3d} (ep {(upd+1)*BATCH_EP:4d}): ma20 {ma:+.3f}, VAL {val:+.3f}, acts/step {nact:.1f}, reserve {torch.sigmoid(RR.mean()).item():.2f}")
    sys.stdout.flush()

torch.save(net.state_dict(), SAVE_PATH)
proc.stdin.close(); proc.terminate()
print(f"saved multi-action policy to {SAVE_PATH}")
