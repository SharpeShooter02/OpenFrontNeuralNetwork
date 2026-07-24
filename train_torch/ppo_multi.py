# PPO with STRUCTURED multi-action + troop reserve (Phase 1 v2).
# v1 used 13 independent Bernoullis -> 2^13 combos, unlearnable, stuck firing ~half of everything.
# v2 factors one decision into a few SMALL sub-decisions made simultaneously (like the nations):
#   expand? (bern) + which-attack (cat: none/weak/strong/boat/nuke) + which-build (cat: none/6 builds)
#   + diplomacy? (bern) + reserve (fraction to keep). Up to ~4 actions/step, each a tiny learnable head.
# The policy assembles the env's 13-bit actions array from these; env is unchanged.
# Run:  MAP=box DIFFICULTY=medium python train_torch/ppo_multi.py
import subprocess, json, os, sys
import torch, torch.nn as nn
from torch.distributions import Bernoulli, Normal, Categorical

N_OBS, N_HID = 16, 32
FCAND, KCAND = 7, 6
KP, FP = 8, 4
ATTACK_MAP = [None, 1, 2, 10, 9]              # cat idx -> action bit (none/weakest/strongest/boat/nuke)
BUILD_MAP = [None, 5, 6, 7, 8, 11, 12]        # cat idx -> action bit (none/city/defense/silo/SAM/port/factory)
PLACEABLE = {2, 3, 4, 6}                       # build cat idxs that use the placement target (defense/silo/SAM/factory)
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
        self.expand = nn.Linear(N_HID, 1)                  # bern
        self.attack = nn.Linear(N_HID, len(ATTACK_MAP))    # cat
        self.build = nn.Linear(N_HID, len(BUILD_MAP))      # cat
        self.diplo = nn.Linear(N_HID, 1)                   # bern
        self.res = nn.Linear(N_HID, 1)                     # reserve mean
        self.vf = nn.Linear(N_HID, 1)
        self.cand = nn.Sequential(nn.Linear(FCAND, 16), nn.Tanh(), nn.Linear(16, 1))
        self.place = nn.Sequential(nn.Linear(FP, 16), nn.Tanh(), nn.Linear(16, 1))
        self.res_log_std = nn.Parameter(torch.tensor(-0.5))
    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        return (self.expand(h).squeeze(-1), self.attack(h), self.build(h),
                self.diplo(h).squeeze(-1), self.res(h).squeeze(-1), self.vf(h).squeeze(-1))
    def score_cands(self, c): return self.cand(c).squeeze(-1)
    def score_place(self, f): return self.place(f).squeeze(-1)

net = Net(); opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def pad(rows, K, Fd):
    c = torch.zeros(K, Fd); m = torch.zeros(K)
    for i, r in enumerate(rows[:K]): c[i] = torch.tensor(r, dtype=torch.float32); m[i] = 1.0
    return c, m
def cdist(cp, cm): return Categorical(logits=net.score_cands(cp).masked_fill(cm == 0, -1e9))
def pdist(pp, pm): return Categorical(logits=net.score_place(pp).masked_fill(pm == 0, -1e9))

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

def build_actions(ex, at, bd):
    a = [0]*13
    if ex: a[0] = 1
    if ATTACK_MAP[at] is not None: a[ATTACK_MAP[at]] = 1
    if BUILD_MAP[bd] is not None: a[BUILD_MAP[bd]] = 1
    return a

def rollout(seed):
    r0 = rpc({"cmd": "reset", "seed": seed}); o, cds, pts = r0["obs"], r0["cands"], r0["ptiles"]
    O, EX, AT, BD, DP, RR, TG, PTG, CD, MK, PT, PM, LP, VAL, REW = ([] for _ in range(15))
    done, steps, total = False, 0, 0.0
    while not done and steps < MAX_STEPS:
        cpad, cmask = pad(cds, KCAND, FCAND); ppad, pmask = pad(pts, KP, FP)
        with torch.no_grad():
            elog, alog, blog, dlog, rmean, value = net(torch.tensor(o, dtype=torch.float32))
            ed = Bernoulli(logits=elog); ad = Categorical(logits=alog); bd_ = Categorical(logits=blog); dd = Bernoulli(logits=dlog)
            rd = Normal(rmean, net.res_log_std.exp())
            ex = ed.sample(); at = ad.sample(); bd = bd_.sample(); dp = dd.sample(); rraw = rd.sample()
            tgt = cdist(cpad, cmask).sample(); ptg = pdist(ppad, pmask).sample()
            lp = ed.log_prob(ex) + ad.log_prob(at) + bd_.log_prob(bd) + dd.log_prob(dp) + rd.log_prob(rraw)
            if dp == 1: lp = lp + cdist(cpad, cmask).log_prob(tgt)
            if int(bd) in PLACEABLE: lp = lp + pdist(ppad, pmask).log_prob(ptg)
        acts = build_actions(int(ex), int(at), int(bd))
        if int(dp) == 1: acts[3] = 1
        r = rpc({"cmd": "step", "actions": acts, "reserve": float(torch.sigmoid(rraw)), "target": int(tgt), "ptarget": int(ptg)})
        O.append(o); EX.append(int(ex)); AT.append(int(at)); BD.append(int(bd)); DP.append(int(dp)); RR.append(float(rraw))
        TG.append(int(tgt)); PTG.append(int(ptg)); CD.append(cpad); MK.append(cmask); PT.append(ppad); PM.append(pmask)
        LP.append(float(lp)); VAL.append(float(value)); REW.append(r["reward"])
        o = r["obs"]; cds = r["cands"]; pts = r["ptiles"]; done = r["done"]; total += r["reward"]; steps += 1
    adv, ret = gae(REW, VAL)
    return O, EX, AT, BD, DP, RR, TG, PTG, CD, MK, PT, PM, LP, adv, ret, total

def run_val():
    tot = 0.0
    for sd in VAL_SEEDS:
        r0 = rpc({"cmd": "reset", "seed": sd}); o, cds, pts = r0["obs"], r0["cands"], r0["ptiles"]; done, steps, rep = False, 0, 0.0
        while not done and steps < MAX_STEPS:
            cpad, cmask = pad(cds, KCAND, FCAND); ppad, pmask = pad(pts, KP, FP)
            with torch.no_grad():
                elog, alog, blog, dlog, rmean, _ = net(torch.tensor(o, dtype=torch.float32))
                ex = int(Bernoulli(logits=elog).sample()); at = int(Categorical(logits=alog).sample())
                bd = int(Categorical(logits=blog).sample()); dp = int(Bernoulli(logits=dlog).sample())
                reserve = float(torch.sigmoid(Normal(rmean, net.res_log_std.exp()).sample()))
                tgt = int(cdist(cpad, cmask).sample()); ptg = int(pdist(ppad, pmask).sample())
            acts = build_actions(ex, at, bd)
            if dp == 1: acts[3] = 1
            r = rpc({"cmd": "step", "actions": acts, "reserve": reserve, "target": tgt, "ptarget": ptg})
            o = r["obs"]; cds = r["cands"]; pts = r["ptiles"]; done = r["done"]; rep += r["reward"]; steps += 1
        tot += rep
    return tot / len(VAL_SEEDS)

recent = []
for upd in range(EPISODES // BATCH_EP):
    O, EX, AT, BD, DP, RR, TG, PTG, CD, MK, PT, PM, LP, ADV, RET = ([] for _ in range(15))
    for i in range(BATCH_EP):
        seed = TRAIN_SEEDS[(upd*BATCH_EP + i) % len(TRAIN_SEEDS)]
        o, ex, at, bd, dp, rr, tg, ptg, cd, mk, pt, pm, lp, adv, ret, total = rollout(seed)
        O += o; EX += ex; AT += at; BD += bd; DP += dp; RR += rr; TG += tg; PTG += ptg; CD += cd; MK += mk; PT += pt; PM += pm; LP += lp; ADV += adv; RET += ret
        recent.append(total); recent = recent[-20:]
    O = torch.tensor(O, dtype=torch.float32); EX = torch.tensor(EX, dtype=torch.float32); AT = torch.tensor(AT); BD = torch.tensor(BD)
    DP = torch.tensor(DP, dtype=torch.float32); RR = torch.tensor(RR, dtype=torch.float32); TG = torch.tensor(TG); PTG = torch.tensor(PTG)
    CD = torch.stack(CD); MK = torch.stack(MK); PT = torch.stack(PT); PM = torch.stack(PM)
    LP = torch.tensor(LP, dtype=torch.float32); ADV = torch.tensor(ADV, dtype=torch.float32); RET = torch.tensor(RET, dtype=torch.float32)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8); N = len(O)
    placeable = torch.tensor([1.0 if int(b) in PLACEABLE else 0.0 for b in BD])
    for _ in range(EPOCHS):
        idx = torch.randperm(N)
        for s in range(0, N, MINIBATCH):
            mb = idx[s:s+MINIBATCH]
            elog, alog, blog, dlog, rmean, value = net(O[mb])
            ed = Bernoulli(logits=elog); ad = Categorical(logits=alog); bdd = Categorical(logits=blog); dd = Bernoulli(logits=dlog)
            rd = Normal(rmean, net.res_log_std.exp())
            cd_ = Categorical(logits=net.score_cands(CD[mb]).masked_fill(MK[mb] == 0, -1e9))
            pd_ = Categorical(logits=net.score_place(PT[mb]).masked_fill(PM[mb] == 0, -1e9))
            new_lp = (ed.log_prob(EX[mb]) + ad.log_prob(AT[mb]) + bdd.log_prob(BD[mb]) + dd.log_prob(DP[mb]) + rd.log_prob(RR[mb])
                      + DP[mb]*cd_.log_prob(TG[mb]) + placeable[mb]*pd_.log_prob(PTG[mb]))
            ratio = torch.exp(new_lp - LP[mb])
            pg = -torch.min(ratio*ADV[mb], torch.clamp(ratio, 1-CLIP, 1+CLIP)*ADV[mb]).mean()
            ent = ad.entropy().mean() + bdd.entropy().mean() + ed.entropy().mean() + dd.entropy().mean()
            loss = pg + VF_COEF*((value - RET[mb])**2).mean() - ENT_COEF*ent
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
    ma = sum(recent)/len(recent); val = run_val()
    nact = EX.mean().item() + (AT > 0).float().mean().item() + (BD > 0).float().mean().item() + DP.mean().item()
    print(f"update {upd:3d} (ep {(upd+1)*BATCH_EP:4d}): ma20 {ma:+.3f}, VAL {val:+.3f}, acts/step {nact:.2f}, reserve {torch.sigmoid(RR.mean()).item():.2f}")
    sys.stdout.flush()

torch.save(net.state_dict(), SAVE_PATH)
proc.stdin.close(); proc.terminate()
print(f"saved structured multi-action policy to {SAVE_PATH}")
