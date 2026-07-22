# PPO over the OpenFront env bridge, with a CANDIDATE-SCORING head for diplomacy.
# Upgrades over REINFORCE: value critic (GAE), sample reuse (K epochs), clipped trust region.
# Candidate scoring: when the policy picks the diplomacy action, a small head scores each candidate
# player (accept-offer / request-enemy / break-ally) from its features and samples WHICH one — so the
# agent *learns* who to ally with or betray, not a hardcoded "accept/request all". The target log-prob
# is included in the PPO ratio only on diplomacy steps (conditional sub-action).
# The actor's fc1->fc2 stays the same shape as policy.ts (export/run_agent still work); vf + cand are
# extra heads. Run from project root:  python train_torch/ppo.py
import subprocess, json, os, sys
import torch, torch.nn as nn
from torch.distributions import Categorical, Normal

N_OBS, N_ACT, N_HID = 16, 13, 24
FCAND, KCAND, DIPLO = 7, 6, 3     # candidate features, max candidates, diplomacy action index
GAMMA, LAMBDA = 0.99, 0.95
CLIP = 0.2
LR = 1.5e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 4
MINIBATCH = 512
ENT_COEF, VF_COEF = 0.01, 0.5
EPISODES = int(os.environ.get("EPISODES", "1000"))
BATCH_EP = int(os.environ.get("BATCH", "10"))
POOL = int(os.environ.get("POOL", "32"))
TRAIN_SEEDS = [1000 + i for i in range(POOL)]
VAL_SEEDS = [90001, 90002, 90003]
MAX_STEPS = 2000
SAVE_PATH = os.path.join("data", "torch_policy.pt")

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_OBS, N_HID)
        self.fc2 = nn.Linear(N_HID, N_ACT + 1)                        # actor: action logits + troop mean
        self.vf = nn.Linear(N_HID, 1)                                 # critic
        self.cand = nn.Sequential(nn.Linear(FCAND, 16), nn.Tanh(), nn.Linear(16, 1))  # per-candidate score
        self.troop_log_std = nn.Parameter(torch.tensor(-0.5))
    def forward(self, x):
        h = torch.tanh(self.fc1(x)); out = self.fc2(h)
        return out[..., :N_ACT], out[..., N_ACT], self.vf(h).squeeze(-1)
    def score_cands(self, cands):                                     # cands: [..., K, FCAND] -> [..., K]
        return self.cand(cands).squeeze(-1)

net = ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def policy_dists(action_logits, troop_mean):
    return Categorical(logits=action_logits), Normal(troop_mean, net.troop_log_std.exp())

def pad_cands(cands):                        # variable-length list of rows -> [K,F] tensor + [K] mask
    c = torch.zeros(KCAND, FCAND); m = torch.zeros(KCAND)
    for i, row in enumerate(cands[:KCAND]):
        c[i] = torch.tensor(row, dtype=torch.float32); m[i] = 1.0
    return c, m

def cand_dist(cpad, cmask):                  # masked Categorical over candidates
    scores = net.score_cands(cpad).masked_fill(cmask == 0, -1e9)
    return Categorical(logits=scores)

# ---- env bridge ----
def tsx_cmd():
    if os.environ.get("TSX"): return os.environ["TSX"].split()
    local = os.path.join("node_modules", ".bin", "tsx.cmd" if os.name == "nt" else "tsx")
    return [local] if os.path.exists(local) else ["npx", "tsx"]
ENV = os.environ.get("ENVFILE", "src/env/env_server.ts")
proc = subprocess.Popen(tsx_cmd() + [ENV], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
def rpc(msg):
    proc.stdin.write(json.dumps(msg) + "\n"); proc.stdin.flush()
    return json.loads(proc.stdout.readline())

def gae(rewards, values):
    adv = [0.0] * len(rewards); last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + GAMMA * next_v - values[t]
        last = delta + GAMMA * LAMBDA * last
        adv[t] = last
    return adv, [adv[t] + values[t] for t in range(len(rewards))]

def rollout_episode(seed):
    r0 = rpc({"cmd": "reset", "seed": seed}); o = r0["obs"]; cands = r0["cands"]
    O, A, TR, TG, CD, MK, LP, VAL, REW = [], [], [], [], [], [], [], [], []
    done, steps, total = False, 0, 0.0
    while not done and steps < MAX_STEPS:
        cpad, cmask = pad_cands(cands)
        with torch.no_grad():
            logits, tmean, value = net(torch.tensor(o, dtype=torch.float32))
            cat, tdist = policy_dists(logits, tmean)
            a = cat.sample(); traw = tdist.sample(); tgt = cand_dist(cpad, cmask).sample()
            logp = cat.log_prob(a) + tdist.log_prob(traw) + (cand_dist(cpad, cmask).log_prob(tgt) if int(a) == DIPLO else 0.0)
        troop = float(torch.sigmoid(traw))
        r = rpc({"cmd": "step", "action": int(a), "troop": troop, "target": int(tgt)})
        O.append(o); A.append(int(a)); TR.append(float(traw)); TG.append(int(tgt)); CD.append(cpad); MK.append(cmask)
        LP.append(float(logp)); VAL.append(float(value)); REW.append(r["reward"])
        o = r["obs"]; cands = r["cands"]; done = r["done"]; total += r["reward"]; steps += 1
    adv, ret = gae(REW, VAL)
    return O, A, TR, TG, CD, MK, LP, adv, ret, total, steps

def run_val():
    tot = 0.0
    for sd in VAL_SEEDS:
        r0 = rpc({"cmd": "reset", "seed": sd}); o = r0["obs"]; cands = r0["cands"]; done, steps, rep = False, 0, 0.0
        while not done and steps < MAX_STEPS:
            cpad, cmask = pad_cands(cands)
            with torch.no_grad():
                logits, tmean, _ = net(torch.tensor(o, dtype=torch.float32))
                cat, tdist = policy_dists(logits, tmean)
                a = int(cat.sample()); troop = float(torch.sigmoid(tdist.sample())); tgt = int(cand_dist(cpad, cmask).sample())
            r = rpc({"cmd": "step", "action": a, "troop": troop, "target": tgt}); o = r["obs"]; cands = r["cands"]; done = r["done"]; rep += r["reward"]; steps += 1
        tot += rep
    return tot / len(VAL_SEEDS)

NUM_UPDATES = EPISODES // BATCH_EP
recent = []
for upd in range(NUM_UPDATES):
    O, A, TR, TG, CD, MK, LP, ADV, RET = [], [], [], [], [], [], [], [], []
    for i in range(BATCH_EP):
        seed = TRAIN_SEEDS[(upd * BATCH_EP + i) % len(TRAIN_SEEDS)]
        o, a, tr, tg, cd, mk, lp, adv, ret, total, steps = rollout_episode(seed)
        O += o; A += a; TR += tr; TG += tg; CD += cd; MK += mk; LP += lp; ADV += adv; RET += ret
        recent.append(total); recent = recent[-20:]

    O = torch.tensor(O, dtype=torch.float32); A = torch.tensor(A); TR = torch.tensor(TR, dtype=torch.float32)
    TG = torch.tensor(TG); CD = torch.stack(CD); MK = torch.stack(MK); LP = torch.tensor(LP, dtype=torch.float32)
    ADV = torch.tensor(ADV, dtype=torch.float32); RET = torch.tensor(RET, dtype=torch.float32)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    N = len(O)

    for _ in range(EPOCHS):
        idx = torch.randperm(N)
        for s in range(0, N, MINIBATCH):
            mb = idx[s:s + MINIBATCH]
            logits, tmean, value = net(O[mb])
            cat, tdist = policy_dists(logits, tmean)
            cdist = cand_dist(CD[mb], MK[mb])
            use = (A[mb] == DIPLO).float()                            # target log-prob only on diplomacy steps
            new_logp = cat.log_prob(A[mb]) + tdist.log_prob(TR[mb]) + use * cdist.log_prob(TG[mb])
            ratio = torch.exp(new_logp - LP[mb])
            surr1 = ratio * ADV[mb]; surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * ADV[mb]
            pg_loss = -torch.min(surr1, surr2).mean()
            v_loss = ((value - RET[mb]) ** 2).mean()
            ent = cat.entropy().mean()
            loss = pg_loss + VF_COEF * v_loss - ENT_COEF * ent
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 0.5)
            opt.step()

    ma = sum(recent) / len(recent)
    val = run_val()
    ep = (upd + 1) * BATCH_EP
    print(f"update {upd:3d} (ep {ep:4d}): ma20 {ma:+.3f}, pg {pg_loss.item():+.3f}, v {v_loss.item():.3f}, VAL {val:+.3f}")
    sys.stdout.flush()

torch.save(net.state_dict(), SAVE_PATH)
proc.stdin.close(); proc.terminate()
print(f"saved policy to {SAVE_PATH}")
