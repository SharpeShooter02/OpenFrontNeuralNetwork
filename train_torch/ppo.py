# PPO (Proximal Policy Optimization) over the OpenFront env bridge.
# Upgrades vanilla REINFORCE (reinforce.py) with three things:
#   1. a VALUE head (critic) -> low-variance GAE advantages; discounts spawn luck and
#      credits the long build->grow->hold chain instead of blaming actions for bad draws.
#   2. sample REUSE -> K epochs of minibatch updates per batch of games (far more learning/hour).
#   3. a CLIPPED objective (trust region) -> stable, no climb-then-collapse.
# The actor is the SAME fc1(16->24)->fc2(24->13) shape as policy.ts, so export_weights.py and
# run_agent.ts keep working; the critic (vf) is an extra head that's ignored on export.
# Run from project root:  python train_torch/ppo.py
import subprocess, json, os, sys
import torch, torch.nn as nn
from torch.distributions import Categorical, Normal

N_OBS, N_ACT, N_HID = 16, 13, 24
GAMMA, LAMBDA = 0.99, 0.95        # discount, GAE smoothing
CLIP = 0.2                        # PPO trust-region clip
LR = 1.5e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 4                        # gradient passes over each collected batch
MINIBATCH = 512
ENT_COEF, VF_COEF = 0.01, 0.5     # entropy bonus (exploration), value-loss weight
EPISODES = int(os.environ.get("EPISODES", "1000"))
BATCH_EP = int(os.environ.get("BATCH", "10"))   # episodes collected before each update
POOL = int(os.environ.get("POOL", "32"))
TRAIN_SEEDS = [1000 + i for i in range(POOL)]
VAL_SEEDS = [90001, 90002, 90003]
MAX_STEPS = 2000
SAVE_PATH = os.path.join("data", "torch_policy.pt")

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_OBS, N_HID)
        self.fc2 = nn.Linear(N_HID, N_ACT + 1)          # actor: 12 logits + troop mean (matches policy.ts)
        self.vf = nn.Linear(N_HID, 1)                   # critic: state value
        self.troop_log_std = nn.Parameter(torch.tensor(-0.5))
    def forward(self, x):
        h = torch.tanh(self.fc1(x)); out = self.fc2(h)
        return out[..., :N_ACT], out[..., N_ACT], self.vf(h).squeeze(-1)

net = ActorCritic()
opt = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

def policy_dists(action_logits, troop_mean):
    return Categorical(logits=action_logits), Normal(troop_mean, net.troop_log_std.exp())

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
    # episodes always terminate here, so bootstrap value past the end is 0.
    adv = [0.0] * len(rewards); last = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else 0.0
        delta = rewards[t] + GAMMA * next_v - values[t]
        last = delta + GAMMA * LAMBDA * last
        adv[t] = last
    ret = [adv[t] + values[t] for t in range(len(rewards))]
    return adv, ret

def rollout_episode(seed):
    o = rpc({"cmd": "reset", "seed": seed})["obs"]
    obs_l, act_l, traw_l, logp_l, val_l, rew_l = [], [], [], [], [], []
    done, steps, total = False, 0, 0.0
    while not done and steps < MAX_STEPS:
        x = torch.tensor(o, dtype=torch.float32)
        with torch.no_grad():
            logits, tmean, value = net(x)
            cat, tdist = policy_dists(logits, tmean)
            a = cat.sample(); traw = tdist.sample()
            logp = cat.log_prob(a) + tdist.log_prob(traw)
        troop = float(torch.sigmoid(traw))
        r = rpc({"cmd": "step", "action": int(a), "troop": troop})
        obs_l.append(o); act_l.append(int(a)); traw_l.append(float(traw))
        logp_l.append(float(logp)); val_l.append(float(value)); rew_l.append(r["reward"])
        o = r["obs"]; done = r["done"]; total += r["reward"]; steps += 1
    adv, ret = gae(rew_l, val_l)
    return obs_l, act_l, traw_l, logp_l, adv, ret, total, steps

def run_val():
    tot = 0.0
    for sd in VAL_SEEDS:
        o = rpc({"cmd": "reset", "seed": sd})["obs"]; done, steps, rep = False, 0, 0.0
        while not done and steps < MAX_STEPS:
            with torch.no_grad():
                logits, tmean, _ = net(torch.tensor(o, dtype=torch.float32))
                cat, tdist = policy_dists(logits, tmean)
                a = int(cat.sample()); troop = float(torch.sigmoid(tdist.sample()))
            r = rpc({"cmd": "step", "action": a, "troop": troop}); o = r["obs"]; done = r["done"]; rep += r["reward"]; steps += 1
        tot += rep
    return tot / len(VAL_SEEDS)

NUM_UPDATES = EPISODES // BATCH_EP
recent = []
for upd in range(NUM_UPDATES):
    O, A, TR, LP, ADV, RET = [], [], [], [], [], []
    for i in range(BATCH_EP):
        seed = TRAIN_SEEDS[(upd * BATCH_EP + i) % len(TRAIN_SEEDS)]
        obs_l, act_l, traw_l, logp_l, adv, ret, total, steps = rollout_episode(seed)
        O += obs_l; A += act_l; TR += traw_l; LP += logp_l; ADV += adv; RET += ret
        recent.append(total); recent = recent[-20:]

    O = torch.tensor(O, dtype=torch.float32); A = torch.tensor(A)
    TR = torch.tensor(TR, dtype=torch.float32); LP = torch.tensor(LP, dtype=torch.float32)
    ADV = torch.tensor(ADV, dtype=torch.float32); RET = torch.tensor(RET, dtype=torch.float32)
    ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
    N = len(O)

    for _ in range(EPOCHS):
        idx = torch.randperm(N)
        for s in range(0, N, MINIBATCH):
            mb = idx[s:s + MINIBATCH]
            logits, tmean, value = net(O[mb])
            cat, tdist = policy_dists(logits, tmean)
            new_logp = cat.log_prob(A[mb]) + tdist.log_prob(TR[mb])
            ratio = torch.exp(new_logp - LP[mb])
            surr1 = ratio * ADV[mb]
            surr2 = torch.clamp(ratio, 1 - CLIP, 1 + CLIP) * ADV[mb]
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
