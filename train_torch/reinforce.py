# REINFORCE (vanilla policy gradient) over the OpenFront env bridge.
# For each episode we collect (log_prob, reward) per step, compute discounted returns,
# and update the policy so that decisions leading to higher return become more likely:
#   loss = -sum_t  log_prob_t * (G_t - baseline)
# The discrete action is sampled from a Categorical; the troop fraction is sampled from a
# Normal in logit space so it, too, gets a policy gradient (it's the key "how much" lever).
# Run from project root:  python train_torch/reinforce.py
import subprocess, json, os, sys
import torch, torch.nn as nn
from torch.distributions import Categorical, Normal

N_OBS, N_ACT = 12, 12
GAMMA = 0.99
LR = 3e-3
EPISODES = int(os.environ.get("EPISODES", "300"))
BATCH = int(os.environ.get("BATCH", "10"))   # episodes pooled per gradient update
MAX_STEPS = 2000
SAVE_PATH = os.path.join("data", "torch_policy.pt")

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_OBS, 16)
        self.fc2 = nn.Linear(16, N_ACT + 1)     # 12 action logits + 1 troop-fraction mean (logit space)
        self.troop_log_std = nn.Parameter(torch.tensor(-0.5))  # learned exploration width for troop
    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        out = self.fc2(h)
        return out[..., :N_ACT], out[..., N_ACT]

net = PolicyNet()
opt = torch.optim.Adam(net.parameters(), lr=LR)

def choose(obs):
    x = torch.tensor(obs, dtype=torch.float32)
    action_logits, troop_mean = net(x)
    dist = Categorical(logits=action_logits)
    action = dist.sample()
    tdist = Normal(troop_mean, net.troop_log_std.exp())
    troop_raw = tdist.sample()
    troop = torch.sigmoid(troop_raw)            # squash to (0,1) for the env
    logp = dist.log_prob(action) + tdist.log_prob(troop_raw)
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

def returns_to_go(rewards):
    G, out = 0.0, []
    for r in reversed(rewards):
        G = r + GAMMA * G
        out.append(G)
    out.reverse()
    return torch.tensor(out, dtype=torch.float32)

recent = []
batch_logps, batch_G = [], []          # pooled across BATCH episodes before each update
for ep in range(EPISODES):
    o = rpc({"cmd": "reset", "seed": ep})["obs"]
    logps, rewards = [], []
    done, steps, total = False, 0, 0.0
    while not done and steps < MAX_STEPS:
        a, tr, logp = choose(o)
        r = rpc({"cmd": "step", "action": a, "troop": tr})
        o = r["obs"]; done = r["done"]
        logps.append(logp); rewards.append(r["reward"])
        total += r["reward"]; steps += 1

    batch_logps.extend(logps)
    batch_G.append(returns_to_go(rewards))     # keep raw returns; normalize across the whole batch

    recent.append(total); recent = recent[-20:]
    ma = sum(recent) / len(recent)
    print(f"episode {ep:3d}: {steps:4d} steps, reward {total:+.3f}, ma20 {ma:+.3f}")
    sys.stdout.flush()

    if len(batch_G) >= BATCH or ep == EPISODES - 1:
        G = torch.cat(batch_G)
        G = (G - G.mean()) / (G.std() + 1e-8)  # baseline ACROSS episodes -> good games get +adv, bad -adv
        loss = -(torch.stack(batch_logps) * G).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        print(f"  -- update @ ep {ep}: loss {loss.item():+.3f}, batch_steps {len(G)}")
        batch_logps, batch_G = [], []

torch.save(net.state_dict(), SAVE_PATH)
proc.stdin.close(); proc.terminate()
print(f"saved policy to {SAVE_PATH}")
