# Proves the TS<->Python bridge: drives the env with random actions (no PyTorch).
# Run from the project root:  python train_torch/drive_random.py
import subprocess, json, random, os
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
    o = rpc({"cmd": "reset", "seed": ep}); total = 0.0; steps = 0; done = False
    while not done and steps < 2000:
        r = rpc({"cmd": "step", "action": random.randint(0, 11), "troop": random.random()})
        total += r["reward"]; done = r["done"]; steps += 1
    print(f"episode {ep}: {steps} steps, total reward {total:.3f}")
proc.stdin.close(); proc.terminate()
print("bridge OK")
