# Export the trained torch policy into the flat weight layout that src/agent/policy.ts's
# setFlat() expects, so run_agent.ts can play & record the trained brain in the visualizer.
# Layout: fc1.weight[16x12] flat, fc1.bias[16], fc2.weight[13x16] flat, fc2.bias[13].
# Run from project root:  python train_torch/export_weights.py
import json, os, torch

sd = torch.load(os.path.join("data", "torch_policy.pt"))
flat = []
flat += sd["fc1.weight"].flatten().tolist()   # [16][12]
flat += sd["fc1.bias"].tolist()               # [16]
flat += sd["fc2.weight"].flatten().tolist()   # [13][16]
flat += sd["fc2.bias"].tolist()               # [13]
out = os.path.join("data", "torch_weights.json")
json.dump(flat, open(out, "w"))
print(f"wrote {out} ({len(flat)} params)")
