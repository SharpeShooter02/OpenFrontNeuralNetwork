# Setup — OpenFrontNN Phase 0

Run these on your machine (Windows PowerShell shown; the equivalent works on macOS/Linux).
You need **Node 20+**, **npm**, and **git** installed.

## One-time setup

```powershell
# from inside the OpenFrontNN folder
git init

# add the OpenFront engine as a pinned submodule
git submodule add https://github.com/openfrontio/OpenFrontIO.git vendor/OpenFrontIO

# pin to the exact engine commit this harness was verified against (reproducibility)
git -C vendor/OpenFrontIO checkout 774d98ddad3123a7d12709ae029c853aea1b0039

# install the engine's runtime deps.
# --ignore-scripts is REQUIRED: the engine's "prepare" hook runs husky, which
# fails outside its own repo and would abort the install.
cd vendor/OpenFrontIO
npm install --ignore-scripts
cd ../..

# install our own dev deps (tsx, to run TypeScript directly)
npm install

# commit the scaffold + submodule pointer
git add -A
git commit -m "Phase 0: headless harness + pinned OpenFront submodule"
```

Or just run the helper: `./scripts/setup.ps1` (Windows) or `bash scripts/setup.sh`.

## Run the harness

```powershell
npm run harness
```

Expected output (deterministic — same tick count every run):

```
Map: 100x100, land tiles: 10000
tick 0: 0 alive, leader - @ 0% land
tick 1000: 10 alive, leader Norwegian Brotherhood @ 20.6% land
...
DONE in 7721 ticks / ~0.3s. Winner: Burmese Patriarchy (100.0% land)
```

A full 15-bot game plays to completion in about a third of a second, no browser.

## What this proves (Phase 0 done)

- The OpenFront engine runs **headless** (pure simulation, no rendering).
- It is **deterministic** (identical tick count and winner across runs) — essential for reproducible RL experiments.
- It is **fast** (~7700 ticks in ~0.3s), so training rollouts are cheap.
- We can **construct a game, add AI bots, step it tick-by-tick, and read state** (`player.isAlive()`, `numTilesOwned()`, `numLandTiles()`) — the foundation for observations, actions, and rewards.

## What's next (Phase 1)

Insert an *agent execution* that, each decision tick, serializes an observation
(the map's ownership grid + our scalar stats) and applies one action by adding an
`Execution` (e.g. `AttackExecution`). Then bridge to a Python Gymnasium wrapper over
a local socket so PyTorch can drive it. See `openfront_rl_design.md` for the full plan.

## Notes

- The engine is **AGPLv3**. It stays isolated in `vendor/OpenFrontIO` as a submodule;
  your code in `src/` is kept separate.
- To update the engine later: `git -C vendor/OpenFrontIO checkout <newer-commit>` then
  re-run `npm install --ignore-scripts` inside it. Pin deliberately — don't float.
