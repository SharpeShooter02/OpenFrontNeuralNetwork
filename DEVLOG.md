# DEVLOG — building an OpenFront bot, step by step

A running journal of *what* we built and *why*. Newest entries at the bottom.
Each future step gets its own git commit whose message mirrors the entry here, so
`git log` and this file tell the same story.

---

## Phase 0 — Prove the engine runs headless
**Files:** `src/harness/run_headless.ts`, `SETUP.md`, `scripts/`, `package.json`

We pinned the OpenFront engine as a git submodule (`vendor/OpenFrontIO`) and wrote a
script that loads a small map, spawns AI bots, and steps the game with
`game.executeNextTick()` until a winner emerges — no browser, no rendering.

**Why:** everything later (observations, actions, training) wraps this loop. We had to
prove the engine could run as a pure, fast, *deterministic* simulation first.
**Result:** a 15-bot game finishes in ~0.3s and is deterministic (same tick count every
run) — the reproducibility RL depends on.
**Lesson:** new executions are `init()`-ed one tick and `tick()`-ed the *next* — that
off-by-one caused a false "game over" until we added a spawn guard.

## Visualizer — watch games play out
**Files:** `src/harness/record_replay.ts`, `viz/index.html`

The recorder snapshots tile ownership into `viz/replay.js`; the HTML page animates it on
a canvas (play/pause/scrub, colored by territory, live scoreboard).

**Why:** metrics hide dumb behavior; *watching* the bot is how we'll debug the learner.
**Lesson (delta encoding):** storing the whole 10k-tile map every frame made a 3.4 MB
file (and crashed editors). Storing only *changed* tiles per frame cut it to ~0.5 MB with
identical visuals. Replays are opt-in and overwrite one file — training games are never
saved, so there's no storage blow-up.

## Step 1 — A player we control (no brain)
**File:** `src/harness/run_agent.ts` · run: `npm run agent`

Added one `Human` player we hold a reference to, spawned it among a few bots, and each
tick just *read its state*. It sits still and gets eaten around tick ~900.

**Why:** you can't give a bot a brain until you can (a) hold "our" player and (b) observe
it. The sitting-duck death proves both — and shows exactly why a brain is needed.

## Step 2 — First action: expand
Added a decide+act block: every 20 ticks, expand into empty land by attacking
"terra nullius" (unowned land) with half our troops.

**Why:** this is the control *seam* — look, decide, act — where a learned policy will
later plug in. **Result:** tiles jumped from 52 to thousands and the agent survived the
whole game. But its territory slowly bled once empty land ran out — it could grab land
but not fight players.

## Step 3 — Attack a neighbor (and a big lesson)
Added "attack the weakest bordering player" to the policy.

**Naive version** (throw half the army at a neighbor every 20 ticks) made it *worse* — it
bled its army dry and died at tick 1458. **Guarded version** (attack only if we have 2×
their troops, commit 1/3) made it **win the whole map**.
**Lesson:** the gap between dying and dominating was two hand-picked numbers. Hand-tuning
strategy is brittle and doesn't generalize — this is the concrete argument for *learning*
the policy instead of coding it. Caveat: we're only beating the easy "tribe" bots, which
don't build structures or spend gold.

## Step 4 — Observation (scalars)
**Added:** `observe()` — turns the live game into a fixed vector of 7 normalized numbers
(land share, troops, gold, enemies alive, empty-land-adjacent, enemy neighbors, troop
ratio vs weakest). Printed each interval so the board-as-numbers is visible.

**Why:** the policy network never sees the game object — only this vector. It defines what
the bot is *allowed to know*. Values are scaled toward 0..1 ("normalization") because nets
train badly on wildly different scales.
**Limitation:** these are global summaries — they say "boxed in by 4 enemies" but not
*where*. Deciding *where* to act needs the spatial observation (the map as image channels)
— that's the next step.

## Step 5 — Observation (spatial channels)
**Added:** `observeSpatial()` — the whole map as a `[3, H, W]` tensor of binary grids:
"mine", "enemy", "neutral". Plus `printSpatial()` to eyeball it as shrunk ASCII.

**Why:** the scalar vector only knew our border — it couldn't say *where* anything was.
The spatial tensor gives full-map vision (including players we aren't touching), which is
what a CNN reads to decide *which direction* to act. We used the all-land `plains` map so
we could skip a water/terrain channel for now.
**Note:** each symbol (#/O/.) is its own 0/1 channel; the ASCII is shrunk 3× for display
but the real tensor is full resolution. Buildings/terrain/strength would each be a new
channel later.

## Step 6 — Complete the vision (8 channels)
**Changed:** `observeSpatial()` now returns 8 channels: mine, hostile, allied, neutral,
blocked (water/terrain), fallout (nuke damage), myStruct, enemyStruct. `printSpatial()`
shows the richer map and prints per-channel sums.

**Why:** we're about to face Nations, which build structures, nuke, and form/break
alliances. The bot can't strategize about those unless it can *see* them, so we gave it a
channel for each ahead of time. On the current all-land plains + tribe setup, the
`blocked/fallout/struct/allied` channels read 0 — they'll light up once we switch to
Nations on a real (water-containing) map. That opponent switch is the next step.

## Step 7 — The real environment: world map, Nations + tribes
**Changed:** `run_agent.ts` now runs the **world** map (2000×1000) with **61 Nations**
built from the map manifest, plus 20 tribes. Replay is **pooled** to a fixed ~223×112
grid (the raw 2M-tile map is too big to store or feed a network). Added
`useDefineForClassFields: false` to `tsconfig.json`.

**Three gotchas solved:**
1. Nations spawn *only during the spawn phase* — we now run ~150 spawn ticks, then end it.
2. A Human spawning in Singleplayer *ends the spawn phase*, so our agent is added **after**
   the Nations/tribes have placed.
3. `NationNukeBehavior` relies on legacy class-field init order — `tsx`/esbuild needs
   `useDefineForClassFields: false` or it crashes (`this.game` undefined). **The real trap:**
   a `"include": ["src/**/*.ts"]` in `tsconfig.json` scopes that compiler option to `src/`
   only, so the vendored engine under `vendor/` never gets it and crashes. **Fix: remove
   `include`** so the option applies to every transpiled file, engine included. (Also set
   `"target": "ES2020"` to match the engine.) Reproduced in a faithful mini-repo:
   with `include` → crash; without → Nations spawn and build normally.

**Result:** Nations build structures (0 → 183) and eat the tribes; the previously-dark
vision channels light up (`blocked` = ocean = 1.35M tiles, `enemyStruct` = 12 buildings).
Our hand-policy that *crushed* the tribes now **dies by ~tick 2000** against Nations — the
concrete motivation for learning a policy instead of hand-tuning one.

## Step 8 — The policy network (untrained)
**Added:** `src/agent/policy.ts` — a tiny hand-written MLP (7 inputs → 8 hidden (tanh) →
3 outputs → softmax). `run_agent.ts` now calls `policy.forward(observe().values)` to pick
the action (`expand` / `attackWeakest` / `wait`) instead of the `if/else` chooser.

**Why:** this is *the* seam a learned brain plugs into. The decision now flows
`observation → network → probabilities → action`. With **random weights** it plays badly
(favors "wait", dies fast) — which is exactly the point: it's bad only because the weights
are random numbers we haven't trained yet. A "neural network" here is just multiply-add-bend
repeated; you can read every number in `policy.ts`.
**Next:** a reward signal (score a game), then a training loop (nudge the weights toward
what won).

## Step 9 — Reward (scoring a game)
**Added:** `src/agent/reward.ts` — `computeReward({peakLandShare, survived, won})` =
peak land share + 0.5 for surviving + 5 for winning. `run_agent.ts` tracks peak land and
prints `REWARD` at game end.

**Why:** the reward is the single number training maximizes — whatever it rewards is what
the agent learns, so it must match what we actually want (get big, survive, win). The
untrained policy scores ~0.0001 (a speck that dies instantly); training's job is to push
that number up.
**Design note for next step:** on the *world* map our agent is a speck and winning is
near-impossible, so the reward signal is weak/sparse there. Training will likely work far
better first on a *small* map (plains + tribes) where the agent can actually grow and win,
giving a meaningful reward gradient — then scale up.

## Step 10 — Training loop (it learns!)
**Added:** `src/harness/train.ts` (run: `npm run train`) plus `getFlat`/`setFlat` in
`policy.ts`. Uses a **(1+1) evolution strategy** on the small plains map vs tribes: mutate
the weights, play a game, keep the mutation only if its reward is higher. No gradients/backprop.

**Result:** reward climbed from **0.0052** (random weights — a helpless speck) to **6.5**
(full map + survived + won) in ~10 generations. The agent taught itself to play. Training
saves the weights to `data/best_weights.json` and records the trained agent's winning game
to `viz/replay.js` so you can watch it.

**Gotcha fixed:** the game-end check "≤1 player alive" fired at tick 0 (nobody spawned yet),
quitting every game instantly — same `started` guard as Phase 0.

**This completes the core arc:** observation → policy → reward → training. Everything else
is refinement: train on Nations/world, richer actions (boats/building), spatial CNN policy,
gradient-based learning, etc.

## Step 11 — Train vs Nations, with generalization
**Why:** the tribe-trained policy *overfit* — it learned to hoard troops behind tribe-scale
caps and strike once, which fails against Nations' bigger economies. Two overfitting leaks
fixed: (1) **opponent** — now trains vs injected Nations + tribes on the medium `big_plains`
(200×200) map; (2) **seed** — each candidate is now evaluated over several games with
different seeds and averaged, and progress is measured on **held-out** validation seeds, so
it learns general skill rather than one game's trick.

**How:** `train.ts` injects `NUM_NATIONS` Nation objects at seed-varied positions (Nations
need a spawn phase, so we run ~150 spawn ticks then end it, then drop in the agent), plus
tribes. (1+1)-ES compares best vs candidate on the *same* fresh seeds each generation.
**Result:** held-out validation reward climbs from ~0.001 to ~0.09–0.17 — the agent learns
to carve out territory against Nations (no more trivial wins; Nations are genuinely hard).
Saves `data/best_weights.json` and records a trained game to `viz/replay.js`.

## Step 12 — Diplomacy: alliances (action + observation)
**Why:** watching the trained agent, it would expand, grab an easy kill, then get
**backstabbed and ganged up on** by Nations — because it had no diplomacy. The Nations run
their full toolkit (attack, alliances, structures, nukes, MIRV, warships), so a 3-action
agent is hopelessly outmatched.

**Added:** (1) auto-accept incoming alliance offers each decision (free protection);
(2) a 4th action `requestAlliance` (ask all bordering enemies for a pact — Nations often
accept); (3) two observation features — number of allies and whether a request is incoming.
Policy is now `9 inputs → 4 actions`. Applied to both `train.ts` and `run_agent.ts`.

**Verified it matters:** with the same seeds, seeking alliances multiplied survival time
(e.g. death@720 → survived the whole game with 3 allies and 33% of the map; death@696 →
death@3618). Untrained baseline reward jumped 0.001 → ~0.19 from alliances alone.
**Note:** changing the observation/action size means old 7-input weights are incompatible —
retrain from scratch. **Next:** building (cities for economy).

---

## Git workflow (how we commit going forward)
One commit per step. After each step:
```
git add -A
git commit -m "Step N: <short title>" -m "<why + result, mirroring the DEVLOG entry>"
git push
```
