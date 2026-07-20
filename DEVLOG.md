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

---

## Git workflow (how we commit going forward)
One commit per step. After each step:
```
git add -A
git commit -m "Step N: <short title>" -m "<why + result, mirroring the DEVLOG entry>"
git push
```
