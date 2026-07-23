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

## Step 13 — Economy: build cities
**Added:** a 5th action `buildCity` (`ConstructionExecution` of a `City` near our spawn via
`me.canBuild(...)`) and a `cities` observation feature. Policy now `10 inputs → 5 actions`.
**Verified it matters:** a single city ~doubled troop count (300k → 562k) — cities boost
troop growth, so this is a real economy lever.

**Important limitation (the "where" problem):** the scalar policy only chooses *when* to
build; the *placement* is a hardcoded heuristic (near spawn). It literally cannot see or
reason about *where* to place buildings — e.g. along rail lines. Learning placement needs
the spatial upgrade: (1) add a rails/stations channel to the spatial observation, (2) swap
the scalar MLP for a **CNN** over the map channels, (3) give it a **spatial action head**
(output a probability map over tiles = "build HERE"). That's the standard AlphaStar-style
technique — a bigger jump (larger action space, more compute), but not far-fetched. It's
the natural next architectural milestone.

## Step 14 — Full action toolkit (10 actions)
**Added:** the rest of the land-relevant moves so the agent has (most of) what a Nation has:
`attackStrongest`, `buildDefensePost`, `buildMissileSilo`, `buildSAMLauncher`, and
`launchNuke` (atom bomb at the strongest enemy, gated on owning a missile silo). Plus a
`hasSilo` observation feature. Policy is now **11 inputs → 10 actions** (hidden layer bumped
to 16). Verified all APIs fire (nuke launched at tick 220 in a test); pricey structures are
naturally gated by gold via `canBuild`. Naval (ports/boats/warships) skipped — the training
map is all land; add them when training on a water map.

**Expected trade-off:** 10 actions is a *big* space for a (1+1) evolution strategy to
explore with a tiny MLP — training gets slower and noisier, and it may not master the
advanced moves (nukes need the build-silo-then-launch sequence). That difficulty is itself
the argument for the next era: a CNN policy + spatial action head + **gradient-based**
learning (PyTorch), which explores large action spaces far more efficiently.

---

## Git workflow (how we commit going forward)
One commit per step. After each step:
```
git add -A
git commit -m "Step N: <short title>" -m "<why + result, mirroring the DEVLOG entry>"
git push
```

## Step 15 — Full world map + boats + time caps
**Changed:** training and watching now run on the **full world map** (2000×1000, ~70% water)
with `map4x` as the mini-map (needed for water pathfinding / nav-mesh). Added two naval
actions — **`boatAttack`** (transport ship to the weakest enemy across water, via
`TransportShipExecution` gated by `canBuildTransportShip`) and **`buildPort`** (on an owned
shore tile) — plus a **`coastal`** observation feature. Policy is now **12 inputs → 12
actions**. Games **end when the agent dies**, or at `MAX_TICKS` (12000), or a wall-clock
`MAX_GAME_MS` safety cap. Agent spawns at the nearest unowned land to a seed-varied point,
so it learns from different continents. `run_agent.ts` now **loads `data/best_weights.json`**
so you watch the *trained* agent.

**Note:** the create-only mount silently truncated several files during incremental Edits;
switched to writing whole files via `cp` and verified brace balance after each.

## Step 16 — Tribe-heavy, dense reward, faster training, visible agent
- **Ratio:** flipped to **6 nations / 30 tribes (5:1)** — tribes are the farmable early game.
- **Reward shaping:** added a **survival-time** term (`0.5 * ticksAlive/MAX_TICKS`) so the
  reward varies continuously instead of sitting at a flat ~0.005. A policy that lives longer
  now scores measurably higher — the gradient the evolution strategy was missing.
- **Speed:** training now uses the `map4x` world (1000×500, still ~68% water) instead of the
  full 2000×1000 — ~4× fewer tiles, ~4× faster. (The "40ms" was just `GameImpl` construction;
  a full-world game is ~1–2s, so ~300 games = ~10 min.) `run_agent.ts` stays on the full world
  for watching.
- **Visible agent:** the pooled replay was too coarse to show the tiny agent (so the viewer
  said "eliminated" while it was alive). Fix: **stamp the agent's own tiles onto the pooled
  grid** each frame, so it always appears (white) while alive.
- **Note:** changing obs/action dims means old weights are incompatible — **retrain** to
  regenerate `data/best_weights.json` before watching.

## Step 17 — Learned troop commitment + land-weighted reward + denser map
**The key fix:** troop commitment was a *hardcoded* fraction (expand=troops/2, attack=troops/3),
so the agent had no "how much" lever. With every move dumping a fixed huge chunk of the army,
early expansion was suicidal, so the learned-optimal behavior was passivity (wait until forced).
Now the policy outputs an **extra sigmoid** = a troop-commitment fraction (0.1–0.95); the agent
learns *how much* to commit, not just what to do. `commit = troops * troopFraction` drives all
attacks/expansions/boats. (`Policy.nOut = nActions + 1`.)
**Reward:** territory weighted **5×** (`5*peakLandShare + 0.3*survival + ...`) so grabbing
wilderness is clearly worth it — pushes the agent to expand instead of turtle.
**Population:** bumped to **10 nations + 50 tribes** (was 6/30) — map felt underpopulated.
**Result:** validation reward now sits in the ~0.2 range (was 0.005) — real signal to climb.
**Retrain required** (network output grew, so old weights are the wrong size).

## Step 18 — PyTorch era, part 1: the environment bridge
**Why:** to use gradient learning, Python (PyTorch) must *drive* the game. So the TS sim
becomes a Gym-style **environment server**.
**Added:** `src/env/env_server.ts` — a Node process speaking line-delimited JSON over stdio:
`{"cmd":"reset","seed":N}` starts a fresh game (world map4x, 10 nations + 50 tribes) and
returns the 12-number observation; `{"cmd":"step","action":k,"troop":f}` applies the move,
advances 20 ticks, and returns `{obs, reward, done}`. Reward is now **per-step and dense**:
`5*(change in land share) + small survival bonus`, plus terminal win/death bonuses — much
better for gradient credit assignment than the old per-episode score. Engine console output
is silenced so stdout carries only JSON.
**Added:** `train_torch/drive_random.py` — spawns the env and drives 3 episodes with random
actions (no PyTorch). **Verified:** Python drove full games over the pipe; random policy gets
negative reward (loses land), exactly as expected. The bridge works.
**Next:** a PyTorch policy that reads obs and outputs action + troop fraction (part 2), then
REINFORCE/PPO to actually learn (part 3).

## Step 19 — PyTorch era, part 3: REINFORCE (and two env bugs it exposed)
**Added:** `train_torch/reinforce.py` — vanilla policy gradient. Per step we collect
`(log_prob, reward)`; per episode we compute discounted returns-to-go (γ=0.99); we pool a
**batch** of episodes and normalize returns *across the batch* (baseline), then update with
`loss = -Σ logπ · Â`. The discrete action is sampled from a `Categorical`; the troop fraction
is now sampled from a `Normal` (learned std) so it, too, gets a gradient — the "how much" lever.
Also `train_torch/diagnose.py` — loads the trained policy and prints WHAT it does (action
histogram + decoded per-step trajectory).

**Lesson 1 (the baseline bug):** the first version normalized returns **per-episode**, which
whitens every game to zero-mean — erasing the across-game signal (a +0.6 survival and a −0.2
death produce identical-magnitude gradients). Fix: pool a batch and normalize across it, so
good games get positive advantages and bad games negative.

**Lesson 2 (density):** the env was fabricating **10 random nations + 50 tribes** — ~2× too
sparse *and* nations at fake positions. Switched to the **real 61 world-manifest nations**
(coords scaled to map4x, snapped to land), counts configurable via env. Default is now
**density-matched to a real full-world game** (~1 player / 1400 land tiles → 15 nations +
100 tribes on map4x's 157,860 land tiles). Behaves like a real game: hundreds spawn, then
consolidate to ~40 survivors. Fast: ~0.035 s/step even with 273 alive.

**Lesson 3 (the build bug — why it "won't build cities"):** the behavioral trace showed the
agent *does* get rich (gold reaches **1.9M**; the obs merely caps display at 200K), yet every
build silently failed. Cause: the env built at the fixed initial `spawn` tile, but the engine's
`validStructureSpawnTiles` requires the target tile be **currently owned**, and by mid-game the
spawn tile is captured. Fix: build on a currently-owned tile. Verified — the agent now builds
up to 3 cities as it grows, and the trained policy builds 1–2 in surviving games.

**Result:** with the fixed env, REINFORCE shows a **weak early climb** (ma20 −0.13 → +0.045
over ~150 episodes) then oscillates back toward 0 — the classic high-variance REINFORCE
pattern (no trust region / value baseline). Economy is now reachable *and* functional, so the
remaining problem is the learner, not the environment. **Next (part 4):** PPO with a
value-function baseline for lower-variance, sustained learning; optionally light reward shaping
toward economy.

## Step 20 — Momentum reward (why it turtled and wouldn't build)
**The diagnosis:** watching the trained agent, it would grow early then *stop* — no more
cities, attacks, or boats — and cling to life behind indiscriminate alliances. The reward was
the culprit, in two ways: (1) `5*(Δ land share)` **telescopes to `5*(final − initial)`**, so
dying (final share = 0) **claws back every tile it ever gained** — growing big then dying nets
~0; (2) the flat `+0.001/step` survival bonus sums to **+0.6 over the 600-step cap**, dwarfing
the land term (shares ~0.01–0.05). Net effect: a 42-tile turtle that survives to the cap
scored *higher* than a 2,500-tile empire that died. We were **paying the agent to be passive**,
and building (125K gold, delayed/diffuse payoff, then clawed back) never got credited.

**The fix (per-step reward v2):** reward **momentum**, not mere survival —
`3*(Δ land share) + 0.2*(new city built)` per step, and at game end **bank the peak**:
`+5*peakShare` (so growth counts even if it later dies), `+5` decisive win, `−0.15` death.
Dropped the survival term entirely. Peak-banking makes aggression stop being suicidal; the
per-city bonus bootstraps economy that the weak REINFORCE credit chain missed on its own.

**Result:** reward climbs and **sustains** (ma20 −0.03 → +0.33 by ep 150, settling ~+0.2 —
the first real, non-oscillating learning curve). Behavior flips: the action histogram is no
longer flat — `attackWeak / attackStrong / buildCity` become the top actions, and the agent
reliably builds a city. **New bottleneck it exposed:** it grows to ~4,300 tiles then gets
**ganged down to a sliver** — it learns to *take* land but not *hold* it, and it can't place a
2nd city (`structureMinDist` on a tiny territory). This is where alliance management bites:
auto-accepting every offer + requesting alliances from *weaker* neighbors removes its own prey
and invites backstabs. **Next:** make alliances a *learned* decision (accept/reject as an
action + a relative-strength observation) so it can learn to ally *up*, not indiscriminately.

## Step 21 — Relational senses + learned alliances + variance reduction
**Why:** the agent could *see* only global summaries, so it couldn't reason about *who* to
trust or how outnumbered it was — and alliances were hardcoded (auto-accept everything,
request everyone), so there was nothing for the gradient to learn about diplomacy.

**Observations 12 → 16** (each earns its seat by enabling a decision): strongest-neighbor troop
ratio, **total border pressure** (Σ all neighbor troops ÷ mine — "how outnumbered am I"), ally
backing (Σ ally troops ÷ mine), and **offerer strength** (strength of whoever's offering an
alliance, so it can learn to ally *up*). Also **gold is now log-scaled** `ln(1+g)/ln(1+25M)`
instead of `÷200K` — it was saturating at 1.0 by the first city, blind to the 125K→25M range
that actually governs what you can afford.

**Alliances as a learned decision:** removed the unconditional auto-accept; `acceptAlliance`
is now its **own action** (the policy learns *when* to accept, conditioned on the offerer-
strength obs), and `requestAlliance` only targets **stronger** players (forced "ally up" — a
scalar net can't pick *which* player, same limitation as boat/attack targeting; only the accept
side is truly learnable). Repurposed the dead `wait` slot, so action count stays 12.

**Learner hygiene:** hidden layer **16 → 24**, **L2 weight decay** (1e-4) to shrink features
that aren't pulling their weight, a **held-out validation** readout (3 unseen seeds) to catch
overfitting, and a **32-seed cycled training pool** (each world seen ~30× over 1000 episodes)
to cut spawn-luck variance while still generalizing. Repro: `EPISODES=1000 BATCH=10 POOL=32`.

**Result:** best agent so far. Training reward climbs and sustains (ma20 ~+0.29 → +0.56 over
the back half); **held-out val holds ~+0.2 while training rises — generalizing, not overfitting**.
Peak territory **~2.7× larger** (≈4,300 → ≈11,750 tiles); economy (`buildCity`/`buildSilo`) and
the new alliance actions are firmly in play. **Still open:** it grows big then gets ganged down
and dies (~tick 2,800) — takes land but doesn't *hold*/compound it — and stays at **1 city**
(`structureMinDist` leaves no room for a 2nd on fragmented land). **Next:** PPO (value-function
critic) for the long build→grow→hold credit chain and spawn-luck discounting; and solving the
building/placement limit (the "where" problem — spatial action head).

## Step 22 — Building placement fix (the 1-city ceiling) + watch validation games
**The ceiling:** the agent could afford many cities but stayed stuck at **one**, because the build
helper always handed `canBuild` the *first* owned tile — right next to the existing structure — and
`structureMinDist` rejects anything that close. Every build stacked on the same spot, so only the
first city ever placed. This was a heuristic bug, not a gold or learner problem.

**The fix:** `buildTile()` now scans a sample of owned tiles and returns the one **farthest from any
existing structure** (max-min distance), spreading builds across the territory. `canBuild`'s local
radius-15 search then finds a valid, well-spaced spot. Verified: forcing city-building now yields
**up to 8 cities** as land grows (was hard-capped at 1). Applied to both `env_server.ts` and
`run_agent.ts`. (Gotcha: `Player.units(...)` only reads the first 3 type args unless you pass an
**array** — `units([City, DefensePost, …])` — else it silently misses structure types.)

**Result:** retrained (same 1000-ep / 32-seed config); the multi-city economy **≈doubled reward**
(train ma20 ~+0.56 → ~+1.1) and held-out val now spikes to **+5.2 / +2.3** — i.e. it *wins* some
held-out worlds (the +5 bonus fires at ≥80% land). It builds 3–8 cities and survives longer. Still
imperfect at holding on the hardest worlds, but a clear, large improvement.

**Also — watch the *actual* validation games:** `run_agent.ts` previously played one fixed
center-spawn world that was neither a training nor a validation seed. It now takes a `SEED` and
replicates `env_server`'s exact reset (gameID `env_<seed>`, seed-varied spawn), so `SEED=90001
npm run agent` plays held-out world #1 — you can *see* the games behind the val spikes. (World +
policy match the env; the exact sampled moves differ since JS vs torch use different RNGs.) The
diagnostic (`diagnose.py`) now also reports on held-out seeds (90001+), not training-pool worlds.

## Step 23 — PPO era: value critic + The Box map + territory-dominant reward (it wins)
**PPO (`train_torch/ppo.py`):** REINFORCE + three upgrades — a **value-function critic** (GAE
advantages → low variance, discounts spawn luck, credits the long build→grow→hold chain), **sample
reuse** (K epochs of minibatch updates per batch), and a **clipped trust region** (stable, no
climb-then-collapse). Actor keeps the same `fc1→fc2` shape so `export_weights.py`/`run_agent.ts`
still work; the critic is an extra head (`diagnose.py` loads `strict=False` to ignore it). Added a
13th action **`buildFactory`** and made economy structures = city+port+factory.

**The reward lesson (a real detour):** we first *directly* rewarded economy (`log(structures)`) and
troop-capacity (`peak-troops`). It **backfired** — the agent maxed those on a tiny base and turtled
(scored +3 owning <3% of the map). Taking land is hard and pays little until you're big; building
economy is easy and pays on any base — so a directly-rewarded economy becomes the *goal*, not the
means. **Fix:** territory-DOMINANT reward — `4·Δshare`, terminal `8·peakShare`, `+10` win; economy
kept only as a **small uncapped log bonus** (rewards spending even late, can't dominate); dropped
peak-troops entirely. Now PPO learns economy as the *means* to territory.

**The map (thanks to the user's push):** all-land squares consolidate too fast when small
(big_plains 200×200 → games end in ~50 ticks, no room). Switched training to **"The Box"**
(`MAP=box`, resources/maps/thebox, map16x 512×512, all-land) with **sparse** density (6 nations +
15 tribes) so its size actually functions as room. ~100–400-step games at ~1.5 s each — **~5× faster
iteration than the world map** (25 min vs 2 h per 1000 eps). `MAP` env var also supports `world`.

**Result — the payoff of the whole arc:** on held-out worlds the agent now **conquers the entire
map** — land climbs `7% → 28% → 59% → 82% → 100%`, building 8 cities *as it expands*, for **+22
reward** (2 of 3 held-out seeds = total wins; greedy replays survive full games at ~53% of the map).
Economy is finally a *means* to territory, not a turtle. `run_agent.ts` gained `MAP=box` support and
switched to **greedy** action selection (the strong policy plays best deterministically; the JS
sampler didn't match torch). **Next:** learned alliance management (break-alliance action + relational
obs) and the spatial "where" upgrades remain the open architectural frontiers.

## Step 23c — Realistic density + Step 24 — Learned diplomacy (candidate scoring)
**Density (23c):** the sparse Box let the agent "win" by outlasting ~21 opponents (`aliveP<=1`)
without taking 80% — a bad habit. Made nation counts unbounded (fabricate beyond the map's 13) and
set box default to **60 nations + 400 tribes (~460 players)**. Now it can't cheese it (forced-optimal
peak ~5%); games end by elimination in the crowd. The trained agent builds economy under pressure
(15–36% empires with cities/factories then overwhelmed), dies most games, and learned **measured
troop commitment** (0.55→0.22) — transferable skill.

**Diplomacy via candidate scoring (24) — the first move past the scalar MLP.** The MLP picks an
action *type* but never *which* player. Fix (Tier A, candidate scoring): the env emits up to 6
**candidate players** each step (offerers→accept, allies→break, bordering enemies→request) with 7
relational features; the policy got a **candidate-scoring head** (`7→16→1`, softmax→sampled target),
and PPO includes the target log-prob in the ratio **only on diplomacy steps** (conditional sub-action).
**Break-alliance is now a real learnable action** (`me.breakAlliance`). The old accept-all/request-all
heuristics collapsed into one learned `diplomacy` action + a `wait`.

**Result — machinery works, but the agent barely uses it.** Verified the mechanism fires
(accept/request/break all reachable), but the trained policy picks diplomacy only 0–3×/game (mostly
requests). **Why:** we made the reward territory-ONLY with no survival term (to kill turtling) — but an
alliance's payoff *is* survival/protection, which now has no reward hook, so PPO down-weights diplomacy
to ~zero (plus a chicken-and-egg: rare diplomacy → untrained candidate head → stays unrewarding). Same
recurring lesson: PPO uses what the reward rewards. The capability is built; making it *matter* needs
alliances to pay off — longer games (Box `map4x`) where protection buys time to expand, and/or a mild
survival/coordination incentive. `diagnose.py` now reports accept/request/break counts; `run_agent.ts`
uses a heuristic for the diplomacy action (no candidate head in the TS viewer yet).

## Step 25 — Learned structure placement (candidate scoring, part 2)
Reused the diplomacy candidate-scoring pattern for the "where" of building. The env emits up to 8
**owned-tile candidates** per step with 4 tactical features (frontline, near-own-structure, interiorness,
spacing); the policy got a **type-aware placement head** (`[tile feats + structure-type one-hot] -> 16
-> 1`) so it can learn different rules per structure (defense->frontline, factory->interior). PPO adds
the placement log-prob to the ratio only on defense/silo/SAM/factory builds. City/port/nuke stay heuristic.

**Result:** the machinery works end-to-end and placement is actively used — the agent builds many
factories (16-20/game) and places them at ~0% frontline (interior/safe, which is correct). Reward is
strong (held-out seed 90001 = +22.75, a near-total win). **But** the behavioral payoff is limited by the
same wall as diplomacy: **PPO only exercises what the reward values.** Defense posts are barely built
(0-3/game, unrewarded) so their placement head is starved; and the factory "interior" signal is
suggestive but not conclusive (frontline tiles are rare in the candidate pool regardless). Diplomacy
still dormant. Takeaway: candidate scoring is a proven, general mechanism for learned who/where, but to
make *defense* placement and *alliances* actually matter, the reward has to value holding/survival --
otherwise the capabilities sit unused. `run_agent.ts` still uses the spread heuristic for placement.
