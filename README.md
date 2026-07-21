# OpenFrontNN

Reinforcement-learning experiments on the [OpenFront](https://github.com/openfrontio/OpenFrontIO) RTS engine — teaching a neural network to play the game from scratch.

The engine is pinned as a git submodule under `vendor/OpenFrontIO` (AGPLv3; we only read it). Our code lives in `src/` (the game-side harness/environment, in TypeScript) and `train_torch/` (the PyTorch learner, in Python).

## What we're trying to do

Take the OpenFront RTS — where players expand territory, build economy (cities), fight, form alliances, and use nukes/boats — and train an agent that learns to play it *well* by trial and error, rather than by hand-coded rules. The north star is an agent that discovers real strategy (grow economy, ally up against threats, keep momentum) on its own from a reward signal.

## The story so far

The full step-by-step journal is in [`DEVLOG.md`](DEVLOG.md). The short version:

1. **Ran the engine headless** and built a canvas replay viewer (`viz/index.html`) so we can *watch* games, not just read metrics.
2. **Defined the agent's senses and moves** — a fixed observation vector (its situation as numbers) and an action set (expand, attack, build city/defense/silo/SAM, nuke, boat, port, request/accept alliance) plus a learned troop-commitment fraction.
3. **First learner (evolution strategy)** — mutate the network weights, keep the mutation if the game scored higher. It learned to beat weak "tribe" bots but couldn't handle real "Nation" opponents or credit delayed payoffs (it refused to build economy).
4. **Switched to gradient RL (PyTorch).** The TS game became a Gym-style **environment server** ([`src/env/env_server.ts`](src/env/env_server.ts)) that Python drives over a pipe, and we implemented **REINFORCE** (policy gradient) in [`train_torch/reinforce.py`](train_torch/reinforce.py).
5. **Fixed the environment to be realistic and correct** — real world-map nations at density matched to a full game (not a sparse handful), and a build bug where structures silently failed (it was building on a captured tile).
6. **Reshaped the reward around *momentum*** — the old reward secretly paid the agent to turtle behind alliances (a flat survival bonus dwarfed everything, and land gains were clawed back on death). The new reward banks the peak size reached and rewards building, so growth and aggression pay off. This flipped behavior: it now attacks and builds cities instead of hiding.
7. **Made alliances a learned decision** — richer observations (relative strength of neighbors, total border pressure, ally backing, and the strength of whoever's offering an alliance), log-scaled gold so it can tell 125K from 25M, and `acceptAlliance` as its own action so it can learn *when* to trust an offer.

## How it works (the loop)

```
env_server.ts  <--JSON over stdio-->  reinforce.py
 (OpenFront game)                      (PyTorch policy)

reset(seed) -> observation
step(action, troop) -> {observation, reward, done}
```

Each step, the policy reads ~16 numbers describing its situation, outputs a probability over actions + a troop fraction, and **samples** a move (sampling is how it explores). Over many games, REINFORCE nudges the network weights so that decisions which preceded higher reward become more likely. Run `npm run agent` to play a trained game and record a replay; open `viz/index.html` to watch it.

## Current state

The agent reliably expands, builds a city, uses boats, and its alliance behavior is becoming a learned choice. Reward climbs and sustains. The open problem: it learns to *take* territory but not *hold* it — it peaks then gets ganged down.

## Known limitations & where we're going

The policy is a small MLP that reads **global summaries** and picks an action *type* + amount — it never chooses *where* or *which player*, so all targeting (which tile to build on, which enemy to boat) is hand-coded heuristics. That single constraint drives most of our ceilings. The roadmap:

1. **PPO (a value-function critic)** — cheapest big win; absorbs spawn-luck variance and sharpens credit assignment.
2. **Relational / per-entity head** — score each player through the net to *learn* who to attack/ally/boat.
3. **Spatial CNN + tile action head** — read the map as an image and output "build/attack *here*", unlocking learned placement (and eventually learned spawn selection).

## Setup
```
git clone --recurse-submodules <this repo>
npm install
pip install -r train_torch/requirements.txt

npm run agent            # play & record a game with the current weights -> viz/replay.js
python train_torch/reinforce.py   # train the policy (REINFORCE)
```
