// STEP 1 (+ recording): add one player WE control (no brain yet), read its state,
// AND record a replay so we can watch it in viz/index.html.
//
// Our player is drawn in bright WHITE so it stands out from the bots. With no brain
// it just sits on its spawn and gets eaten around tick ~900 - that is the expected,
// instructive result: it proves we can create/observe our player, and shows exactly
// why it needs a brain (which we add in step 2).
//
// Run:  npm run agent      then open viz/index.html in a browser.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Config } from "../../vendor/OpenFrontIO/src/core/configuration/Config";
import { createGame } from "../../vendor/OpenFrontIO/src/core/game/GameImpl";
import { genTerrainFromBin } from "../../vendor/OpenFrontIO/src/core/game/TerrainMapLoader";
import { Executor } from "../../vendor/OpenFrontIO/src/core/execution/ExecutionManager";
import { WinCheckExecution } from "../../vendor/OpenFrontIO/src/core/execution/WinCheckExecution";
import { SpawnExecution } from "../../vendor/OpenFrontIO/src/core/execution/SpawnExecution";
import { AttackExecution } from "../../vendor/OpenFrontIO/src/core/execution/AttackExecution";
import {
  Difficulty, GameMapType, GameMapSize, GameMode, GameType, Player, PlayerInfo, PlayerType,
} from "../../vendor/OpenFrontIO/src/core/game/Game";

const gameID = "agent_game";
const FRAME_EVERY = 20; // record a frame every N ticks
const DECISION_EVERY = 20; // our bot makes a decision every N ticks
const PALETTE = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6",
  "#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#800000","#808000","#000075"];

// --- load the small plains map ---
const dir = path.dirname(fileURLToPath(import.meta.url));
const mapDir = path.join(dir, "../../vendor/OpenFrontIO/tests/testdata/maps/plains");
const manifest = JSON.parse(fs.readFileSync(path.join(mapDir, "manifest.json"), "utf8"));
const gameMap = await genTerrainFromBin(manifest.map, fs.readFileSync(path.join(mapDir, "map.bin")));
const miniGameMap = await genTerrainFromBin(manifest.map4x, fs.readFileSync(path.join(mapDir, "map4x.bin")));

const gameConfig: any = {
  gameMap: GameMapType.Plains, gameMapSize: GameMapSize.Normal,
  gameMode: GameMode.FFA, gameType: GameType.Singleplayer, difficulty: Difficulty.Medium,
  nations: "default", donateGold: false, donateTroops: false, bots: 6,
  infiniteGold: false, infiniteTroops: false, instantBuild: false, randomSpawn: false,
};

const config = new Config(gameConfig, null as any, false);
const game = createGame([], [], gameMap, miniGameMap, config);

// --- (1) create OUR player and remember its id ---
const AGENT_ID = "agent";
const agentInfo = new PlayerInfo("AGENT", PlayerType.Human, null, AGENT_ID);
game.addPlayer(agentInfo);

// --- (2) find an empty land tile near the center to spawn it on ---
function findSpawnTile(): number {
  const cx = Math.floor(game.width() / 2), cy = Math.floor(game.height() / 2);
  for (let r = 0; r < Math.max(game.width(), game.height()); r++)
    for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      const x = cx + dx, y = cy + dy;
      if (x < 0 || y < 0 || x >= game.width() || y >= game.height()) continue;
      const t = game.ref(x, y);
      if (game.isLand(t) && !game.isImpassable(t) && !game.hasOwner(t)) return t;
    }
  throw new Error("no spawn tile found");
}
const spawnTile = findSpawnTile();

// --- (3) queue executions: our spawn, the bots, and the win checker ---
game.addExecution(new SpawnExecution(gameID, agentInfo, spawnTile));
const exec = new Executor(game, gameID, undefined);
if (config.bots() > 0) game.addExecution(...exec.spawnTribes(config.bots()));
game.addExecution(new WinCheckExecution());
game.endSpawnPhase();

// --- replay recording (same delta format the viewer expects) ---
const W = game.width(), H = game.height();
const terrain = new Uint8Array(W * H);
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const t = game.ref(x, y);
  terrain[y * W + x] = game.isLand(t) && !game.isImpassable(t) ? 1 : 0;
}
const idToIdx = new Map<number, number>();
const legend: { name: string; color: string }[] = [];
const deltas: string[] = [];
const frameTicks: number[] = [];
let prev = new Uint8Array(W * H);
function ownerIdx(t: number): number {
  if (!game.hasOwner(t)) return 0;
  const sid = game.ownerID(t);
  let idx = idToIdx.get(sid);
  if (idx === undefined) {
    idx = legend.length + 1;
    idToIdx.set(sid, idx);
    const p: any = game.playerBySmallID(sid);
    const isAgent = p?.id?.() === AGENT_ID;             // OUR player -> white
    legend.push({ name: p?.name?.() ?? `#${sid}`, color: isAgent ? "#ffffff" : PALETTE[(idx - 1) % PALETTE.length] });
  }
  return idx;
}
function snapshot() {
  const cur = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) cur[y * W + x] = ownerIdx(game.ref(x, y));
  const changed: number[] = [];
  for (let i = 0; i < cur.length; i++) if (cur[i] !== prev[i]) changed.push(i, cur[i]);
  const n = changed.length / 2;
  const buf = new Uint8Array(n * 5);
  const dv = new DataView(buf.buffer);
  for (let k = 0; k < n; k++) { dv.setUint32(k * 5, changed[k * 2], true); dv.setUint8(k * 5 + 4, changed[k * 2 + 1]); }
  deltas.push(Buffer.from(buf).toString("base64"));
  frameTicks.push(game.ticks());
  prev = cur;
}

const BOTS = config.bots();

// OBSERVATION: turn the live game into a fixed list of numbers the policy can read.
// The network never sees the game object — only this vector. If a fact isn't here,
// the bot is blind to it. Values are scaled toward 0..1 because neural nets train
// best when their inputs are small and in a similar range (this is "normalization").
function observe(): { labels: string[]; values: number[] } {
  const landTotal = game.numLandTiles();
  const troops = me.troops();
  // scan our border once: is empty land adjacent, and which enemies do we touch?
  let emptyLandAdjacent = 0;
  const enemies = new Set<Player>();
  for (const t of me.borderTiles()) {
    game.forEachNeighbor(t, (n) => {
      if (!game.isLand(n) || game.isImpassable(n)) return;
      if (game.ownerID(n) === me.smallID()) return;
      if (!game.hasOwner(n)) { emptyLandAdjacent = 1; return; }
      const o = game.playerBySmallID(game.ownerID(n));
      if (o.isPlayer() && !me.isFriendly(o)) enemies.add(o);
    });
  }
  const weakest = [...enemies].sort((a, b) => a.troops() - b.troops())[0];
  const enemiesAlive = game.players().filter((p) => p.isPlayer() && p.isAlive() && p.id() !== AGENT_ID).length;

  const labels = ["landShare", "troops", "gold", "enemiesAlive", "emptyAdj", "enemyNbrs", "vsWeakest"];
  const values = [
    me.numTilesOwned() / landTotal,                                        // share of the map we own (0..1)
    Math.min(1, troops / 200000),                                          // our army size, scaled
    Math.min(1, Number(me.gold()) / 200000),                               // our gold, scaled
    enemiesAlive / BOTS,                                                   // fraction of enemies still alive
    emptyLandAdjacent,                                                     // empty land next to us? 0 or 1
    Math.min(1, enemies.size / 6),                                         // how boxed-in we are
    weakest ? Math.min(1, troops / Math.max(1, weakest.troops()) / 2) : 1, // troops vs weakest nbr (0.5 = even)
  ];
  return { labels, values };
}

// SPATIAL OBSERVATION: the whole map as a stack of binary grids ("channels").
// Shape is [3, H, W] - exactly what a CNN reads. Unlike the scalar vector, this
// shows WHERE everything is, including players we're not in contact with.
// (plains is all land, so we don't need a water/terrain channel yet.)
function observeSpatial(): { channels: string[]; data: Uint8Array[] } {
  const W = game.width(), H = game.height();
  const mine = new Uint8Array(W * H);
  const enemy = new Uint8Array(W * H);
  const neutral = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const t = game.ref(x, y), i = y * W + x;
    if (!game.isLand(t)) continue;
    if (!game.hasOwner(t)) { neutral[i] = 1; continue; }
    if (game.ownerID(t) === me.smallID()) mine[i] = 1; else enemy[i] = 1;
  }
  return { channels: ["mine", "enemy", "neutral"], data: [mine, enemy, neutral] };
}

// shrunk ASCII view so we can eyeball the tensor: # = us, O = enemy, . = empty land
function printSpatial(cols = 40): void {
  const sp = observeSpatial();
  const W = game.width(), H = game.height();
  const step = Math.max(1, Math.ceil(W / cols));
  for (let y = 0; y < H; y += step) {
    let row = "";
    for (let x = 0; x < W; x += step) {
      const i = y * W + x;
      row += sp.data[0][i] ? "#" : sp.data[1][i] ? "O" : sp.data[2][i] ? "." : " ";
    }
    console.log("   " + row);
  }
}

// --- (4) the control loop: LOOK, DECIDE, ACT (and now, print the OBSERVATION) ---
const me = game.player(AGENT_ID);
console.log(`Agent spawned at tile (${game.x(spawnTile)}, ${game.y(spawnTile)})`);
let tick = 0;
for (; tick < 4000; tick++) {
  // DECIDE + ACT: every DECISION_EVERY ticks.
  if (me.isAlive() && tick % DECISION_EVERY === 0 && me.troops() > 1) {
    // OBSERVE the border: is there empty land next to us, and which enemies do we touch?
    let emptyLandAdjacent = false;
    const enemies = new Set<Player>();
    for (const t of me.borderTiles()) {
      game.forEachNeighbor(t, (n) => {
        if (!game.isLand(n) || game.isImpassable(n)) return;
        if (game.ownerID(n) === me.smallID()) return;
        if (!game.hasOwner(n)) { emptyLandAdjacent = true; return; }
        const o = game.playerBySmallID(game.ownerID(n));
        if (o.isPlayer() && !me.isFriendly(o)) enemies.add(o);
      });
    }
    // DECIDE: grow into empty land if we can; otherwise attack the weakest neighbor,
    // but ONLY if we clearly outnumber them (attacking is expensive — don't bleed the army).
    if (emptyLandAdjacent) {
      game.addExecution(new AttackExecution(me.troops() / 2, me, game.terraNullius().id()));
    } else if (enemies.size > 0) {
      const weakest = [...enemies].sort((a, b) => a.troops() - b.troops())[0];
      if (me.troops() > weakest.troops() * 2) {
        game.addExecution(new AttackExecution(me.troops() / 3, me, weakest.id()));
      }
    }
  }
  game.executeNextTick();
  if (tick % FRAME_EVERY === 0) snapshot();
  if (tick % 500 === 0) {
    console.log(`tick ${tick} | alive=${me.isAlive()} tiles=${me.numTilesOwned()} troops=${Math.floor(me.troops())} gold=${me.gold()}`);
    if (me.isAlive()) {
      const obs = observe();
      console.log("   observation: " + obs.labels.map((l, i) => `${l}=${obs.values[i].toFixed(2)}`).join("  "));
      if (tick === 500) { console.log("   spatial map (#=us  O=enemy  .=empty):"); printSpatial(); }
    }
  }
  if (!me.isAlive() && tick > 50) { console.log(`Agent died at tick ${tick}.`); break; }
}
snapshot();

// --- write the replay ---
const outDir = path.join(dir, "../../viz");
fs.mkdirSync(outDir, { recursive: true });
const payload = {
  W, H, interval: FRAME_EVERY, winner: "(agent demo)",
  terrain: Buffer.from(terrain).toString("base64"), legend, frameTicks, deltas,
};
fs.writeFileSync(path.join(outDir, "replay.js"), "window.REPLAY = " + JSON.stringify(payload) + ";");
console.log(`Wrote viz/replay.js (${deltas.length} frames). AGENT is the WHITE territory. Open viz/index.html.`);
