// STEP 7: the real environment — WORLD map, NATIONS + tribes. Our agent still plays
// its hand-written policy. Replay is pooled down to a fixed small grid because the raw
// 2000x1000 map is too big to store or (later) feed a network.
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
import { Policy, ACTIONS } from "../agent/policy";
import { computeReward } from "../agent/reward";
import { PseudoRandom } from "../../vendor/OpenFrontIO/src/core/PseudoRandom";
import {
  Cell, Difficulty, GameMapType, GameMapSize, GameMode, GameType,
  Nation, Player, PlayerInfo, PlayerType, UnitType,
} from "../../vendor/OpenFrontIO/src/core/game/Game";

// silence the engine's noisy "cannot build ..." warnings (Nations failing to place a structure)
const _warn = console.warn.bind(console);
console.warn = (...args: any[]) => { if (typeof args[0] === "string" && args[0].startsWith("cannot build")) return; _warn(...args); };

const gameID = "agent_world";
const FRAME_EVERY = 30, DECISION_EVERY = 20, BOTS = 20;

const dir = path.dirname(fileURLToPath(import.meta.url));
const md = path.join(dir, "../../vendor/OpenFrontIO/tests/testdata/maps/world");
const man = JSON.parse(fs.readFileSync(path.join(md, "manifest.json"), "utf8"));
const gameMap = await genTerrainFromBin(man.map, fs.readFileSync(path.join(md, "map.bin")));
const mini = await genTerrainFromBin(man.map4x, fs.readFileSync(path.join(md, "map4x.bin")));

const cfg: any = { gameMap: GameMapType.World, gameMapSize: GameMapSize.Normal, gameMode: GameMode.FFA,
  gameType: GameType.Singleplayer, difficulty: Difficulty.Medium, nations: "default",
  donateGold: false, donateTroops: false, bots: BOTS, infiniteGold: false, infiniteTroops: false,
  instantBuild: false, randomSpawn: false };
const config = new Config(cfg, null as any, false);

// build Nation objects from the manifest (mirrors the engine's createNationsForGame)
const rng = new PseudoRandom(12345);
const nations: Nation[] = (man.nations || []).map((n: any) =>
  new Nation(n.coordinates ? new Cell(n.coordinates[0], n.coordinates[1]) : undefined,
    new PlayerInfo(n.name, PlayerType.Nation, null, rng.nextID())));

const game = createGame([], nations, gameMap, mini, config);

// our agent
const AGENT_ID = "agent";
const agentInfo = new PlayerInfo("AGENT", PlayerType.Human, null, AGENT_ID);
function findSpawnTile(): number {
  const cx = Math.floor(game.width() / 2), cy = Math.floor(game.height() / 2);
  for (let r = 0; r < Math.max(game.width(), game.height()); r++)
    for (let dy = -r; dy <= r; dy += Math.max(1, r)) for (let dx = -r; dx <= r; dx += Math.max(1, r)) {
      const x = cx + dx, y = cy + dy;
      if (x < 0 || y < 0 || x >= game.width() || y >= game.height()) continue;
      const t = game.ref(x, y);
      if (game.isLand(t) && !game.isImpassable(t) && !game.hasOwner(t)) return t;
    }
  throw new Error("no spawn tile");
}
const exec = new Executor(game, gameID, undefined);
if (config.spawnNations()) game.addExecution(...exec.nationExecutions());
if (config.bots() > 0) game.addExecution(...exec.spawnTribes(config.bots()));
game.addExecution(new WinCheckExecution());

// run the spawn phase so nations place themselves, then end it
for (let sp = 0; sp < 150; sp++) game.executeNextTick();
game.endSpawnPhase();

// now that nations/tribes have spawned, drop in our agent on free land
game.addPlayer(agentInfo);
const spawnTile = findSpawnTile();
game.addExecution(new SpawnExecution(gameID, agentInfo, spawnTile));
game.executeNextTick(); // let the agent spawn
const me = game.player(AGENT_ID);
const OPPONENTS = nations.length + BOTS;
const policy = new Policy(7); // 7 scalar observations in, 3 actions out (random weights for now)

// SCALAR OBSERVATION (7 normalized numbers)
function observe(): { labels: string[]; values: number[] } {
  const landTotal = game.numLandTiles(); const troops = me.troops();
  let emptyAdj = 0; const enemies = new Set<Player>();
  for (const t of me.borderTiles()) game.forEachNeighbor(t, (nb) => {
    if (!game.isLand(nb) || game.isImpassable(nb)) return;
    if (game.ownerID(nb) === me.smallID()) return;
    if (!game.hasOwner(nb)) { emptyAdj = 1; return; }
    const o = game.playerBySmallID(game.ownerID(nb)); if (o.isPlayer() && !me.isFriendly(o)) enemies.add(o);
  });
  const weakest = [...enemies].sort((a, b) => a.troops() - b.troops())[0];
  const enemiesAlive = game.players().filter((p) => p.isPlayer() && p.isAlive() && p.id() !== AGENT_ID).length;
  const labels = ["landShare","troops","gold","enemiesAlive","emptyAdj","enemyNbrs","vsWeakest"];
  const values = [me.numTilesOwned()/landTotal, Math.min(1,troops/200000), Math.min(1,Number(me.gold())/200000),
    enemiesAlive/OPPONENTS, emptyAdj, Math.min(1,enemies.size/6),
    weakest ? Math.min(1, troops/Math.max(1,weakest.troops())/2) : 1];
  return { labels, values };
}

// SPATIAL OBSERVATION: 8 full-map channels (see DEVLOG step 6)
const STRUCT_TYPES = new Set<UnitType>([UnitType.City,UnitType.Port,UnitType.Factory,UnitType.MissileSilo,UnitType.DefensePost,UnitType.SAMLauncher]);
function observeSpatial() {
  const W = game.width(), H = game.height(); const mk = () => new Uint8Array(W*H);
  const mine=mk(),hostile=mk(),allied=mk(),neutral=mk(),blocked=mk(),fallout=mk(),myStruct=mk(),enemyStruct=mk();
  for (let y=0;y<H;y++) for (let x=0;x<W;x++){ const t=game.ref(x,y), i=y*W+x;
    if (!game.isLand(t)||game.isImpassable(t)){ blocked[i]=1; continue; }
    if (game.hasFallout(t)) fallout[i]=1;
    if (!game.hasOwner(t)){ neutral[i]=1; continue; }
    const sid=game.ownerID(t);
    if (sid===me.smallID()) mine[i]=1;
    else { const o=game.playerBySmallID(sid); if (o.isPlayer()&&me.isFriendly(o)) allied[i]=1; else hostile[i]=1; } }
  for (const u of game.units()){ if(!u.isActive()||!STRUCT_TYPES.has(u.type())) continue;
    const i=game.y(u.tile())*W+game.x(u.tile()); if(u.owner().smallID()===me.smallID()) myStruct[i]=1; else enemyStruct[i]=1; }
  return { channels:["mine","hostile","allied","neutral","blocked","fallout","myStruct","enemyStruct"],
           data:[mine,hostile,allied,neutral,blocked,fallout,myStruct,enemyStruct] };
}

// ---- pooled replay recording (fixed small grid) ----
const W = game.width(), H = game.height();
const DS = Math.max(1, Math.ceil(Math.max(W, H) / 240)); // downsample factor
const RW = Math.ceil(W / DS), RH = Math.ceil(H / DS);
const PALETTE = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#800000","#808000","#000075"];
const terrain = new Uint8Array(RW * RH);
for (let ry = 0; ry < RH; ry++) for (let rx = 0; rx < RW; rx++) {
  const t = game.ref(Math.min(W - 1, rx * DS), Math.min(H - 1, ry * DS));
  terrain[ry * RW + rx] = game.isLand(t) && !game.isImpassable(t) ? 1 : 0;
}
const idToIdx = new Map<number, number>();
const legend: { name: string; color: string }[] = [];
const deltas: string[] = []; const frameTicks: number[] = [];
let prev = new Uint8Array(RW * RH);
function ownerIdxAt(x: number, y: number): number {
  const t = game.ref(x, y);
  if (!game.hasOwner(t)) return 0;
  const sid = game.ownerID(t);
  let idx = idToIdx.get(sid);
  if (idx === undefined) { idx = legend.length + 1; idToIdx.set(sid, idx);
    const p: any = game.playerBySmallID(sid);
    legend.push({ name: p?.name?.() ?? `#${sid}`, color: p?.id?.() === AGENT_ID ? "#ffffff" : PALETTE[(idx - 1) % PALETTE.length] });
  }
  return idx;
}
function snapshot() {
  const cur = new Uint8Array(RW * RH);
  for (let ry = 0; ry < RH; ry++) for (let rx = 0; rx < RW; rx++)
    cur[ry * RW + rx] = ownerIdxAt(Math.min(W - 1, rx * DS), Math.min(H - 1, ry * DS));
  const changed: number[] = []; for (let i = 0; i < cur.length; i++) if (cur[i] !== prev[i]) changed.push(i, cur[i]);
  const n = changed.length / 2; const buf = new Uint8Array(n * 5); const dv = new DataView(buf.buffer);
  for (let k = 0; k < n; k++) { dv.setUint32(k * 5, changed[k * 2], true); dv.setUint8(k * 5 + 4, changed[k * 2 + 1]); }
  deltas.push(Buffer.from(buf).toString("base64")); frameTicks.push(game.ticks()); prev = cur;
}

const STR = new Set([UnitType.City,UnitType.Port,UnitType.Factory,UnitType.MissileSilo,UnitType.DefensePost,UnitType.SAMLauncher]);
console.log(`WORLD ${W}x${H} pooled to ${RW}x${RH} | ${nations.length} nations + ${BOTS} tribes | agent at (${game.x(spawnTile)},${game.y(spawnTile)})`);
let tick = 0;
let peakTiles = 0; // track the most land we ever held (for the reward)
for (; tick < 6000; tick++) {
  if (me.isAlive() && tick % DECISION_EVERY === 0 && me.troops() > 1) {
    let emptyLandAdjacent = false; const enemies = new Set<Player>();
    for (const t of me.borderTiles()) game.forEachNeighbor(t, (nb) => {
      if (!game.isLand(nb) || game.isImpassable(nb)) return;
      if (game.ownerID(nb) === me.smallID()) return;
      if (!game.hasOwner(nb)) { emptyLandAdjacent = true; return; }
      const o = game.playerBySmallID(game.ownerID(nb));
      if (o.isPlayer() && !me.isFriendly(o)) enemies.add(o);
    });
    const weakest = [...enemies].sort((a, b) => a.troops() - b.troops())[0];
    // THE POLICY NETWORK chooses the action from the observation (random weights -> plays badly)
    const { action, probs } = policy.forward(observe().values);
    if (action === 0 && emptyLandAdjacent) game.addExecution(new AttackExecution(me.troops() / 2, me, game.terraNullius().id()));
    else if (action === 1 && weakest) game.addExecution(new AttackExecution(me.troops() / 3, me, weakest.id()));
    // action 2 = wait (do nothing)
    if (tick % 100 === 0) console.log(`   policy: probs=[${probs.map((p) => p.toFixed(2)).join(", ")}] -> ${ACTIONS[action]}`);
  }
  game.executeNextTick();
  if (me.isAlive()) peakTiles = Math.max(peakTiles, me.numTilesOwned());
  if (tick % FRAME_EVERY === 0) snapshot();
  if (tick % 1000 === 0) {
    const nat = game.players().filter(p => p.type() === PlayerType.Nation && p.isAlive()).length;
    const structs = game.units().filter(u => u.isActive() && STR.has(u.type())).length;
    console.log(`tick ${tick} | agent alive=${me.isAlive()} tiles=${me.numTilesOwned()} | nations alive=${nat} structures=${structs}`);
    if (me.isAlive() && tick === 1000) {
      const o = observe(); console.log("   scalar: " + o.labels.map((l,i)=>`${l}=${o.values[i].toFixed(2)}`).join("  "));
      const sp = observeSpatial();
      console.log("   channel sums: " + sp.channels.map((c,k)=>`${c}=${sp.data[k].reduce((a,v)=>a+v,0)}`).join("  "));
    }
  }
  const aliveP = game.players().filter(p => p.isPlayer() && p.isAlive());
  if (aliveP.length <= 1) break;
}
snapshot();
// ---- score this game with the reward ----
const landTotal = game.numLandTiles();
const survived = me.isAlive();
const won = survived && me.numTilesOwned() >= 0.8 * landTotal;
const stats = { peakLandShare: peakTiles / landTotal, survived, won };
const reward = computeReward(stats);
console.log(`REWARD = ${reward.toFixed(4)}  (peakLandShare=${stats.peakLandShare.toFixed(4)}, survived=${survived}, won=${won})`);

const outDir = path.join(dir, "../../viz"); fs.mkdirSync(outDir, { recursive: true });
const payload = { W: RW, H: RH, interval: FRAME_EVERY, winner: "(world demo)", terrain: Buffer.from(terrain).toString("base64"), legend, frameTicks, deltas };
fs.writeFileSync(path.join(outDir, "replay.js"), "window.REPLAY = " + JSON.stringify(payload) + ";");
console.log(`agent ${me.isAlive() ? "survived to tick " + tick : "died"} | wrote viz/replay.js (${deltas.length} frames, ${(fs.statSync(path.join(outDir,"replay.js")).size/1e6).toFixed(2)} MB, pooled ${RW}x${RH})`);
