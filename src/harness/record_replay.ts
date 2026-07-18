// Records a headless OpenFront game to viz/replay.js, which viz/index.html animates.
// Run:  npm run replay   then open viz/index.html in a browser.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Config } from "../../vendor/OpenFrontIO/src/core/configuration/Config";
import { createGame } from "../../vendor/OpenFrontIO/src/core/game/GameImpl";
import { genTerrainFromBin } from "../../vendor/OpenFrontIO/src/core/game/TerrainMapLoader";
import { Executor } from "../../vendor/OpenFrontIO/src/core/execution/ExecutionManager";
import { WinCheckExecution } from "../../vendor/OpenFrontIO/src/core/execution/WinCheckExecution";
import {
  Difficulty, GameMapType, GameMapSize, GameMode, GameType,
} from "../../vendor/OpenFrontIO/src/core/game/Game";

// --- knobs ---
const BOTS = 15;          // number of AI bots
const FRAME_EVERY = 30;   // record one frame every N ticks (lower = smoother, bigger file)

const PALETTE = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6",
  "#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#fffac8","#800000","#aaffc3",
  "#808000","#ffd8b1","#000075","#a9a9a9"];
const colorFor = (i: number) => PALETTE[(i - 1) % PALETTE.length];

const gameID = "replay_game";
const dir = path.dirname(fileURLToPath(import.meta.url));
const mapDir = path.join(dir, "../../vendor/OpenFrontIO/tests/testdata/maps/plains");
const manifest = JSON.parse(fs.readFileSync(path.join(mapDir, "manifest.json"), "utf8"));
const gameMap = await genTerrainFromBin(manifest.map, fs.readFileSync(path.join(mapDir, "map.bin")));
const miniGameMap = await genTerrainFromBin(manifest.map4x, fs.readFileSync(path.join(mapDir, "map4x.bin")));

const gameConfig: any = {
  gameMap: GameMapType.Plains, gameMapSize: GameMapSize.Normal,
  gameMode: GameMode.FFA, gameType: GameType.Singleplayer, difficulty: Difficulty.Medium,
  nations: "default", donateGold: false, donateTroops: false, bots: BOTS,
  infiniteGold: false, infiniteTroops: false, instantBuild: false, randomSpawn: false,
};

const config = new Config(gameConfig, null as any, false);
const game = createGame([], [], gameMap, miniGameMap, config);
const exec = new Executor(game, gameID, undefined);
if (config.spawnNations()) game.addExecution(...exec.nationExecutions());
if (config.bots() > 0) game.addExecution(...exec.spawnTribes(config.bots()));
game.addExecution(new WinCheckExecution());
game.endSpawnPhase();

const W = game.width(), H = game.height();
const terrain = new Uint8Array(W * H); // 1 = playable land, 0 = water/impassable
for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
  const t = game.ref(x, y);
  terrain[y * W + x] = game.isLand(t) && !game.isImpassable(t) ? 1 : 0;
}

const idToIdx = new Map<number, number>();
const legend: { name: string; color: string }[] = [];
const deltas: string[] = [];     // per-frame CHANGED tiles only (5 bytes each: u32 index + u8 owner)
const frameTicks: number[] = [];
let prev = new Uint8Array(W * H); // previous frame's ownership, for diffing
function ownerIdx(t: number): number {
  if (!game.hasOwner(t)) return 0;
  const sid = game.ownerID(t);
  let idx = idToIdx.get(sid);
  if (idx === undefined) {
    idx = legend.length + 1;
    idToIdx.set(sid, idx);
    const p: any = game.playerBySmallID(sid);
    legend.push({ name: p?.name?.() ?? `#${sid}`, color: colorFor(idx) });
  }
  return idx;
}
function snapshot() {
  const cur = new Uint8Array(W * H);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) cur[y * W + x] = ownerIdx(game.ref(x, y));
  // record only tiles that changed since the previous frame
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

let tick = 0, started = false;
for (; tick < 30000; tick++) {
  game.executeNextTick();
  const alive = game.players().filter((p) => p.isAlive());
  if (alive.length > 1) started = true;
  if (tick % FRAME_EVERY === 0) snapshot();
  if (started && alive.length <= 1) break;
}
snapshot();

const winner = game.players().find((p) => p.isAlive());
const outDir = path.join(dir, "../../viz");
fs.mkdirSync(outDir, { recursive: true });
const payload = {
  W, H, interval: FRAME_EVERY, winner: winner?.name() ?? "none",
  terrain: Buffer.from(terrain).toString("base64"), legend, frameTicks, deltas,
};
fs.writeFileSync(path.join(outDir, "replay.js"), "window.REPLAY = " + JSON.stringify(payload) + ";");
const sizeMB = (fs.statSync(path.join(outDir, "replay.js")).size / 1e6).toFixed(2);
console.log(`Recorded ${deltas.length} frames over ${tick} ticks. Players: ${legend.length}. Winner: ${payload.winner}.`);
console.log(`Wrote viz/replay.js (${sizeMB} MB). Open viz/index.html in a browser.`);
