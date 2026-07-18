// Phase 0 headless harness: load a small map, spawn AI bots, tick a full
// OpenFront game to completion with no browser and no rendering.
//
// Run (after setup — see SETUP.md):
//   npm run harness
//
// The engine lives in the pinned submodule at vendor/OpenFrontIO.
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { Config } from "../../vendor/OpenFrontIO/src/core/configuration/Config";
import { createGame } from "../../vendor/OpenFrontIO/src/core/game/GameImpl";
import { genTerrainFromBin } from "../../vendor/OpenFrontIO/src/core/game/TerrainMapLoader";
import { Executor } from "../../vendor/OpenFrontIO/src/core/execution/ExecutionManager";
import { WinCheckExecution } from "../../vendor/OpenFrontIO/src/core/execution/WinCheckExecution";
import {
  Difficulty,
  GameMapType,
  GameMapSize,
  GameMode,
  GameType,
} from "../../vendor/OpenFrontIO/src/core/game/Game";

const gameID = "headless_game";

// --- Load a small pre-baked test map straight from the engine's test data ---
const dir = path.dirname(fileURLToPath(import.meta.url));
const mapDir = path.join(dir, "../../vendor/OpenFrontIO/tests/testdata/maps/plains");
const manifest = JSON.parse(fs.readFileSync(path.join(mapDir, "manifest.json"), "utf8"));
const gameMap = await genTerrainFromBin(manifest.map, fs.readFileSync(path.join(mapDir, "map.bin")));
const miniGameMap = await genTerrainFromBin(manifest.map4x, fs.readFileSync(path.join(mapDir, "map4x.bin")));

// --- Game configuration (single-player FFA, all bots) ---
const gameConfig: any = {
  gameMap: GameMapType.Plains,
  gameMapSize: GameMapSize.Normal,
  gameMode: GameMode.FFA,
  gameType: GameType.Singleplayer,
  difficulty: Difficulty.Medium,
  nations: "default",
  donateGold: false,
  donateTroops: false,
  bots: 15, // number of AI bots to fill the map
  infiniteGold: false,
  infiniteTroops: false,
  instantBuild: false,
  randomSpawn: false,
};

// --- Build the game (mirrors what GameRunner.init() does, minus the browser) ---
const config = new Config(gameConfig, null as any, false);
const game = createGame([], [], gameMap, miniGameMap, config);
const exec = new Executor(game, gameID, undefined);
if (config.spawnNations()) game.addExecution(...exec.nationExecutions());
if (config.bots() > 0) game.addExecution(...exec.spawnTribes(config.bots()));
game.addExecution(new WinCheckExecution());
game.endSpawnPhase();

console.log(`Map: ${game.width()}x${game.height()}, land tiles: ${game.numLandTiles()}`);

// --- The tick loop: this is the "step" function everything else will wrap ---
const t0 = performance.now();
const maxTicks = 30000;
let tick = 0;
let started = false; // guards against ending before bots have spawned
for (; tick < maxTicks; tick++) {
  game.executeNextTick();
  const alive = game.players().filter((p) => p.isAlive());
  if (alive.length > 1) started = true;
  if (tick % 1000 === 0) {
    const top = [...alive].sort((a, b) => b.numTilesOwned() - a.numTilesOwned())[0];
    const pct = top ? ((100 * top.numTilesOwned()) / game.numLandTiles()).toFixed(1) : "0";
    console.log(`tick ${tick}: ${alive.length} alive, leader ${top?.name() ?? "-"} @ ${pct}% land`);
  }
  if (started && alive.length <= 1) break;
}

const secs = ((performance.now() - t0) / 1000).toFixed(2);
const winner = game.players().find((p) => p.isAlive());
const wpct = winner ? ((100 * winner.numTilesOwned()) / game.numLandTiles()).toFixed(1) : "0";
console.log(`\nDONE in ${tick} ticks / ${secs}s. Winner: ${winner?.name() ?? "none"} (${wpct}% land)`);
