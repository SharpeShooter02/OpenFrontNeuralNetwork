// Environment SERVER: exposes the OpenFront agent game to an external driver (Python) over
// stdio as line-delimited JSON. Commands: {"cmd":"reset","seed":N} and
// {"cmd":"step","action":k,"troop":f}. Responds {obs:[...12], reward, done}. Per-step reward
// = 5*(change in land share) + small survival bonus, with terminal win/death bonuses.
import readline from "readline";
import fs from "fs"; import path from "path"; import { fileURLToPath } from "url";
import { Config } from "../../vendor/OpenFrontIO/src/core/configuration/Config";
import { createGame } from "../../vendor/OpenFrontIO/src/core/game/GameImpl";
import { genTerrainFromBin } from "../../vendor/OpenFrontIO/src/core/game/TerrainMapLoader";
import { Executor } from "../../vendor/OpenFrontIO/src/core/execution/ExecutionManager";
import { WinCheckExecution } from "../../vendor/OpenFrontIO/src/core/execution/WinCheckExecution";
import { SpawnExecution } from "../../vendor/OpenFrontIO/src/core/execution/SpawnExecution";
import { AttackExecution } from "../../vendor/OpenFrontIO/src/core/execution/AttackExecution";
import { ConstructionExecution } from "../../vendor/OpenFrontIO/src/core/execution/ConstructionExecution";
import { NukeExecution } from "../../vendor/OpenFrontIO/src/core/execution/NukeExecution";
import { TransportShipExecution } from "../../vendor/OpenFrontIO/src/core/execution/TransportShipExecution";
import { canBuildTransportShip } from "../../vendor/OpenFrontIO/src/core/game/TransportShipUtils";
import { Cell, Difficulty, GameMapType, GameMapSize, GameMode, GameType, Nation, Player, PlayerInfo, PlayerType, UnitType } from "../../vendor/OpenFrontIO/src/core/game/Game";

console.log = () => {}; console.warn = () => {}; console.debug = () => {}; // keep stdout clean for JSON

const NUM_NATIONS = 10, BOTS = 50, MAX_TICKS = 12000, DECIDE_EVERY = 20;
const dir = path.dirname(fileURLToPath(import.meta.url));
const md = path.join(dir, "../../vendor/OpenFrontIO/tests/testdata/maps/world");
const man = JSON.parse(fs.readFileSync(path.join(md, "manifest.json"), "utf8"));
const mapBuf = fs.readFileSync(path.join(md, "map4x.bin")), miniBuf = fs.readFileSync(path.join(md, "map16x.bin"));

let game: any, me: any, land = 1, tick = 0, spawn = -1, prevShare = 0, W = 0, H = 0;

function scan() {
  let empty = false, coastal = 0, shoreTile = -1; const enemies = new Set<Player>();
  for (const t of me.borderTiles()) {
    if (game.isShore(t)) { coastal = 1; if (shoreTile < 0) shoreTile = t; }
    game.forEachNeighbor(t, (nb: number) => {
      if (!game.isLand(nb) || game.isImpassable(nb)) return;
      if (game.ownerID(nb) === me.smallID()) return;
      if (!game.hasOwner(nb)) { empty = true; return; }
      const o = game.playerBySmallID(game.ownerID(nb)); if (o.isPlayer() && !me.isFriendly(o)) enemies.add(o);
    });
  }
  const sorted = [...enemies].sort((a, b) => a.troops() - b.troops());
  return { empty, coastal, shoreTile, enemies, weakest: sorted[0], strongest: sorted[sorted.length - 1] };
}

function observe(): number[] {
  if (!me.isAlive()) return new Array(12).fill(0);
  const s = scan();
  const enemiesAlive = game.players().filter((p: any) => p.isPlayer() && p.isAlive() && p.id() !== "agent").length;
  return [me.numTilesOwned()/land, Math.min(1,me.troops()/200000), Math.min(1,Number(me.gold())/200000),
    enemiesAlive/(NUM_NATIONS+BOTS), s.empty?1:0, Math.min(1,s.enemies.size/6),
    s.weakest?Math.min(1,me.troops()/Math.max(1,s.weakest.troops())/2):1,
    Math.min(1, me.allies().length/5), me.incomingAllianceRequests().length>0?1:0,
    Math.min(1, me.unitCount(UnitType.City)/8), me.unitCount(UnitType.MissileSilo)>0?1:0, s.coastal];
}

function act(action: number, troopFraction: number) {
  const s = scan();
  for (const req of me.incomingAllianceRequests()) req.accept();
  const commit = Math.floor(me.troops() * Math.max(0.01, Math.min(1, troopFraction)));
  const build = (u: UnitType, tile: number) => { const bt = me.canBuild(u, tile); if (bt) game.addExecution(new ConstructionExecution(me, u, bt)); };
  if (action === 0 && s.empty) game.addExecution(new AttackExecution(commit, me, game.terraNullius().id()));
  else if (action === 1 && s.weakest) game.addExecution(new AttackExecution(commit, me, s.weakest.id()));
  else if (action === 2 && s.strongest) game.addExecution(new AttackExecution(commit, me, s.strongest.id()));
  else if (action === 4) { for (const e of s.enemies) if (me.canSendAllianceRequest(e)) me.createAllianceRequest(e); }
  else if (action === 5) build(UnitType.City, spawn);
  else if (action === 6) build(UnitType.DefensePost, spawn);
  else if (action === 7) build(UnitType.MissileSilo, spawn);
  else if (action === 8) build(UnitType.SAMLauncher, spawn);
  else if (action === 9 && s.strongest && me.unitCount(UnitType.MissileSilo) > 0) { let tgt: number | null = null; for (const t of s.strongest.tiles()) { tgt = t; break; } if (tgt !== null) game.addExecution(new NukeExecution(UnitType.AtomBomb, me, tgt)); }
  else if (action === 10) { const ge = game.players().filter((p: any)=>p.isPlayer()&&p.isAlive()&&p.id()!=="agent"&&!me.isFriendly(p)).sort((a: any,b: any)=>a.troops()-b.troops())[0];
    if (ge) { let dst = -1; for (const t of ge.tiles()) { if (game.isShore(t)) { dst = t; break; } } if (dst < 0) for (const t of ge.tiles()) { dst = t; break; }
      if (dst >= 0 && canBuildTransportShip(game, me, dst) !== false) game.addExecution(new TransportShipExecution(me, dst, commit)); } }
  else if (action === 11 && s.shoreTile >= 0) build(UnitType.Port, s.shoreTile);
}

async function reset(seed: number): Promise<number[]> {
  const gameMap = await genTerrainFromBin(man.map4x, mapBuf);
  const mini = await genTerrainFromBin(man.map16x, miniBuf);
  const cfg: any = { gameMap: GameMapType.World, gameMapSize: GameMapSize.Normal, gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer, difficulty: Difficulty.Medium, nations: "default",
    donateGold:false, donateTroops:false, bots: BOTS, infiniteGold:false, infiniteTroops:false, instantBuild:false, randomSpawn:false };
  const config = new Config(cfg, null as any, false);
  let s = seed >>> 0; const rand = () => (s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 0xffffffff;
  W = gameMap.width(); H = gameMap.height();
  const nations: Nation[] = [];
  for (let i = 0; i < NUM_NATIONS; i++) { let x, y, t; do { x = Math.floor(rand()*W); y = Math.floor(rand()*H); t = gameMap.ref(x,y); } while (!gameMap.isLand(t)); nations.push(new Nation(new Cell(x,y), new PlayerInfo("Nat"+i, PlayerType.Nation, null, "nat"+i))); }
  game = createGame([], nations, gameMap, mini, config);
  const exec = new Executor(game, "env_" + seed, undefined);
  game.addExecution(...exec.nationExecutions()); game.addExecution(...exec.spawnTribes(BOTS)); game.addExecution(new WinCheckExecution());
  for (let sp = 0; sp < 150; sp++) game.executeNextTick();
  game.endSpawnPhase();
  const info = new PlayerInfo("AGENT", PlayerType.Human, null, "agent"); game.addPlayer(info);
  const tx = Math.floor(rand()*W), ty = Math.floor(rand()*H); spawn = -1; let bd = Infinity;
  for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) { const t = game.ref(x, y);
    if (game.isLand(t) && !game.isImpassable(t) && !game.hasOwner(t)) { const d = (x-tx)*(x-tx)+(y-ty)*(y-ty); if (d < bd) { bd = d; spawn = t; } } }
  game.addExecution(new SpawnExecution("env_" + seed, info, spawn));
  game.executeNextTick();
  me = game.player("agent"); land = game.numLandTiles(); tick = 0; prevShare = me.isAlive() ? me.numTilesOwned()/land : 0;
  return observe();
}

function step(action: number, troop: number) {
  if (me.isAlive() && me.troops() > 1) act(action, troop);
  for (let i = 0; i < DECIDE_EVERY; i++) { game.executeNextTick(); tick++; if (!me.isAlive() && tick > 50) break; }
  const curShare = me.isAlive() ? me.numTilesOwned()/land : 0;
  let reward = 5 * (curShare - prevShare) + (me.isAlive() ? 0.001 : 0);
  prevShare = curShare;
  const aliveP = game.players().filter((p: any) => p.isPlayer() && p.isAlive()).length;
  const done = !me.isAlive() || tick >= MAX_TICKS || aliveP <= 1;
  if (done) { if (me.isAlive() && me.numTilesOwned() >= 0.8*land) reward += 5; if (!me.isAlive()) reward -= 0.3; }
  return { obs: observe(), reward, done };
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", async (line) => {
  if (!line.trim()) return;
  const cmd = JSON.parse(line);
  let resp: any;
  if (cmd.cmd === "reset") resp = { obs: await reset(cmd.seed ?? 0), reward: 0, done: false };
  else if (cmd.cmd === "step") resp = step(cmd.action, cmd.troop);
  else resp = { error: "unknown cmd" };
  process.stdout.write(JSON.stringify(resp) + "\n");
});
