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

// Density-matched to a real full-world game (~1 player / 1400 land tiles) on map4x's 157,860
// land tiles: ~15 nations + ~100 tribes. Override via env for full-world (61/300) or tuning.
const NUM_NATIONS = +(process.env.NUM_NATIONS ?? 15), BOTS = +(process.env.BOTS ?? 100), MAX_TICKS = 12000, DECIDE_EVERY = 20;
const dir = path.dirname(fileURLToPath(import.meta.url));
const md = path.join(dir, "../../vendor/OpenFrontIO/tests/testdata/maps/world");
const man = JSON.parse(fs.readFileSync(path.join(md, "manifest.json"), "utf8"));
const mapBuf = fs.readFileSync(path.join(md, "map4x.bin")), miniBuf = fs.readFileSync(path.join(md, "map16x.bin"));

let game: any, me: any, land = 1, tick = 0, spawn = -1, prevShare = 0, peakShare = 0, prevCities = 0, W = 0, H = 0;

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
  const sumTroops = sorted.reduce((a, p) => a + p.troops(), 0);
  return { empty, coastal, shoreTile, enemies, weakest: sorted[0], strongest: sorted[sorted.length - 1], sumTroops };
}

function observe(): number[] {
  if (!me.isAlive()) return new Array(16).fill(0);
  const s = scan();
  const troops = Math.max(1, me.troops());
  const enemiesAlive = game.players().filter((p: any) => p.isPlayer() && p.isAlive() && p.id() !== "agent").length;
  const allyTroops = me.allies().reduce((a: number, p: any) => a + p.troops(), 0);
  let offererTroops = 0; for (const r of me.incomingAllianceRequests()) offererTroops = Math.max(offererTroops, r.requestor().troops());
  return [me.numTilesOwned()/land, Math.min(1,me.troops()/200000),
    Math.log1p(Number(me.gold()))/Math.log1p(25_000_000),                            // gold: log-scaled (spans 0..25M MIRV)
    enemiesAlive/(NUM_NATIONS+BOTS), s.empty?1:0, Math.min(1,s.enemies.size/6),
    s.weakest?Math.min(1,me.troops()/Math.max(1,s.weakest.troops())/2):1,
    Math.min(1, me.allies().length/5), me.incomingAllianceRequests().length>0?1:0,
    Math.min(1, me.unitCount(UnitType.City)/8), me.unitCount(UnitType.MissileSilo)>0?1:0, s.coastal,
    s.strongest?Math.min(1,me.troops()/Math.max(1,s.strongest.troops())/2):1,        // strongest-neighbor ratio
    Math.min(1, s.sumTroops/troops/3),                                                // total border pressure (all neighbors)
    Math.min(1, allyTroops/troops),                                                   // ally backing
    offererTroops>0?Math.min(1, offererTroops/troops/2):0];                           // strength of strongest alliance offerer
}

function act(action: number, troopFraction: number) {
  const s = scan();
  const commit = Math.floor(me.troops() * Math.max(0.01, Math.min(1, troopFraction)));
  // Build on a currently-owned tile (the fixed spawn tile is often captured by mid-game).
  const ownedTile = () => { if (game.ownerID(spawn) === me.smallID()) return spawn; for (const t of me.tiles()) return t; return spawn; };
  const build = (u: UnitType, tile: number) => { const bt = me.canBuild(u, tile); if (bt) game.addExecution(new ConstructionExecution(me, u, bt)); };
  if (action === 0 && s.empty) game.addExecution(new AttackExecution(commit, me, game.terraNullius().id()));
  else if (action === 1 && s.weakest) game.addExecution(new AttackExecution(commit, me, s.weakest.id()));
  else if (action === 2 && s.strongest) game.addExecution(new AttackExecution(commit, me, s.strongest.id()));
  else if (action === 3) { for (const req of me.incomingAllianceRequests()) req.accept(); }                    // accept incoming (now a learned choice)
  else if (action === 4) { for (const e of s.enemies) if (e.troops() > me.troops() && me.canSendAllianceRequest(e)) me.createAllianceRequest(e); }  // request only STRONGER (ally up, don't ally prey)
  else if (action === 5) build(UnitType.City, ownedTile());
  else if (action === 6) build(UnitType.DefensePost, ownedTile());
  else if (action === 7) build(UnitType.MissileSilo, ownedTile());
  else if (action === 8) build(UnitType.SAMLauncher, ownedTile());
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
  // Real nations from the world manifest, scaled to map4x space and snapped to the nearest land.
  const scaleX = W / man.map.width, scaleY = H / man.map.height;
  const snapLand = (x: number, y: number) => {
    for (let r = 0; r < 20; r++) for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
      if (gameMap.isLand(gameMap.ref(xx, yy))) return [xx, yy]; }
    return [Math.max(0, Math.min(W - 1, x)), Math.max(0, Math.min(H - 1, y))]; };
  const manNats: any[] = man.nations.filter((n: any) => n.coordinates);
  const stride = NUM_NATIONS >= manNats.length ? 1 : Math.ceil(manNats.length / NUM_NATIONS);
  const chosen = manNats.filter((_, i) => i % stride === 0).slice(0, NUM_NATIONS);
  const nations: Nation[] = [];
  for (const mn of chosen) { const [x, y] = snapLand(Math.floor(mn.coordinates[0] * scaleX), Math.floor(mn.coordinates[1] * scaleY));
    nations.push(new Nation(new Cell(x, y), new PlayerInfo(mn.name, PlayerType.Nation, null, "nat" + nations.length))); }
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
  peakShare = prevShare; prevCities = me.isAlive() ? me.unitCount(UnitType.City) : 0;
  return observe();
}

function step(action: number, troop: number) {
  if (me.isAlive() && me.troops() > 1) act(action, troop);
  for (let i = 0; i < DECIDE_EVERY; i++) { game.executeNextTick(); tick++; if (!me.isAlive() && tick > 50) break; }
  const alive = me.isAlive();
  const curShare = alive ? me.numTilesOwned()/land : 0;
  const curCities = alive ? me.unitCount(UnitType.City) : prevCities;
  // Momentum reward: dense expansion delta + a bootstrap bonus for each city built (economy).
  // No flat survival term — that just paid the agent to turtle behind alliances.
  let reward = 3 * (curShare - prevShare) + 0.2 * Math.max(0, curCities - prevCities);
  prevShare = curShare; prevCities = curCities;
  if (alive) peakShare = Math.max(peakShare, curShare);
  const aliveP = game.players().filter((p: any) => p.isPlayer() && p.isAlive()).length;
  const done = !alive || tick >= MAX_TICKS || aliveP <= 1;
  if (done) {
    reward += 5 * peakShare;                                  // BANK the peak: growth counts even if it later dies
    if (alive && me.numTilesOwned() >= 0.8*land) reward += 5; // decisive win
    if (!alive) reward -= 0.15;                               // modest death penalty (< peak bonus, so aggression pays)
  }
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
