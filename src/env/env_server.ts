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

// Training map. MAP=box (default): "The Box", a big all-land square (map16x = 512x512) — its size
// gives players room to grow instead of instantly colliding, so games stay long and strategic (and
// all-land = no water/nav overhead). MAP=world: the big world map4x. MAP=bigplains: tiny 200x200
// (games consolidate too fast — kept for quick probes only). Override players via NUM_NATIONS/BOTS.
const MAP = process.env.MAP ?? "box";
const MAPS: any = {
  world:     { rel: "tests/testdata/maps/world",      game: "map4x",  mini: "map16x", realNations: true },
  bigplains: { rel: "tests/testdata/maps/big_plains", game: "map",    mini: "map4x",  realNations: false },
  box:       { rel: "resources/maps/thebox",          game: "map16x", mini: "map16x", realNations: true },
};
const mc = MAPS[MAP];
const DEF: any = { world: [15, 100], bigplains: [6, 24], box: [60, 400] };  // realistic crowd: ~460 players, no cheap outlast-wins
const NUM_NATIONS = +(process.env.NUM_NATIONS ?? DEF[MAP][0]), BOTS = +(process.env.BOTS ?? DEF[MAP][1]), MAX_TICKS = 12000, DECIDE_EVERY = 20;
const dir = path.dirname(fileURLToPath(import.meta.url));
const md = path.join(dir, "../../vendor/OpenFrontIO/" + mc.rel);
const man = JSON.parse(fs.readFileSync(path.join(md, "manifest.json"), "utf8"));
const mapBuf = fs.readFileSync(path.join(md, mc.game + ".bin")), miniBuf = fs.readFileSync(path.join(md, mc.mini + ".bin"));

let game: any, me: any, land = 1, tick = 0, spawn = -1, prevShare = 0, peakShare = 0, peakEcon = 0, W = 0, H = 0;
const econCount = (p: any) => p.unitCount(UnitType.City) + p.unitCount(UnitType.Port) + p.unitCount(UnitType.Factory); // gold-economy structures

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

// --- Diplomacy candidate scoring: the policy picks WHICH player to accept/request/break, instead of
// the old "accept all / request all" heuristics. candidates() returns up to KCAND rows of FCAND
// features (row 0 = no-op); curCands holds the matching {player, kind, req} so step() can apply the
// chosen target index. kind: offer->accept, enemy->request, ally->break.
const KCAND = 6, FCAND = 7;
let curCands: any[] = [{ kind: "noop" }];
function candidates(): number[][] {
  curCands = [{ kind: "noop" }];
  const noop = [1, 0, 0, 0, 0, 0, 0];
  if (!me.isAlive()) return [noop];
  const myT = Math.max(1, me.troops());
  const feat = (p: any, kind: string) => [0, Math.min(1, p.troops()/myT/2), kind === "ally" ? 1 : 0,
    kind === "offer" ? 1 : 0, kind === "enemy" ? 1 : 0, Math.min(1, p.allies().length/5), Math.min(1, p.numTilesOwned()/land)];
  const rows: number[][] = [noop];
  const seen = new Set<number>();
  const add = (p: any, kind: string, req?: any) => {
    if (curCands.length >= KCAND || !p || !p.isAlive() || seen.has(p.smallID())) return;
    seen.add(p.smallID()); curCands.push({ player: p, kind, req }); rows.push(feat(p, kind)); };
  for (const r of me.incomingAllianceRequests()) add(r.requestor(), "offer", r);   // accept candidates
  for (const a of me.allies()) add(a, "ally");                                      // break candidates
  for (const e of [...scan().enemies].sort((x: any, y: any) => y.troops() - x.troops())) if (me.canSendAllianceRequest(e)) add(e, "enemy");  // request candidates
  return rows;
}

// --- Placement candidate scoring: the policy picks WHICH owned tile to build a defense post / silo /
// SAM / factory on (was a fixed heuristic). placeTiles() returns up to KP owned-tile candidates with
// FP tactical features [frontline, near-own-structure, interiorness, spacing]; the policy appends the
// structure type so its head can learn type-specific placement (defense->frontline, factory->interior).
const KP = 8, FP = 4;
let curPlace: number[] = [];
function placeTiles(): number[][] {
  curPlace = [];
  if (!me.isAlive()) return [];
  const structs = me.units([UnitType.City, UnitType.DefensePost, UnitType.MissileSilo, UnitType.SAMLauncher, UnitType.Port, UnitType.Factory]);
  const sxy = structs.map((u: any) => [game.x(u.tile()), game.y(u.tile())]);
  const border = new Set<number>(me.borderTiles());
  const enemyAdj = (t: number) => { let e = false; game.forEachNeighbor(t, (nb: number) => {
    if (game.isLand(nb) && game.hasOwner(nb) && game.ownerID(nb) !== me.smallID()) { const o = game.playerBySmallID(game.ownerID(nb)); if (o.isPlayer() && !me.isFriendly(o)) e = true; } }); return e; };
  let cx = 0, cy = 0, nc = 0, n = 0; const sample: number[] = [];
  for (const t of me.tiles()) { cx += game.x(t); cy += game.y(t); nc++; if ((n++ & 7) === 0 && sample.length < 64) sample.push(t); if (nc > 6000) break; }
  cx /= Math.max(1, nc); cy /= Math.max(1, nc); const halfDiag = Math.hypot(W, H) / 2;
  const rows: number[][] = [];
  for (const t of sample) { if (rows.length >= KP) break;
    const tx = game.x(t), ty = game.y(t); let md = Infinity;
    for (const [sx, sy] of sxy) { const d = Math.hypot(tx - sx, ty - sy); if (d < md) md = d; }
    curPlace.push(t);
    rows.push([border.has(t) && enemyAdj(t) ? 1 : 0, sxy.length ? Math.max(0, 1 - md/30) : 0,
      1 - Math.min(1, Math.hypot(tx - cx, ty - cy)/halfDiag), Math.min(1, md/15)]); }
  return rows;
}

function act(action: number, troopFraction: number, target: number, ptarget: number) {
  const s = scan();
  const commit = Math.floor(me.troops() * Math.max(0.01, Math.min(1, troopFraction)));
  // Build on a currently-owned tile (the fixed spawn tile is often captured by mid-game).
  const ownedTile = () => { if (game.ownerID(spawn) === me.smallID()) return spawn; for (const t of me.tiles()) return t; return spawn; };
  // Spread structures out: hand canBuild an owned tile FAR from existing structures, so structureMinDist
  // doesn't reject it — otherwise every build stacks at one spot and only the 1st city ever places.
  const buildTile = () => {
    const structs = me.units([UnitType.City, UnitType.DefensePost, UnitType.MissileSilo, UnitType.SAMLauncher, UnitType.Port]);
    if (structs.length === 0) return ownedTile();
    const sx = structs.map((u: any) => game.x(u.tile())), sy = structs.map((u: any) => game.y(u.tile()));
    let best = -1, bestD = -1, n = 0;
    for (const t of me.tiles()) { if ((n++ & 3) !== 0) continue; if (n > 2000) break;
      const tx = game.x(t), ty = game.y(t); let md = Infinity;
      for (let i = 0; i < sx.length; i++) { const dx = tx - sx[i], dy = ty - sy[i], d = dx*dx + dy*dy; if (d < md) md = d; }
      if (md > bestD) { bestD = md; best = t; } }
    return best >= 0 ? best : ownedTile();
  };
  const build = (u: UnitType, tile: number) => { const bt = me.canBuild(u, tile); if (bt) game.addExecution(new ConstructionExecution(me, u, bt)); };
  const placeTile = () => (ptarget >= 0 && ptarget < curPlace.length) ? curPlace[ptarget] : buildTile();  // learned placement, else spread heuristic
  if (action === 0 && s.empty) game.addExecution(new AttackExecution(commit, me, game.terraNullius().id()));
  else if (action === 1 && s.weakest) game.addExecution(new AttackExecution(commit, me, s.weakest.id()));
  else if (action === 2 && s.strongest) game.addExecution(new AttackExecution(commit, me, s.strongest.id()));
  else if (action === 3 && target > 0 && target < curCands.length) {   // diplomacy: act on the SCORED player
    const c = curCands[target];
    if (c.kind === "offer") { try { c.req.accept(); } catch {} }
    else if (c.kind === "enemy") { if (c.player.isAlive() && me.canSendAllianceRequest(c.player)) me.createAllianceRequest(c.player); }
    else if (c.kind === "ally") { for (const a of me.alliances()) if (a.other(me) === c.player) { me.breakAlliance(a); break; } }
  }
  // action 4 = wait / no-op (do nothing this decision)
  else if (action === 5) build(UnitType.City, buildTile());
  else if (action === 6) build(UnitType.DefensePost, placeTile());   // learned placement
  else if (action === 7) build(UnitType.MissileSilo, placeTile());
  else if (action === 8) build(UnitType.SAMLauncher, placeTile());
  else if (action === 9 && s.strongest && me.unitCount(UnitType.MissileSilo) > 0) { let tgt: number | null = null; for (const t of s.strongest.tiles()) { tgt = t; break; } if (tgt !== null) game.addExecution(new NukeExecution(UnitType.AtomBomb, me, tgt)); }
  else if (action === 10) { const ge = game.players().filter((p: any)=>p.isPlayer()&&p.isAlive()&&p.id()!=="agent"&&!me.isFriendly(p)).sort((a: any,b: any)=>a.troops()-b.troops())[0];
    if (ge) { let dst = -1; for (const t of ge.tiles()) { if (game.isShore(t)) { dst = t; break; } } if (dst < 0) for (const t of ge.tiles()) { dst = t; break; }
      if (dst >= 0 && canBuildTransportShip(game, me, dst) !== false) game.addExecution(new TransportShipExecution(me, dst, commit)); } }
  else if (action === 11 && s.shoreTile >= 0) build(UnitType.Port, s.shoreTile);
  else if (action === 12) build(UnitType.Factory, placeTile());   // learned placement
}

async function reset(seed: number): Promise<number[]> {
  const gameMap = await genTerrainFromBin(man[mc.game], mapBuf);
  const mini = await genTerrainFromBin(man[mc.mini], miniBuf);
  const cfg: any = { gameMap: GameMapType.World, gameMapSize: GameMapSize.Normal, gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer, difficulty: Difficulty.Medium, nations: "default",
    donateGold:false, donateTroops:false, bots: BOTS, infiniteGold:false, infiniteTroops:false, instantBuild:false, randomSpawn:false };
  const config = new Config(cfg, null as any, false);
  let s = seed >>> 0; const rand = () => (s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 0xffffffff;
  W = gameMap.width(); H = gameMap.height();
  const nations: Nation[] = [];
  if (mc.realNations) {
    // Real manifest nations scaled to the game-map resolution and snapped to the nearest land.
    const scaleX = W / man.map.width, scaleY = H / man.map.height;
    const snapLand = (x: number, y: number) => {
      for (let r = 0; r < 20; r++) for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
        const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
        if (gameMap.isLand(gameMap.ref(xx, yy))) return [xx, yy]; }
      return [Math.max(0, Math.min(W - 1, x)), Math.max(0, Math.min(H - 1, y))]; };
    const manNats: any[] = man.nations.filter((n: any) => n.coordinates);
    const useN = Math.min(NUM_NATIONS, manNats.length);
    const stride = useN >= manNats.length ? 1 : Math.ceil(manNats.length / useN);
    const chosen = manNats.filter((_, i) => i % stride === 0).slice(0, useN);
    for (const mn of chosen) { const [x, y] = snapLand(Math.floor(mn.coordinates[0] * scaleX), Math.floor(mn.coordinates[1] * scaleY));
      nations.push(new Nation(new Cell(x, y), new PlayerInfo(mn.name, PlayerType.Nation, null, "nat" + nations.length))); }
  }
  // Fabricate extra nations at random land to reach NUM_NATIONS (maps ship with few; we want realistic density).
  while (nations.length < NUM_NATIONS) { let x, y, t; do { x = Math.floor(rand()*W); y = Math.floor(rand()*H); t = gameMap.ref(x,y); } while (!gameMap.isLand(t));
    nations.push(new Nation(new Cell(x, y), new PlayerInfo("Nat" + nations.length, PlayerType.Nation, null, "nat" + nations.length))); }
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
  me = game.player("agent");
  for (let k = 0; k < 15 && !me.isAlive(); k++) game.executeNextTick();  // ensure the agent has actually spawned before we start
  land = game.numLandTiles(); tick = 0; prevShare = me.isAlive() ? me.numTilesOwned()/land : 0;
  peakShare = prevShare; peakEcon = me.isAlive() ? econCount(me) : 0;
  return observe();
}

function step(action: number, troop: number, target: number, ptarget: number) {
  if (me.isAlive() && me.troops() > 1) act(action, troop, target, ptarget);
  for (let i = 0; i < DECIDE_EVERY; i++) { game.executeNextTick(); tick++; if (!me.isAlive() && tick > 50) break; }
  const alive = me.isAlive();
  const curShare = alive ? me.numTilesOwned()/land : 0;
  const curEcon = alive ? econCount(me) : peakEcon;
  // Territory-DOMINANT reward: expansion is the main dense signal. Economy is a SMALL, uncapped,
  // log-scaled bonus per new structure (money spent) — it keeps paying even late (never caps) but
  // stays too small to dominate, so PPO learns economy as a *means* to territory, not the goal.
  // Banked (peak-tracked) so losing/rebuilding can't farm it; no survival term (that bred turtling).
  let reward = 4 * (curShare - prevShare)
    + 0.15 * Math.max(0, Math.log1p(curEcon) - Math.log1p(peakEcon));
  prevShare = curShare;
  if (alive) { peakShare = Math.max(peakShare, curShare); peakEcon = Math.max(peakEcon, curEcon); }
  const aliveP = game.players().filter((p: any) => p.isPlayer() && p.isAlive()).length;
  const done = !alive || tick >= MAX_TICKS || aliveP <= 1;
  if (done) {
    reward += 8 * peakShare;                                   // BANK the peak territory (dominant term)
    if (alive && me.numTilesOwned() >= 0.8*land) reward += 10; // winning is the biggest payoff by far
    if (!alive) reward -= 0.15;                                // modest death penalty (< peak bonus, so aggression pays)
  }
  return { obs: observe(), cands: candidates(), ptiles: placeTiles(), reward, done };
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", async (line) => {
  if (!line.trim()) return;
  const cmd = JSON.parse(line);
  let resp: any;
  if (cmd.cmd === "reset") resp = { obs: await reset(cmd.seed ?? 0), cands: candidates(), ptiles: placeTiles(), reward: 0, done: false };
  else if (cmd.cmd === "step") resp = step(cmd.action, cmd.troop, cmd.target ?? 0, cmd.ptarget ?? -1);
  else resp = { error: "unknown cmd" };
  process.stdout.write(JSON.stringify(resp) + "\n");
});
