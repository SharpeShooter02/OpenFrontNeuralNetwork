// Watch the agent on the FULL WORLD map. Loads trained weights if present. Tribe-heavy
// opponents; records territory + buildings + gold, and stamps the agent's tiles so it's
// always visible even when small.
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
import { Policy, ACTIONS, setFlat } from "../agent/policy";
import { computeReward } from "../agent/reward";

const _warn = console.warn.bind(console);
console.warn = (...a: any[]) => { if (typeof a[0] === "string" && a[0].startsWith("cannot build")) return; _warn(...a); };

const SEED = +(process.env.SEED ?? 777);          // e.g. SEED=90001 to watch held-out validation world #1
const gameID = "env_" + SEED;                     // mirror env_server's reset exactly (tribe placement + spawn)
let rs = SEED >>> 0; const rand = () => (rs = (Math.imul(rs, 1103515245) + 12345) >>> 0) / 0xffffffff;
// Map selection mirrors env_server (MAP=box default). Watch the trained agent on the same maps it trains on.
const MAP = process.env.MAP ?? "box";
const MAPS: any = {
  world:     { rel: "tests/testdata/maps/world",      game: "map4x",  mini: "map16x", realNations: true },
  bigplains: { rel: "tests/testdata/maps/big_plains", game: "map",    mini: "map4x",  realNations: false },
  box:       { rel: "resources/maps/thebox",          game: "map16x", mini: "map16x", realNations: true },
};
const mc = MAPS[MAP];
const DEF: any = { world: [15, 100], bigplains: [6, 24], box: [6, 15] };
const NUM_NATIONS = +(process.env.NUM_NATIONS ?? DEF[MAP][0]), BOTS = +(process.env.BOTS ?? DEF[MAP][1]);
const FRAME_EVERY = 30, DECISION_EVERY = 20, MAX_TICKS = 12000, MAX_GAME_MS = 60000;
const dir = path.dirname(fileURLToPath(import.meta.url));
const md = path.join(dir, "../../vendor/OpenFrontIO/" + mc.rel);
const man = JSON.parse(fs.readFileSync(path.join(md, "manifest.json"), "utf8"));
const gameMap = await genTerrainFromBin(man[mc.game], fs.readFileSync(path.join(md, mc.game + ".bin")));
const mini = await genTerrainFromBin(man[mc.mini], fs.readFileSync(path.join(md, mc.mini + ".bin")));

const cfg: any = { gameMap: GameMapType.World, gameMapSize: GameMapSize.Normal, gameMode: GameMode.FFA,
  gameType: GameType.Singleplayer, difficulty: Difficulty.Medium, nations: "default",
  donateGold: false, donateTroops: false, bots: BOTS, infiniteGold: false, infiniteTroops: false, instantBuild: false, randomSpawn: false };
const config = new Config(cfg, null as any, false);
const W = gameMap.width(), H = gameMap.height();
const nations: Nation[] = [];
if (mc.realNations) {
  const scaleX = W / man.map.width, scaleY = H / man.map.height;
  const snapLand = (x: number, y: number) => {
    for (let r = 0; r < 20; r++) for (let dy = -r; dy <= r; dy++) for (let dx = -r; dx <= r; dx++) {
      const xx = x + dx, yy = y + dy; if (xx < 0 || yy < 0 || xx >= W || yy >= H) continue;
      if (gameMap.isLand(gameMap.ref(xx, yy))) return [xx, yy]; }
    return [Math.max(0, Math.min(W - 1, x)), Math.max(0, Math.min(H - 1, y))]; };
  const manNats: any[] = man.nations.filter((n: any) => n.coordinates);
  const stride = NUM_NATIONS >= manNats.length ? 1 : Math.ceil(manNats.length / NUM_NATIONS);
  const chosen = manNats.filter((_, i) => i % stride === 0).slice(0, NUM_NATIONS);
  for (const mn of chosen) { const [x, y] = snapLand(Math.floor(mn.coordinates[0] * scaleX), Math.floor(mn.coordinates[1] * scaleY));
    nations.push(new Nation(new Cell(x, y), new PlayerInfo(mn.name, PlayerType.Nation, null, "nat" + nations.length))); }
} else {
  for (let i = 0; i < NUM_NATIONS; i++) { let x, y, t; do { x = Math.floor(rand()*W); y = Math.floor(rand()*H); t = gameMap.ref(x,y); } while (!gameMap.isLand(t));
    nations.push(new Nation(new Cell(x, y), new PlayerInfo("Nat" + i, PlayerType.Nation, null, "nat" + i))); }
}

const game = createGame([], nations, gameMap, mini, config);
const AGENT_ID = "agent";
const agentInfo = new PlayerInfo("AGENT", PlayerType.Human, null, AGENT_ID);
const exec = new Executor(game, gameID, undefined);
game.addExecution(...exec.nationExecutions());
game.addExecution(...exec.spawnTribes(BOTS));
game.addExecution(new WinCheckExecution());
for (let sp = 0; sp < 150; sp++) game.executeNextTick();
game.endSpawnPhase();
game.addPlayer(agentInfo);
const tx = Math.floor(rand()*W), ty = Math.floor(rand()*H);   // seed-varied spawn point (matches env_server)
let spawnTile = -1, bd = Infinity;
for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) { const t = game.ref(x, y);
  if (game.isLand(t) && !game.isImpassable(t) && !game.hasOwner(t)) { const d = (x-tx)*(x-tx)+(y-ty)*(y-ty); if (d < bd) { bd = d; spawnTile = t; } } }
game.addExecution(new SpawnExecution(gameID, agentInfo, spawnTile));
const me = game.player(AGENT_ID);
for (let k = 0; k < 15 && !me.isAlive(); k++) game.executeNextTick();  // ensure the agent has spawned (mirrors env_server)
const OPPONENTS = NUM_NATIONS + BOTS;

const policy = new Policy(16, 24, 13);
// Prefer the torch-trained weights (exported by train_torch/export_weights.py), else the ES weights.
const torchPath = path.join(dir, "../../data/torch_weights.json");
const esPath = path.join(dir, "../../data/best_weights.json");
const wPath = fs.existsSync(torchPath) ? torchPath : esPath;
if (fs.existsSync(wPath)) { try { setFlat(policy, JSON.parse(fs.readFileSync(wPath, "utf8"))); console.log(`loaded weights: ${path.basename(wPath)}`); } catch { console.log("weights unreadable; random"); } }
else console.log("no trained weights; random policy");

const DS = Math.max(1, Math.ceil(Math.max(W, H) / 240));
const RW = Math.ceil(W / DS), RH = Math.ceil(H / DS);
const PAL = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#800000","#808000","#000075"];
const TC: any = { [UnitType.City]:1, [UnitType.Port]:2, [UnitType.Factory]:3, [UnitType.MissileSilo]:4, [UnitType.DefensePost]:5, [UnitType.SAMLauncher]:6 };
const terrain = new Uint8Array(RW * RH);
for (let ry = 0; ry < RH; ry++) for (let rx = 0; rx < RW; rx++) { const t = game.ref(Math.min(W-1,rx*DS), Math.min(H-1,ry*DS)); terrain[ry*RW+rx] = game.isLand(t) && !game.isImpassable(t) ? 1 : 0; }
const idToIdx = new Map<number, number>();
const legend: { name: string; color: string }[] = [];
const deltas: string[] = []; const frameTicks: number[] = []; const buildingFrames: number[][] = []; const goldFrames: number[][] = []; const allyFrames: number[][] = [];
let prev = new Uint8Array(RW * RH);
function ownerAt(x: number, y: number): number { const t = game.ref(x, y); if (!game.hasOwner(t)) return 0; const sid = game.ownerID(t);
  let k = idToIdx.get(sid); if (k === undefined) { k = legend.length + 1; idToIdx.set(sid, k); const p: any = game.playerBySmallID(sid);
    legend.push({ name: p?.name?.() ?? `#${sid}`, color: p?.id?.() === AGENT_ID ? "#ffffff" : PAL[(k-1)%PAL.length] }); } return k; }
function snapshot() {
  const cur = new Uint8Array(RW * RH);
  for (let ry = 0; ry < RH; ry++) for (let rx = 0; rx < RW; rx++) cur[ry*RW+rx] = ownerAt(Math.min(W-1,rx*DS), Math.min(H-1,ry*DS));
  if (me.isAlive()) { let ak = idToIdx.get(me.smallID()); if (ak === undefined) { ak = legend.length+1; idToIdx.set(me.smallID(), ak); legend.push({ name: "AGENT", color: "#ffffff" }); }
    for (const t of me.tiles()) cur[Math.min(RH-1,Math.floor(game.y(t)/DS))*RW + Math.min(RW-1,Math.floor(game.x(t)/DS))] = ak; }
  const ch: number[] = []; for (let i = 0; i < cur.length; i++) if (cur[i] !== prev[i]) ch.push(i, cur[i]);
  const nn = ch.length/2; const buf = new Uint8Array(nn*5); const dv = new DataView(buf.buffer);
  for (let k = 0; k < nn; k++) { dv.setUint32(k*5, ch[k*2], true); dv.setUint8(k*5+4, ch[k*2+1]); }
  deltas.push(Buffer.from(buf).toString("base64")); frameTicks.push(game.ticks()); prev = cur;
  const bld: number[] = []; for (const u of game.units()) { if (!u.isActive()) continue; const c = TC[u.type()]; if (!c) continue; const tt = u.tile();
    const rx = Math.min(RW-1,Math.floor(game.x(tt)/DS)), ry = Math.min(RH-1,Math.floor(game.y(tt)/DS)); const k = idToIdx.get(u.owner().smallID()); bld.push(ry*RW+rx, c, k ?? 0); }
  buildingFrames.push(bld);
  const gold: number[] = new Array(legend.length).fill(0);
  for (const pl of game.players()) { if (!pl.isPlayer()) continue; const k = idToIdx.get(pl.smallID()); if (k !== undefined) gold[k-1] = Math.round(Number(pl.gold())); }
  goldFrames.push(gold);
  allyFrames.push(me.allies().map((a:any)=>idToIdx.get(a.smallID())).filter((k:any)=>k!==undefined));
}

console.log(`WORLD ${W}x${H} | seed ${SEED} | ${NUM_NATIONS} nations + ${BOTS} tribes`);
const startMs = performance.now();
let tick = 0, peakTiles = 0, lastAlive = 0;
for (; tick < MAX_TICKS; tick++) {
  if (me.isAlive() && tick % DECISION_EVERY === 0 && me.troops() > 1) {
    let empty = false, coastal = 0, shoreTile = -1; const enemies = new Set<Player>();
    for (const t of me.borderTiles()) {
      if (game.isShore(t)) { coastal = 1; if (shoreTile < 0) shoreTile = t; }
      game.forEachNeighbor(t, (nb) => {
        if (!game.isLand(nb) || game.isImpassable(nb)) return;
        if (game.ownerID(nb) === me.smallID()) return;
        if (!game.hasOwner(nb)) { empty = true; return; }
        const o = game.playerBySmallID(game.ownerID(nb)); if (o.isPlayer() && !me.isFriendly(o)) enemies.add(o);
      });
    }
    const sorted = [...enemies].sort((a, b) => a.troops() - b.troops());
    const weakest = sorted[0], strongest = sorted[sorted.length - 1];
    const troopsN = Math.max(1, me.troops());
    const allyTroops = me.allies().reduce((a: number, p: any) => a + p.troops(), 0);
    const sumTroops = sorted.reduce((a, p) => a + p.troops(), 0);
    let offererTroops = 0; for (const rq of me.incomingAllianceRequests()) offererTroops = Math.max(offererTroops, rq.requestor().troops());
    const obs = [me.numTilesOwned()/game.numLandTiles(), Math.min(1,me.troops()/200000),
      Math.log1p(Number(me.gold()))/Math.log1p(25_000_000),
      game.players().filter(p=>p.isPlayer()&&p.isAlive()&&p.id()!==AGENT_ID).length/OPPONENTS, empty?1:0,
      Math.min(1,enemies.size/6), weakest?Math.min(1,me.troops()/Math.max(1,weakest.troops())/2):1,
      Math.min(1, me.allies().length/5), me.incomingAllianceRequests().length>0?1:0,
      Math.min(1, me.unitCount(UnitType.City)/8), me.unitCount(UnitType.MissileSilo)>0?1:0, coastal,
      strongest?Math.min(1,me.troops()/Math.max(1,strongest.troops())/2):1,
      Math.min(1, sumTroops/troopsN/3), Math.min(1, allyTroops/troopsN),
      offererTroops>0?Math.min(1, offererTroops/troopsN/2):0];
    // Greedy (argmax) for a deterministic, representative "best play" replay — the JS sampler
    // doesn't match torch's, so sampling here would show an unrepresentative one-off rollout.
    const { action, troopFraction } = policy.forward(obs);
    const commit = Math.floor(me.troops() * Math.max(0.01, Math.min(1, troopFraction)));
    // Build on a currently-owned tile (the fixed spawn tile is often captured by mid-game).
    const ownedTile = () => { if (game.ownerID(spawnTile) === me.smallID()) return spawnTile; for (const t of me.tiles()) return t; return spawnTile; };
    // Spread structures out (far from existing ones) so structureMinDist doesn't reject the next build.
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
    if (action === 0 && empty) game.addExecution(new AttackExecution(commit, me, game.terraNullius().id()));
    else if (action === 1 && weakest) game.addExecution(new AttackExecution(commit, me, weakest.id()));
    else if (action === 2 && strongest) game.addExecution(new AttackExecution(commit, me, strongest.id()));
    else if (action === 3) { for (const req of me.incomingAllianceRequests()) req.accept(); }
    else if (action === 4) { for (const e of enemies) if (e.troops() > me.troops() && me.canSendAllianceRequest(e)) me.createAllianceRequest(e); }
    else if (action === 5) build(UnitType.City, buildTile());
    else if (action === 6) build(UnitType.DefensePost, buildTile());
    else if (action === 7) build(UnitType.MissileSilo, buildTile());
    else if (action === 8) build(UnitType.SAMLauncher, buildTile());
    else if (action === 9 && strongest && me.unitCount(UnitType.MissileSilo) > 0) { let tgt: number | null = null; for (const t of strongest.tiles()) { tgt = t; break; } if (tgt !== null) game.addExecution(new NukeExecution(UnitType.AtomBomb, me, tgt)); }
    else if (action === 10) { const ge = game.players().filter(p=>p.isPlayer()&&p.isAlive()&&p.id()!==AGENT_ID&&!me.isFriendly(p)).sort((a,b)=>a.troops()-b.troops())[0];
      if (ge) { let dst = -1; for (const t of ge.tiles()) { if (game.isShore(t)) { dst = t; break; } } if (dst < 0) for (const t of ge.tiles()) { dst = t; break; }
        if (dst >= 0 && canBuildTransportShip(game, me, dst) !== false) game.addExecution(new TransportShipExecution(me, dst, commit)); } }
    else if (action === 11 && shoreTile >= 0) build(UnitType.Port, shoreTile);
    else if (action === 12) build(UnitType.Factory, buildTile());
    if (tick % 1000 === 0) console.log(`tick ${tick}: ${ACTIONS[action]} | tiles=${me.numTilesOwned()} gold=${me.gold()}`);
  }
  game.executeNextTick();
  if (me.isAlive()) { peakTiles = Math.max(peakTiles, me.numTilesOwned()); lastAlive = tick; }
  if (tick % FRAME_EVERY === 0) snapshot();
  if (!me.isAlive() && tick > 50) break;
  if (tick % 200 === 0 && performance.now() - startMs > MAX_GAME_MS) break;
  const aliveP = game.players().filter((p) => p.isPlayer() && p.isAlive());
  if (aliveP.length <= 1) break;
}
snapshot();
const land = game.numLandTiles();
const reward = computeReward({ peakLandShare: peakTiles/land, survivalFraction: lastAlive/MAX_TICKS, survived: me.isAlive(), won: me.isAlive() && me.numTilesOwned() >= 0.8*land });
console.log(`REWARD = ${reward.toFixed(4)} (${me.isAlive() ? "survived to tick " + tick : "died at tick " + tick}, peakTiles=${peakTiles})`);
const outDir = path.join(dir, "../../viz"); fs.mkdirSync(outDir, { recursive: true });
const payload = { W: RW, H: RH, interval: FRAME_EVERY, winner: "(world)", terrain: Buffer.from(terrain).toString("base64"), legend, frameTicks, deltas, buildingFrames, goldFrames, allyFrames };
fs.writeFileSync(path.join(outDir, "replay.js"), "window.REPLAY = " + JSON.stringify(payload) + ";");
console.log(`wrote viz/replay.js (${deltas.length} frames)`);
