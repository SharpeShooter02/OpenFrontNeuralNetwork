// TRAINING vs Nations + tribes on the FULL WORLD map (water + continents).
// (1+1) evolution strategy with multi-seed evaluation. Games end when the agent dies,
// or at MAX_TICKS, or after MAX_GAME_MS wall-clock (safety cap).
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
import { Policy, getFlat, setFlat } from "../agent/policy";
import { computeReward } from "../agent/reward";

console.warn = () => {};
const NUM_NATIONS = 10, BOTS = 50; // tribe-heavy (5:1)
const MAX_TICKS = 12000, MAX_GAME_MS = 30000; // end game on agent death, or these caps
const dir = path.dirname(fileURLToPath(import.meta.url));
const md = path.join(dir, "../../vendor/OpenFrontIO/tests/testdata/maps/world");
const man = JSON.parse(fs.readFileSync(path.join(md, "manifest.json"), "utf8"));
const mapBuf = fs.readFileSync(path.join(md, "map4x.bin")), miniBuf = fs.readFileSync(path.join(md, "map16x.bin"));

function seededRand(seed: number) { let s = seed >>> 0; return () => (s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 0xffffffff; }

async function playGame(policy: Policy, seed: number, rec?: any): Promise<number> {
  const gameMap = await genTerrainFromBin(man.map4x, mapBuf);
  const mini = await genTerrainFromBin(man.map16x, miniBuf);
  const cfg: any = { gameMap: GameMapType.World, gameMapSize: GameMapSize.Normal, gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer, difficulty: Difficulty.Medium, nations: "default",
    donateGold:false, donateTroops:false, bots: BOTS, infiniteGold:false, infiniteTroops:false, instantBuild:false, randomSpawn:false };
  const config = new Config(cfg, null as any, false);
  const rand = seededRand(seed);
  const W = gameMap.width(), H = gameMap.height();
  const nations: Nation[] = [];
  for (let i = 0; i < NUM_NATIONS; i++) { let x, y, t; do { x = Math.floor(rand()*W); y = Math.floor(rand()*H); t = gameMap.ref(x,y); } while (!gameMap.isLand(t)); nations.push(new Nation(new Cell(x,y), new PlayerInfo("Nat"+i, PlayerType.Nation, null, "nat"+i))); }
  const game = createGame([], nations, gameMap, mini, config);
  const gid = "train_" + seed;
  const exec = new Executor(game, gid, undefined);
  game.addExecution(...exec.nationExecutions());
  game.addExecution(...exec.spawnTribes(BOTS));
  game.addExecution(new WinCheckExecution());
  for (let sp = 0; sp < 150; sp++) game.executeNextTick();
  game.endSpawnPhase();
  const info = new PlayerInfo("AGENT", PlayerType.Human, null, "agent");
  game.addPlayer(info);
  // spawn at the nearest unowned land to a seeded random point (varied but valid)
  const tx = Math.floor(rand()*W), ty = Math.floor(rand()*H);
  let spawn = -1, bd = Infinity;
  for (let y = 0; y < H; y += 2) for (let x = 0; x < W; x += 2) { const t = game.ref(x, y);
    if (game.isLand(t) && !game.isImpassable(t) && !game.hasOwner(t)) { const d = (x-tx)*(x-tx)+(y-ty)*(y-ty); if (d < bd) { bd = d; spawn = t; } } }
  game.addExecution(new SpawnExecution(gid, info, spawn));
  const me = game.player("agent");
  const land = game.numLandTiles();
  const OPP = NUM_NATIONS + BOTS;
  let peak = 0, started = false, lastAlive = 0;

  // ---- optional replay recording ----
  const idToIdx = new Map<number, number>(); let prevGrid = new Uint8Array(W * H);
  const DS = rec ? Math.max(1, Math.ceil(Math.max(W, H) / 240)) : 1;
  const RW = Math.ceil(W / DS), RH = Math.ceil(H / DS);
  const PAL = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#800000"];
  const TC: any = { [UnitType.City]:1, [UnitType.Port]:2, [UnitType.Factory]:3, [UnitType.MissileSilo]:4, [UnitType.DefensePost]:5, [UnitType.SAMLauncher]:6 };
  const ownerAt = (x: number, y: number) => { const t = game.ref(x, y); if (!game.hasOwner(t)) return 0; const sid = game.ownerID(t);
    let k = idToIdx.get(sid); if (k === undefined) { k = rec.legend.length + 1; idToIdx.set(sid, k); const pp: any = game.playerBySmallID(sid);
      rec.legend.push({ name: pp?.name?.() ?? ("#"+sid), color: pp?.id?.() === "agent" ? "#ffffff" : PAL[(k-1)%PAL.length] }); } return k; };
  const snap = () => { if (!rec) return; const cur = new Uint8Array(RW*RH);
    for (let ry=0; ry<RH; ry++) for (let rx=0; rx<RW; rx++) cur[ry*RW+rx] = ownerAt(Math.min(W-1,rx*DS), Math.min(H-1,ry*DS));
    // stamp the agent's own tiles so it is always visible even when tiny
    if (me.isAlive()) { let ak = idToIdx.get(me.smallID()); if (ak === undefined) { ak = rec.legend.length+1; idToIdx.set(me.smallID(), ak); rec.legend.push({ name: "AGENT", color: "#ffffff" }); }
      for (const t of me.tiles()) cur[Math.min(RH-1,Math.floor(game.y(t)/DS))*RW + Math.min(RW-1,Math.floor(game.x(t)/DS))] = ak; }
    const ch:number[]=[]; for(let i=0;i<cur.length;i++) if(cur[i]!==prevGrid[i]) ch.push(i,cur[i]);
    const nn=ch.length/2; const buf=new Uint8Array(nn*5); const dv=new DataView(buf.buffer);
    for(let j=0;j<nn;j++){dv.setUint32(j*5,ch[j*2],true);dv.setUint8(j*5+4,ch[j*2+1]);}
    rec.frames.push(Buffer.from(buf).toString("base64")); rec.ticks.push(game.ticks()); prevGrid=cur;
    const bld:number[]=[]; for(const u of game.units()){ if(!u.isActive())continue; const c=TC[u.type()]; if(!c)continue; const tt=u.tile();
      const rx=Math.min(RW-1,Math.floor(game.x(tt)/DS)), ry=Math.min(RH-1,Math.floor(game.y(tt)/DS)); const k=idToIdx.get(u.owner().smallID()); bld.push(ry*RW+rx,c,k??0); }
    rec.buildingFrames.push(bld);
    const gold:number[]=new Array(rec.legend.length).fill(0);
    for(const pl of game.players()){ if(!pl.isPlayer())continue; const k=idToIdx.get(pl.smallID()); if(k!==undefined) gold[k-1]=Math.round(Number(pl.gold())); }
    rec.goldFrames.push(gold);
    rec.allyFrames.push(me.allies().map((a:any)=>idToIdx.get(a.smallID())).filter((k:any)=>k!==undefined)); };

  const t0 = performance.now();
  for (let tick = 0; tick < MAX_TICKS; tick++) {
    if (me.isAlive() && tick % 20 === 0 && me.troops() > 1) {
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
      const obs = [me.numTilesOwned()/land, Math.min(1,me.troops()/200000), Math.min(1,Number(me.gold())/200000),
        game.players().filter(p=>p.isPlayer()&&p.isAlive()&&p.id()!=="agent").length/OPP, empty?1:0,
        Math.min(1,enemies.size/6), weakest?Math.min(1,me.troops()/Math.max(1,weakest.troops())/2):1,
        Math.min(1, me.allies().length/5), me.incomingAllianceRequests().length>0?1:0,
        Math.min(1, me.unitCount(UnitType.City)/8), me.unitCount(UnitType.MissileSilo)>0?1:0, coastal];
      for (const req of me.incomingAllianceRequests()) req.accept();
      const { action, troopFraction } = policy.forward(obs);
      const commit = Math.floor(me.troops() * Math.max(0.01, Math.min(1, troopFraction)));
      const build = (u: UnitType, tile: number) => { const bt = me.canBuild(u, tile); if (bt) game.addExecution(new ConstructionExecution(me, u, bt)); };
      if (action === 0 && empty) game.addExecution(new AttackExecution(commit, me, game.terraNullius().id()));
      else if (action === 1 && weakest) game.addExecution(new AttackExecution(commit, me, weakest.id()));
      else if (action === 2 && strongest) game.addExecution(new AttackExecution(commit, me, strongest.id()));
      // action 3 = wait
      else if (action === 4) { for (const e of enemies) if (me.canSendAllianceRequest(e)) me.createAllianceRequest(e); }
      else if (action === 5) build(UnitType.City, spawn);
      else if (action === 6) build(UnitType.DefensePost, spawn);
      else if (action === 7) build(UnitType.MissileSilo, spawn);
      else if (action === 8) build(UnitType.SAMLauncher, spawn);
      else if (action === 9 && strongest && me.unitCount(UnitType.MissileSilo) > 0) { let tgt: number | null = null; for (const t of strongest.tiles()) { tgt = t; break; } if (tgt !== null) game.addExecution(new NukeExecution(UnitType.AtomBomb, me, tgt)); }
      else if (action === 10) { // boat to weakest enemy across water
        const ge = game.players().filter(p=>p.isPlayer()&&p.isAlive()&&p.id()!=="agent"&&!me.isFriendly(p)).sort((a,b)=>a.troops()-b.troops())[0];
        if (ge) { let dst = -1; for (const t of ge.tiles()) { if (game.isShore(t)) { dst = t; break; } } if (dst < 0) for (const t of ge.tiles()) { dst = t; break; }
          if (dst >= 0 && canBuildTransportShip(game, me, dst) !== false) game.addExecution(new TransportShipExecution(me, dst, commit)); }
      }
      else if (action === 11 && shoreTile >= 0) build(UnitType.Port, shoreTile);
    }
    game.executeNextTick();
    if (rec && tick % 20 === 0) snap();
    if (me.isAlive()) { peak = Math.max(peak, me.numTilesOwned()); lastAlive = tick; }
    if (!me.isAlive() && tick > 50) break;                              // end on agent death
    if (tick % 200 === 0 && performance.now() - t0 > MAX_GAME_MS) break; // wall-clock cap
    const n = game.players().filter(p=>p.isPlayer()&&p.isAlive()).length;
    if (n > 1) started = true; if (started && n <= 1) break;
  }
  if (rec) { snap(); rec.terrain = new Uint8Array(RW*RH); for (let ry=0;ry<RH;ry++) for (let rx=0;rx<RW;rx++){ const t=game.ref(Math.min(W-1,rx*DS),Math.min(H-1,ry*DS)); rec.terrain[ry*RW+rx]=game.isLand(t)&&!game.isImpassable(t)?1:0; } rec.W=RW; rec.H=RH; }
  const survived = me.isAlive();
  return computeReward({ peakLandShare: peak/land, survivalFraction: lastAlive/MAX_TICKS, survived, won: survived && me.numTilesOwned() >= 0.8*land });
}

async function evaluate(policy: Policy, seeds: number[]): Promise<number> {
  let sum = 0; for (const s of seeds) sum += await playGame(policy, s); return sum / seeds.length;
}

const K = 3, GENERATIONS = 50, SIGMA = 0.2;   // games per eval, generations, mutation size
const VAL = [9001, 9002, 9003];
const policy = new Policy(12, 16, 12);        // 12 obs -> 12 actions
let best = getFlat(policy);
setFlat(policy, best);
console.log(`gen  0: validation reward ${(await evaluate(policy, VAL)).toFixed(3)}`);
let rng = 42; const rnd = () => (rng = (Math.imul(rng,1103515245)+12345)>>>0)/0xffffffff;
const gauss = () => Math.sqrt(-2*Math.log(1-rnd()))*Math.cos(2*Math.PI*rnd());
for (let gen = 1; gen <= GENERATIONS; gen++) {
  const seeds = Array.from({ length: K }, (_, i) => gen * 7 + 1 + i);
  setFlat(policy, best); const bestScore = await evaluate(policy, seeds);
  const cand = best.map(w => w + gauss()*SIGMA);
  setFlat(policy, cand); const candScore = await evaluate(policy, seeds);
  if (candScore > bestScore) best = cand;
  if (gen % 5 === 0) { setFlat(policy, best); console.log(`gen ${String(gen).padStart(2)}: validation reward ${(await evaluate(policy, VAL)).toFixed(3)}`); }
}
setFlat(policy, best);
console.log(`FINAL validation reward ${(await evaluate(policy, VAL)).toFixed(3)}`);
const dataDir = path.join(dir, "../../data"); fs.mkdirSync(dataDir, { recursive: true });
fs.writeFileSync(path.join(dataDir, "best_weights.json"), JSON.stringify(best));
console.log("saved trained weights -> data/best_weights.json");
const rec: any = { frames: [], ticks: [], terrain: new Uint8Array(1), legend: [], W: 0, H: 0, buildingFrames: [], goldFrames: [], allyFrames: [] };
await playGame(policy, VAL[0], rec);
const vizDir = path.join(dir, "../../viz"); fs.mkdirSync(vizDir, { recursive: true });
const payload = { W: rec.W, H: rec.H, interval: 20, winner: "(trained on world)", terrain: Buffer.from(rec.terrain).toString("base64"), legend: rec.legend, frameTicks: rec.ticks, deltas: rec.frames, buildingFrames: rec.buildingFrames, goldFrames: rec.goldFrames, allyFrames: rec.allyFrames };
fs.writeFileSync(path.join(vizDir, "replay.js"), "window.REPLAY = " + JSON.stringify(payload) + ";");
console.log(`recorded trained game -> viz/replay.js (${rec.frames.length} frames)`);
