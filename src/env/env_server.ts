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
  box:       { rel: "resources/maps/thebox",          game: "map16x", mini: "map16x", realNations: true },  // fast proxy (512x512)
  box4x:     { rel: "resources/maps/thebox",          game: "map4x",  mini: "map16x", realNations: true },  // mid (1024x1024)
  box_full:  { rel: "resources/maps/thebox",          game: "map",    mini: "map4x",  realNations: true },  // FULL SIZE (2048x2048) — eval target (mini must be 2x = map4x)
};
const mc = MAPS[MAP];
// [nations, tribes]. box tuned to real density that fits; full-size gets a near-real crowd (400 tribe cap).
const DEF: any = { world: [15, 100], bigplains: [6, 24], box: [20, 160], box4x: [40, 300], box_full: [60, 400] };
// Opponent skill: DIFFICULTY=easy|medium|hard|impossible (goal is Impossible).
const DIFF = (({ easy: Difficulty.Easy, medium: Difficulty.Medium, hard: Difficulty.Hard, impossible: Difficulty.Impossible } as any)[(process.env.DIFFICULTY ?? "medium").toLowerCase()]) ?? Difficulty.Medium;
const NUM_NATIONS = +(process.env.NUM_NATIONS ?? DEF[MAP][0]), BOTS = +(process.env.BOTS ?? DEF[MAP][1]), MAX_TICKS = 12000;
const DECIDE_EVERY = +(process.env.DECIDE_EVERY ?? 10);   // ticks between decisions — lower = more actions/game (throughput)
const GW = 32, GC = 6;   // spatial observation: GWxGW pooled grid, GC channels (mine/enemy/neutral/impassable/myStruct/enemyStruct)
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

// Spatial observation: the map as GC channels of a GWxGW pooled grid (the agent's "eyes"). Channels:
// 0 mine, 1 enemy, 2 neutral land, 3 impassable/water, 4 my structures, 5 enemy structures. Ownership
// is sampled at each pooled cell's center (cheap, ~GW^2 samples); structures are stamped from units().
// Flattened as grid[c*GW*GW + gy*GW + gx]. Only computed when the driver asked for it (SPATIAL flag).
let SPATIAL = false;
function observeSpatial(): number[] {
  const g = new Array(GC * GW * GW).fill(0);
  if (!me.isAlive()) return g;
  const sx = W / GW, sy = H / GW, mySmall = me.smallID(), area = GW * GW;
  for (let gy = 0; gy < GW; gy++) for (let gx = 0; gx < GW; gx++) {
    const x = Math.min(W-1, Math.floor((gx+0.5)*sx)), y = Math.min(H-1, Math.floor((gy+0.5)*sy)), t = game.ref(x, y), idx = gy*GW + gx;
    if (!game.isLand(t) || game.isImpassable(t)) { g[3*area + idx] = 1; continue; }
    if (!game.hasOwner(t)) { g[2*area + idx] = 1; continue; }
    const oid = game.ownerID(t);
    if (oid === mySmall) g[idx] = 1;
    else { const o = game.playerBySmallID(oid); g[(o.isPlayer() && me.isFriendly(o) ? 0 : 1)*area + idx] = o.isPlayer() && me.isFriendly(o) ? 0.5 : 1; }
  }
  for (const u of game.units([UnitType.City, UnitType.DefensePost, UnitType.MissileSilo, UnitType.SAMLauncher, UnitType.Port, UnitType.Factory])) {
    if (!u.isActive()) continue; const tt = u.tile();
    const gx = Math.min(GW-1, Math.floor(game.x(tt)/sx)), gy = Math.min(GW-1, Math.floor(game.y(tt)/sy)), idx = gy*GW + gx;
    g[(u.owner().smallID() === mySmall ? 4 : 5)*area + idx] = 1;
  }
  return g;
}

// --- Optional replay recording for the viewer (works for ANY driver, incl. the CNN). Enabled per-reset
// via {record:true}; snapshots pooled tile-ownership + buildings/gold/allies each step and writes
// viz/replay.js on episode end (same format run_agent.ts produces).
const PAL = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#800000","#808000","#000075"];  // vivid = NATIONS
const TRIBE_PAL = ["#5a5f66","#665f54","#54615a","#5f5461","#586066","#615a4c","#4f5560","#605050"];  // muted/desaturated = TRIBES
const TCr: any = { [UnitType.City]:1, [UnitType.Port]:2, [UnitType.Factory]:3, [UnitType.MissileSilo]:4, [UnitType.DefensePost]:5, [UnitType.SAMLauncher]:6 };
let REC: any = null;
function recInit() {
  const DS = Math.max(1, Math.ceil(Math.max(W, H) / 240)), RW = Math.ceil(W/DS), RH = Math.ceil(H/DS);
  const terrain = new Uint8Array(RW*RH);
  for (let ry=0; ry<RH; ry++) for (let rx=0; rx<RW; rx++) { const t=game.ref(Math.min(W-1,rx*DS),Math.min(H-1,ry*DS)); terrain[ry*RW+rx]=game.isLand(t)&&!game.isImpassable(t)?1:0; }
  REC = { DS, RW, RH, terrain, idToIdx:new Map(), legend:[] as any[], deltas:[] as string[], frameTicks:[] as number[], buildingFrames:[] as number[][], goldFrames:[] as number[][], allyFrames:[] as number[][], prev:new Uint8Array(RW*RH) };
  recSnap();
}
function recSnap() {
  const R = REC; const { DS, RW, RH } = R;
  const ownerAt = (x:number,y:number)=>{ const t=game.ref(x,y); if(!game.hasOwner(t)) return 0; const sid=game.ownerID(t); let k=R.idToIdx.get(sid);
    if(k===undefined){k=R.legend.length+1;R.idToIdx.set(sid,k);const p:any=game.playerBySmallID(sid);
      const isAgent=p?.id?.()==="agent"; const kind=isAgent?"agent":(p?.type?.()===PlayerType.Nation?"nation":"tribe");
      const color=isAgent?"#ffffff":(kind==="nation"?PAL[(k-1)%PAL.length]:TRIBE_PAL[(k-1)%TRIBE_PAL.length]);
      R.legend.push({name:p?.name?.()??("#"+sid),color,kind});} return k; };
  const cur=new Uint8Array(RW*RH);
  for(let ry=0;ry<RH;ry++)for(let rx=0;rx<RW;rx++)cur[ry*RW+rx]=ownerAt(Math.min(W-1,rx*DS),Math.min(H-1,ry*DS));
  if(me.isAlive()){let ak=R.idToIdx.get(me.smallID());if(ak===undefined){ak=R.legend.length+1;R.idToIdx.set(me.smallID(),ak);R.legend.push({name:"AGENT",color:"#ffffff",kind:"agent"});}for(const t of me.tiles())cur[Math.min(RH-1,Math.floor(game.y(t)/DS))*RW+Math.min(RW-1,Math.floor(game.x(t)/DS))]=ak;}
  const ch:number[]=[];for(let i=0;i<cur.length;i++)if(cur[i]!==R.prev[i])ch.push(i,cur[i]);
  const nn=ch.length/2;const buf=new Uint8Array(nn*5);const dv=new DataView(buf.buffer);for(let k=0;k<nn;k++){dv.setUint32(k*5,ch[k*2],true);dv.setUint8(k*5+4,ch[k*2+1]);}
  R.deltas.push(Buffer.from(buf).toString("base64"));R.frameTicks.push(game.ticks());R.prev=cur;
  const bld:number[]=[];for(const u of game.units()){if(!u.isActive())continue;const c=TCr[u.type()];if(!c)continue;const tt=u.tile();const rx=Math.min(RW-1,Math.floor(game.x(tt)/DS)),ry=Math.min(RH-1,Math.floor(game.y(tt)/DS));const k=R.idToIdx.get(u.owner().smallID());bld.push(ry*RW+rx,c,k??0);}R.buildingFrames.push(bld);
  const gold:number[]=new Array(R.legend.length).fill(0);for(const pl of game.players()){if(!pl.isPlayer())continue;const k=R.idToIdx.get(pl.smallID());if(k!==undefined)gold[k-1]=Math.round(Number(pl.gold()));}R.goldFrames.push(gold);
  R.allyFrames.push(me.isAlive()?me.allies().map((a:any)=>R.idToIdx.get(a.smallID())).filter((k:any)=>k!==undefined):[]);
}
function recWrite() {
  const R=REC; const outDir=path.join(dir,"../../viz"); fs.mkdirSync(outDir,{recursive:true});
  const payload={W:R.RW,H:R.RH,interval:DECIDE_EVERY,winner:"(agent)",terrain:Buffer.from(R.terrain).toString("base64"),legend:R.legend,frameTicks:R.frameTicks,deltas:R.deltas,buildingFrames:R.buildingFrames,goldFrames:R.goldFrames,allyFrames:R.allyFrames};
  fs.writeFileSync(path.join(outDir,"replay.js"),"window.REPLAY = "+JSON.stringify(payload)+";");
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

// Multi-action: the policy fires ANY SUBSET of the 13 actions each decision (throughput, like the
// nations do), and keeps a learned RESERVE of troops (discipline). Committable = troops*(1-reserve)
// is split evenly among the fired troop-attacks (expand/weak/strong/boat); builds/diplomacy/nuke are
// free of troop cost. `actions` is a 0/1 array of length 13.
function actMulti(actions: number[], reserve: number, target: number, ptarget: number) {
  const s = scan();
  const committable = Math.floor(me.troops() * (1 - Math.max(0, Math.min(1, reserve))));
  const firedAtk = [0, 1, 2, 10].filter((a) => actions[a]);
  const per = firedAtk.length ? Math.max(1, Math.floor(committable / firedAtk.length)) : 0;
  const ownedTile = () => { if (game.ownerID(spawn) === me.smallID()) return spawn; for (const t of me.tiles()) return t; return spawn; };
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
  const placeTile = () => (ptarget >= 0 && ptarget < curPlace.length) ? curPlace[ptarget] : buildTile();
  if (actions[0] && s.empty && per > 0) game.addExecution(new AttackExecution(per, me, game.terraNullius().id()));
  if (actions[1] && s.weakest && per > 0) game.addExecution(new AttackExecution(per, me, s.weakest.id()));
  if (actions[2] && s.strongest && per > 0) game.addExecution(new AttackExecution(per, me, s.strongest.id()));
  if (actions[3] && target > 0 && target < curCands.length) {   // diplomacy on the SCORED player
    const c = curCands[target];
    if (c.kind === "offer") { try { c.req.accept(); } catch {} }
    else if (c.kind === "enemy") { if (c.player.isAlive() && me.canSendAllianceRequest(c.player)) me.createAllianceRequest(c.player); }
    else if (c.kind === "ally") { for (const a of me.alliances()) if (a.other(me) === c.player) { me.breakAlliance(a); break; } }
  }
  // actions[4] = wait
  if (actions[5]) build(UnitType.City, buildTile());
  if (actions[6]) build(UnitType.DefensePost, placeTile());
  if (actions[7]) build(UnitType.MissileSilo, placeTile());
  if (actions[8]) build(UnitType.SAMLauncher, placeTile());
  if (actions[9] && s.strongest && me.unitCount(UnitType.MissileSilo) > 0) { let tgt: number | null = null; for (const t of s.strongest.tiles()) { tgt = t; break; } if (tgt !== null) game.addExecution(new NukeExecution(UnitType.AtomBomb, me, tgt)); }
  if (actions[10] && per > 0) { const ge = game.players().filter((p: any)=>p.isPlayer()&&p.isAlive()&&p.id()!=="agent"&&!me.isFriendly(p)).sort((a: any,b: any)=>a.troops()-b.troops())[0];
    if (ge) { let dst = -1; for (const t of ge.tiles()) { if (game.isShore(t)) { dst = t; break; } } if (dst < 0) for (const t of ge.tiles()) { dst = t; break; }
      if (dst >= 0 && canBuildTransportShip(game, me, dst) !== false) game.addExecution(new TransportShipExecution(me, dst, per)); } }
  if (actions[11] && s.shoreTile >= 0) build(UnitType.Port, s.shoreTile);
  if (actions[12]) build(UnitType.Factory, placeTile());
}

async function reset(seed: number): Promise<number[]> {
  const gameMap = await genTerrainFromBin(man[mc.game], mapBuf);
  const mini = await genTerrainFromBin(man[mc.mini], miniBuf);
  const cfg: any = { gameMap: GameMapType.World, gameMapSize: GameMapSize.Normal, gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer, difficulty: DIFF, nations: "default",
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

function step(actions: number[], reserve: number, target: number, ptarget: number) {
  if (me.isAlive() && me.troops() > 1) actMulti(actions, reserve, target, ptarget);
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
  if (REC) { recSnap(); if (done) recWrite(); }
  return { obs: observe(), cands: candidates(), ptiles: placeTiles(), ...(SPATIAL ? { spatial: observeSpatial() } : {}), reward, done };
}

const rl = readline.createInterface({ input: process.stdin });
rl.on("line", async (line) => {
  if (!line.trim()) return;
  const cmd = JSON.parse(line);
  let resp: any;
  if (cmd.cmd === "reset") { SPATIAL = !!cmd.spatial; const obs0 = await reset(cmd.seed ?? 0); REC = null; if (cmd.record) recInit(); resp = { obs: obs0, cands: candidates(), ptiles: placeTiles(), ...(SPATIAL ? { spatial: observeSpatial() } : {}), reward: 0, done: false }; }
  else if (cmd.cmd === "step") {
    let acts = cmd.actions, reserve = cmd.reserve;
    if (!acts) { acts = new Array(13).fill(0); acts[cmd.action] = 1; reserve = 1 - (cmd.troop ?? 0.5); }  // back-compat: single action -> one-hot
    resp = step(acts, reserve ?? 0.5, cmd.target ?? 0, cmd.ptarget ?? -1);
  }
  else resp = { error: "unknown cmd" };
  process.stdout.write(JSON.stringify(resp) + "\n");
});
