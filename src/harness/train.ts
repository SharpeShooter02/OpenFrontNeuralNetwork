// TRAINING vs NATIONS on a medium map, with multi-seed evaluation (so it can't overfit
// to one game). (1+1) evolution strategy: mutate weights, play several games, keep the
// mutation only if its AVERAGE reward beats the current best on the same seeds.
import fs from "fs"; import path from "path"; import { fileURLToPath } from "url";
import { Config } from "../../vendor/OpenFrontIO/src/core/configuration/Config";
import { createGame } from "../../vendor/OpenFrontIO/src/core/game/GameImpl";
import { genTerrainFromBin } from "../../vendor/OpenFrontIO/src/core/game/TerrainMapLoader";
import { Executor } from "../../vendor/OpenFrontIO/src/core/execution/ExecutionManager";
import { WinCheckExecution } from "../../vendor/OpenFrontIO/src/core/execution/WinCheckExecution";
import { SpawnExecution } from "../../vendor/OpenFrontIO/src/core/execution/SpawnExecution";
import { AttackExecution } from "../../vendor/OpenFrontIO/src/core/execution/AttackExecution";
import { Cell, Difficulty, GameMapType, GameMapSize, GameMode, GameType, Nation, Player, PlayerInfo, PlayerType } from "../../vendor/OpenFrontIO/src/core/game/Game";
import { Policy, getFlat, setFlat } from "../agent/policy";
import { computeReward } from "../agent/reward";

console.warn = () => {};
const NUM_NATIONS = 10, BOTS = 10;
const dir = path.dirname(fileURLToPath(import.meta.url));
const md = path.join(dir, "../../vendor/OpenFrontIO/tests/testdata/maps/big_plains");
const man = JSON.parse(fs.readFileSync(path.join(md, "manifest.json"), "utf8"));
const mapBuf = fs.readFileSync(path.join(md, "map.bin")), miniBuf = fs.readFileSync(path.join(md, "map4x.bin"));

function seededRand(seed: number) { let s = seed >>> 0; return () => (s = (Math.imul(s, 1103515245) + 12345) >>> 0) / 0xffffffff; }

async function playGame(policy: Policy, seed: number, rec?: any): Promise<number> {
  const gameMap = await genTerrainFromBin(man.map, mapBuf);
  const mini = await genTerrainFromBin(man.map4x, miniBuf);
  const cfg: any = { gameMap: GameMapType.Plains, gameMapSize: GameMapSize.Normal, gameMode: GameMode.FFA,
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
  let spawn = game.ref(Math.floor(W/2), Math.floor(H/2));
  for (let r = 0; r < W && !(game.isLand(spawn) && !game.hasOwner(spawn)); r++) spawn = game.ref(Math.floor(W/2)+r, Math.floor(H/2));
  game.addExecution(new SpawnExecution(gid, info, spawn));
  const me = game.player("agent");
  const land = game.numLandTiles();
  const OPP = NUM_NATIONS + BOTS;
  let peak = 0, started = false;
  const idToIdx = new Map<number, number>(); let prevGrid = new Uint8Array(W * H);
  const PAL = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#800000"];
  const snap = () => { if (!rec) return; const cur = new Uint8Array(W*H);
    for (let y=0;y<H;y++) for (let x=0;x<W;x++){ const t=game.ref(x,y), i=y*W+x; if(!game.hasOwner(t)) continue; const sid=game.ownerID(t);
      let k=idToIdx.get(sid); if(k===undefined){ k=rec.legend.length+1; idToIdx.set(sid,k); const pp:any=game.playerBySmallID(sid);
        rec.legend.push({name:pp?.name?.()??("#"+sid), color: pp?.id?.()==="agent"?"#ffffff":PAL[(k-1)%PAL.length]}); } cur[i]=k; }
    const ch:number[]=[]; for(let i=0;i<cur.length;i++) if(cur[i]!==prevGrid[i]) ch.push(i,cur[i]);
    const n=ch.length/2; const buf=new Uint8Array(n*5); const dv=new DataView(buf.buffer);
    for(let j=0;j<n;j++){dv.setUint32(j*5,ch[j*2],true);dv.setUint8(j*5+4,ch[j*2+1]);}
    rec.frames.push(Buffer.from(buf).toString("base64")); rec.ticks.push(game.ticks()); prevGrid=cur; };
  for (let tick = 0; tick < 6000; tick++) {
    if (me.isAlive() && tick % 20 === 0 && me.troops() > 1) {
      let empty = false; const enemies = new Set<Player>();
      for (const t of me.borderTiles()) game.forEachNeighbor(t, (nb) => {
        if (!game.isLand(nb) || game.isImpassable(nb)) return;
        if (game.ownerID(nb) === me.smallID()) return;
        if (!game.hasOwner(nb)) { empty = true; return; }
        const o = game.playerBySmallID(game.ownerID(nb)); if (o.isPlayer() && !me.isFriendly(o)) enemies.add(o);
      });
      const weakest = [...enemies].sort((a, b) => a.troops() - b.troops())[0];
      const obs = [me.numTilesOwned()/land, Math.min(1,me.troops()/200000), Math.min(1,Number(me.gold())/200000),
        game.players().filter(p=>p.isPlayer()&&p.isAlive()&&p.id()!=="agent").length/OPP, empty?1:0,
        Math.min(1,enemies.size/6), weakest?Math.min(1,me.troops()/Math.max(1,weakest.troops())/2):1,
        Math.min(1, me.allies().length/5), me.incomingAllianceRequests().length>0?1:0];
      // always accept alliance offers (free protection), then let the policy choose an action
      for (const req of me.incomingAllianceRequests()) req.accept();
      const { action } = policy.forward(obs);
      if (action === 0 && empty) game.addExecution(new AttackExecution(me.troops()/2, me, game.terraNullius().id()));
      else if (action === 1 && weakest) game.addExecution(new AttackExecution(me.troops()/3, me, weakest.id()));
      else if (action === 3) { for (const e of enemies) if (me.canSendAllianceRequest(e)) me.createAllianceRequest(e); }
    }
    game.executeNextTick();
    if (rec && tick % 20 === 0) snap();
    if (me.isAlive()) peak = Math.max(peak, me.numTilesOwned());
    if (!me.isAlive() && tick > 50) break;
    const n = game.players().filter(p=>p.isPlayer()&&p.isAlive()).length;
    if (n > 1) started = true; if (started && n <= 1) break;
  }
  if (rec) { snap(); for (let y=0;y<H;y++) for (let x=0;x<W;x++){ const t=game.ref(x,y); rec.terrain[y*W+x]=game.isLand(t)&&!game.isImpassable(t)?1:0; } rec.W=W; rec.H=H; }
  const survived = me.isAlive();
  return computeReward({ peakLandShare: peak/land, survived, won: survived && me.numTilesOwned() >= 0.8*land });
}

async function evaluate(policy: Policy, seeds: number[]): Promise<number> {
  let sum = 0; for (const s of seeds) sum += await playGame(policy, s); return sum / seeds.length;
}

// ---- (1+1) evolution strategy with multi-seed evaluation ----
const K = 3, GENERATIONS = 50, SIGMA = 0.2;   // games per eval, generations, mutation size
const VAL = [9001, 9002, 9003];               // fixed held-out seeds for progress readout
const policy = new Policy(9, 8, 4);
let best = getFlat(policy);
setFlat(policy, best);
console.log(`gen  0: validation reward ${(await evaluate(policy, VAL)).toFixed(3)}`);
let rng = 42; const rand = () => (rng = (Math.imul(rng,1103515245)+12345)>>>0)/0xffffffff;
const gauss = () => Math.sqrt(-2*Math.log(1-rand()))*Math.cos(2*Math.PI*rand());
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
const rec: any = { frames: [], ticks: [], terrain: new Uint8Array(1), legend: [], W: 0, H: 0 };
await playGame(policy, VAL[0], rec);
const vizDir = path.join(dir, "../../viz"); fs.mkdirSync(vizDir, { recursive: true });
const payload = { W: rec.W, H: rec.H, interval: 20, winner: "(trained vs Nations)", terrain: Buffer.from(rec.terrain).toString("base64"), legend: rec.legend, frameTicks: rec.ticks, deltas: rec.frames };
fs.writeFileSync(path.join(vizDir, "replay.js"), "window.REPLAY = " + JSON.stringify(payload) + ";");
console.log(`recorded trained game -> viz/replay.js (${rec.frames.length} frames). Open viz/index.html.`);
