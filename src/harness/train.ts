// TRAINING via a simple (1+1) evolution strategy: mutate the weights, play a game,
// keep the mutation only if it scored higher reward. Watch REWARD climb.
import fs from "fs"; import path from "path"; import { fileURLToPath } from "url";
import { Config } from "../../vendor/OpenFrontIO/src/core/configuration/Config";
import { createGame } from "../../vendor/OpenFrontIO/src/core/game/GameImpl";
import { genTerrainFromBin } from "../../vendor/OpenFrontIO/src/core/game/TerrainMapLoader";
import { Executor } from "../../vendor/OpenFrontIO/src/core/execution/ExecutionManager";
import { WinCheckExecution } from "../../vendor/OpenFrontIO/src/core/execution/WinCheckExecution";
import { SpawnExecution } from "../../vendor/OpenFrontIO/src/core/execution/SpawnExecution";
import { AttackExecution } from "../../vendor/OpenFrontIO/src/core/execution/AttackExecution";
import { Difficulty, GameMapType, GameMapSize, GameMode, GameType, Player, PlayerInfo, PlayerType } from "../../vendor/OpenFrontIO/src/core/game/Game";
import { Policy, getFlat, setFlat } from "../agent/policy";
import { computeReward } from "../agent/reward";

console.warn = () => {};
const dir = path.dirname(fileURLToPath(import.meta.url));
const md = path.join(dir, "../../vendor/OpenFrontIO/tests/testdata/maps/plains");
const man = JSON.parse(fs.readFileSync(path.join(md, "manifest.json"), "utf8"));
const mapBuf = fs.readFileSync(path.join(md, "map.bin"));
const miniBuf = fs.readFileSync(path.join(md, "map4x.bin"));

async function playGame(policy: Policy, rec?: { frames: string[]; ticks: number[]; terrain: Uint8Array; legend: {name:string;color:string}[]; W:number; H:number }): Promise<number> {
  const gameMap = await genTerrainFromBin(man.map, mapBuf);
  const mini = await genTerrainFromBin(man.map4x, miniBuf);
  const cfg: any = { gameMap: GameMapType.Plains, gameMapSize: GameMapSize.Normal, gameMode: GameMode.FFA,
    gameType: GameType.Singleplayer, difficulty: Difficulty.Medium, nations: "default",
    donateGold:false, donateTroops:false, bots: 6, infiniteGold:false, infiniteTroops:false, instantBuild:false, randomSpawn:false };
  const config = new Config(cfg, null as any, false);
  const game = createGame([], [], gameMap, mini, config);
  const exec = new Executor(game, "train", undefined);
  if (config.bots() > 0) game.addExecution(...exec.spawnTribes(config.bots()));
  game.addExecution(new WinCheckExecution());
  game.endSpawnPhase();
  const info = new PlayerInfo("AGENT", PlayerType.Human, null, "agent");
  game.addPlayer(info);
  const cx = 50, cy = 50; let spawn = game.ref(cx, cy);
  for (let r = 0; r < 100 && !(game.isLand(spawn) && !game.hasOwner(spawn)); r++) spawn = game.ref(cx + r, cy);
  game.addExecution(new SpawnExecution("train", info, spawn));
  const me = game.player("agent");
  const land = game.numLandTiles();
  let peak = 0, started = false;
  const W = game.width(), H = game.height();
  const idToIdx = new Map<number, number>();
  let prev = new Uint8Array(W * H);
  const PAL = ["#e6194B","#3cb44b","#ffe119","#4363d8","#f58231","#911eb4","#42d4f4","#f032e6","#bfef45","#fabed4","#469990","#dcbeff","#9A6324","#800000"];
  function snap() {
    if (!rec) return;
    const cur = new Uint8Array(W * H);
    for (let y=0;y<H;y++) for (let x=0;x<W;x++){ const t=game.ref(x,y), i=y*W+x;
      if(!game.hasOwner(t)) continue; const sid=game.ownerID(t); let k=idToIdx.get(sid);
      if(k===undefined){ k=rec.legend.length+1; idToIdx.set(sid,k); const pp:any=game.playerBySmallID(sid);
        rec.legend.push({name:pp?.name?.()??("#"+sid), color: pp?.id?.()==="agent" ? "#ffffff" : PAL[(k-1)%PAL.length]}); }
      cur[i]=k; }
    const ch:number[]=[]; for(let i=0;i<cur.length;i++) if(cur[i]!==prev[i]) ch.push(i,cur[i]);
    const n=ch.length/2; const buf=new Uint8Array(n*5); const dv=new DataView(buf.buffer);
    for(let j=0;j<n;j++){ dv.setUint32(j*5,ch[j*2],true); dv.setUint8(j*5+4,ch[j*2+1]); }
    rec.frames.push(Buffer.from(buf).toString("base64")); rec.ticks.push(game.ticks()); prev=cur;
  }
  for (let tick = 0; tick < 4000; tick++) {
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
        game.players().filter(p=>p.isPlayer()&&p.isAlive()&&p.id()!=="agent").length/6, empty?1:0,
        Math.min(1,enemies.size/6), weakest?Math.min(1,me.troops()/Math.max(1,weakest.troops())/2):1];
      const { action } = policy.forward(obs);
      if (action === 0 && empty) game.addExecution(new AttackExecution(me.troops()/2, me, game.terraNullius().id()));
      else if (action === 1 && weakest) game.addExecution(new AttackExecution(me.troops()/3, me, weakest.id()));
    }
    game.executeNextTick();
    if (rec && tick % 20 === 0) snap();
    if (me.isAlive()) peak = Math.max(peak, me.numTilesOwned());
    if (!me.isAlive() && tick > 50) break;
    const nAlive = game.players().filter(p=>p.isPlayer()&&p.isAlive()).length;
    if (nAlive > 1) started = true;
    if (started && nAlive <= 1) break;
  }
  if (rec) { snap(); for (let y=0;y<H;y++) for (let x=0;x<W;x++){ const t=game.ref(x,y); rec.terrain[y*W+x]=game.isLand(t)&&!game.isImpassable(t)?1:0; } }
  const survived = me.isAlive();
  return computeReward({ peakLandShare: peak/land, survived, won: survived && me.numTilesOwned() >= 0.8*land });
}

// (1+1) evolution strategy
const policy = new Policy(7);
let best = getFlat(policy);
setFlat(policy, best); let bestScore = await playGame(policy);
console.log(`gen  0: reward ${bestScore.toFixed(4)}`);
const SIGMA = 0.2, GENERATIONS = 60;
let rng = 12345; const rand = () => (rng = (Math.imul(rng,1103515245)+12345)>>>0)/0xffffffff;
const gauss = () => Math.sqrt(-2*Math.log(1-rand()))*Math.cos(2*Math.PI*rand());
for (let gen = 1; gen <= GENERATIONS; gen++) {
  const cand = best.map(w => w + gauss()*SIGMA);
  setFlat(policy, cand);
  const score = await playGame(policy);
  if (score > bestScore) { best = cand; bestScore = score; }
  if (gen % 10 === 0) console.log(`gen ${String(gen).padStart(2)}: best reward ${bestScore.toFixed(4)}`);
}
console.log(`\nFINAL best reward ${bestScore.toFixed(4)} (started ~0)`);

// save the trained weights
const dataDir = path.join(dir, "../../data"); fs.mkdirSync(dataDir, { recursive: true });
fs.writeFileSync(path.join(dataDir, "best_weights.json"), JSON.stringify(best));
console.log("saved trained weights -> data/best_weights.json");

// record one game with the trained weights so we can watch it
setFlat(policy, best);
const g0 = createGame([], [], await genTerrainFromBin(man.map, mapBuf), await genTerrainFromBin(man.map4x, miniBuf), new Config({} as any, null as any, false));
const rec = { frames: [] as string[], ticks: [] as number[], terrain: new Uint8Array(g0.width()*g0.height()), legend: [] as {name:string;color:string}[], W: g0.width(), H: g0.height() };
await playGame(policy, rec);
const vizDir = path.join(dir, "../../viz"); fs.mkdirSync(vizDir, { recursive: true });
const payload = { W: rec.W, H: rec.H, interval: 20, winner: "(trained agent)", terrain: Buffer.from(rec.terrain).toString("base64"), legend: rec.legend, frameTicks: rec.ticks, deltas: rec.frames };
fs.writeFileSync(path.join(vizDir, "replay.js"), "window.REPLAY = " + JSON.stringify(payload) + ";");
console.log(`recorded trained game -> viz/replay.js (${rec.frames.length} frames). Open viz/index.html.`);
