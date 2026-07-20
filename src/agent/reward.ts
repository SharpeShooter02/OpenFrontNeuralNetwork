// The REWARD scores how well one game went - a single number that training will try to
// maximize. Reward EXACTLY what you want the agent to do, or it learns the wrong thing
// (reward hacking). We want it to grow big, survive, and ideally win.

export interface EpisodeStats {
  peakLandShare: number; // largest fraction of the map we ever controlled (0..1)
  survived: boolean;     // still alive at the end of the game?
  won: boolean;          // did we actually win?
}

export function computeReward(s: EpisodeStats): number {
  return (
    s.peakLandShare +          // main signal: how big did we get
    (s.survived ? 0.5 : 0) +   // bonus for not dying
    (s.won ? 5 : 0)            // big bonus for winning
  );
}
