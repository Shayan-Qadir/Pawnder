export interface UserProfile {
  username: string;
  total_games: number;
  losses_analyzed: number;
  blunders_extracted: number;
  weaknesses: string[];
  sample_blunders: any[];
}

export interface Scenario {
  scenario_id: string;
  fen: string;
  player_color: 'white' | 'black';
  clock_remaining: number;
  clock_bucket: string;
  opponent_rating_band: string;
  target_elo: number;
  reason: string;
}

export interface PracticeMoveResult {
  is_legal: boolean;
  engine_move_uci?: string;
  engine_move_san?: string;
  fen_after: string;
  evaluation_cp: number;
  game_over: boolean;
  status: string;
}

const API_BASE = '/api';

export async function importUser(username: string): Promise<UserProfile> {
  const res = await fetch(`${API_BASE}/user/import`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, max_months: 2 }),
  });
  if (!res.ok) throw new Error('Failed to import user');
  return res.json();
}

export async function predictScenario(
  username?: string,
  playerRating = 1300,
  useBaseline = false
): Promise<Scenario> {
  const res = await fetch(`${API_BASE}/scenario/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, player_rating: playerRating, use_baseline: useBaseline }),
  });
  if (!res.ok) throw new Error('Failed to predict scenario');
  return res.json();
}

export async function submitMove(
  fen: string,
  moveUci: string,
  targetElo = 1400
): Promise<PracticeMoveResult> {
  const res = await fetch(`${API_BASE}/practice/move`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ fen, move_uci: moveUci, target_elo: targetElo }),
  });
  if (!res.ok) throw new Error('Failed to submit move');
  return res.json();
}

export async function sendFeedback(
  scenarioId: string,
  username: string,
  result: 'win' | 'loss' | 'draw',
  score = 5
) {
  return fetch(`${API_BASE}/practice/feedback`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      scenario_id: scenarioId,
      username,
      result,
      user_feedback_score: score,
    }),
  });
}
