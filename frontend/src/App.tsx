import React, { useState, useEffect } from 'react';
import { ChessBoard } from './components/ChessBoard';
import { ClockWidget } from './components/ClockWidget';
import { WeaknessSummary } from './components/WeaknessSummary';
import {
  importUser,
  predictScenario,
  submitMove,
  sendFeedback,
  UserProfile,
  Scenario,
} from './services/api';
import {
  BrainCircuit,
  Bot,
  Zap,
  RotateCcw,
  Play,
  CheckCircle2,
  XCircle,
  HelpCircle,
  Sliders,
} from 'lucide-react';

export function App() {
  const [username, setUsername] = useState('bonc_1');
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [currentFen, setCurrentFen] = useState<string>(
    'r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4'
  );
  const [playerColor, setPlayerColor] = useState<'white' | 'black'>('white');
  const [isPlayerTurn, setIsPlayerTurn] = useState<boolean>(true);
  const [isGameActive, setIsGameActive] = useState<boolean>(false);
  const [playerClock, setPlayerClock] = useState<number>(65);
  const [engineClock, setEngineClock] = useState<number>(65);
  const [targetElo, setTargetElo] = useState<number>(1400);
  const [evaluation, setEvaluation] = useState<number>(0);
  const [gameStatus, setGameStatus] = useState<string>('Ready to start practice scenario');
  const [useBaseline, setUseBaseline] = useState<boolean>(false);
  const [moveHistory, setMoveHistory] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  // Initialize with bonc_1 profile on mount
  useEffect(() => {
    handleLoadUser('bonc_1');
  }, []);

  const handleLoadUser = async (uname: string) => {
    setIsLoading(true);
    try {
      const prof = await importUser(uname);
      setProfile(prof);
      await handleGenerateScenario(uname);
    } catch (err) {
      console.error('Error loading user:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateScenario = async (uname = username) => {
    setIsLoading(true);
    try {
      const sc = await predictScenario(uname, targetElo, useBaseline);
      setScenario(sc);
      setCurrentFen(sc.fen);
      setPlayerColor(sc.player_color);
      setPlayerClock(sc.clock_remaining);
      setEngineClock(sc.clock_remaining);
      setTargetElo(sc.target_elo);
      setIsPlayerTurn(true);
      setIsGameActive(true);
      setMoveHistory([]);
      setGameStatus('Scenario loaded! Make your move against Stockfish.');
      setEvaluation(0);
    } catch (err) {
      console.error('Error generating scenario:', err);
    } finally {
      setIsLoading(false);
    }
  };

  const handleMove = async (moveUci: string) => {
    if (!isGameActive) return;

    setIsPlayerTurn(false);
    setGameStatus('Stockfish thinking...');

    try {
      const res = await submitMove(currentFen, moveUci, targetElo);
      if (!res.is_legal) {
        setGameStatus('Illegal move! Try another.');
        setIsPlayerTurn(true);
        return;
      }

      const nextMoves = [...moveHistory, moveUci];
      if (res.engine_move_san) {
        nextMoves.push(res.engine_move_san);
      }
      setMoveHistory(nextMoves);
      setCurrentFen(res.fen_after);
      setEvaluation(res.evaluation_cp);

      if (res.game_over) {
        setIsGameActive(false);
        setGameStatus(res.status);
        if (scenario) {
          sendFeedback(scenario.scenario_id, username, res.evaluation_cp > 0 ? 'win' : 'loss');
        }
      } else {
        setIsPlayerTurn(true);
        setGameStatus(`In play: Stockfish played ${res.engine_move_san || res.engine_move_uci}`);
      }
    } catch (e) {
      console.error('Move submission failed:', e);
      setGameStatus('Engine connection error');
      setIsPlayerTurn(true);
    }
  };

  const evalBarPercent = Math.min(100, Math.max(0, 50 + (evaluation / 1000) * 50));

  return (
    <div className="min-h-screen flex flex-col bg-[#161512] text-stone-200">
      {/* Top Navbar */}
      <header className="border-b border-[#363431] bg-[#1e1c19] px-6 py-3.5 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-[#81b64c] flex items-center justify-center font-bold text-black text-xl shadow-md">
            ♟
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              Pawnder <span className="text-xs bg-[#81b64c]/20 text-[#81b64c] border border-[#81b64c]/40 px-2 py-0.5 rounded-full font-mono">MLOps</span>
            </h1>
            <p className="text-xs text-stone-400">PressureLab History-Conditioned Scenario Practice</p>
          </div>
        </div>

        {/* Username Import Bar */}
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="Chess.com Username"
            className="bg-[#262421] border border-[#363431] focus:border-[#81b64c] text-white px-3 py-1.5 rounded-lg text-sm outline-none transition"
          />
          <button
            onClick={() => handleLoadUser(username)}
            disabled={isLoading}
            className="bg-[#81b64c] hover:bg-[#a3d160] text-black font-semibold px-4 py-1.5 rounded-lg text-sm flex items-center gap-1.5 transition shadow"
          >
            <Play className="w-4 h-4 fill-black" />
            Import & Practice
          </button>
        </div>
      </header>

      {/* Main Grid */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Weakness Summary & Controls */}
        <div className="lg:col-span-4 space-y-5">
          <WeaknessSummary
            profile={profile}
            onRefresh={() => handleLoadUser(username)}
            isLoading={isLoading}
          />

          {/* Practice & Model Settings */}
          <div className="bg-[#262421] border border-[#363431] rounded-xl p-5 shadow-xl space-y-4">
            <h3 className="text-sm font-bold uppercase tracking-wider text-stone-400 flex items-center gap-2">
              <Sliders className="w-4 h-4 text-[#81b64c]" />
              Model & Engine Configuration
            </h3>

            {/* Baseline toggle */}
            <div className="flex items-center justify-between p-3 bg-[#1e1c19] rounded-lg border border-[#363431]">
              <div>
                <span className="text-sm font-medium text-white block">Inference Architecture</span>
                <span className="text-xs text-stone-400">
                  {useBaseline ? 'kNN Baseline Matcher' : 'PyTorch History CNN+GRU Model'}
                </span>
              </div>
              <button
                onClick={() => setUseBaseline(!useBaseline)}
                className={`text-xs px-2.5 py-1 rounded font-semibold transition ${
                  useBaseline
                    ? 'bg-amber-600 text-white'
                    : 'bg-[#81b64c] text-black'
                }`}
              >
                {useBaseline ? 'Baseline kNN' : 'Neural Net'}
              </button>
            </div>

            {/* Target Opponent ELO Slider */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-stone-400">Stockfish Difficulty:</span>
                <span className="text-white font-mono font-bold">{targetElo} Elo</span>
              </div>
              <input
                type="range"
                min="1000"
                max="2400"
                step="50"
                value={targetElo}
                onChange={(e) => setTargetElo(Number(e.target.value))}
                className="w-full accent-[#81b64c] bg-[#363431] rounded-lg h-2 cursor-pointer"
              />
            </div>

            <button
              onClick={() => handleGenerateScenario(username)}
              disabled={isLoading}
              className="w-full bg-[#363431] hover:bg-[#464441] text-white font-medium py-2 rounded-lg text-sm flex items-center justify-center gap-2 transition"
            >
              <Zap className="w-4 h-4 text-amber-400" />
              Generate New Practice Scenario
            </button>
          </div>
        </div>

        {/* Center/Right Column: Chessboard & Play Arena */}
        <div className="lg:col-span-8 flex flex-col items-center justify-start space-y-4">
          {/* Scenario Context Banner */}
          {scenario && (
            <div className="w-full bg-[#262421] border border-[#363431] rounded-xl p-4 flex flex-wrap items-center justify-between gap-3 shadow-md">
              <div>
                <span className="text-xs font-semibold uppercase tracking-wider text-[#81b64c]">
                  Active Scenario #{scenario.scenario_id}
                </span>
                <h2 className="text-base font-bold text-white">{scenario.reason}</h2>
                <div className="flex items-center gap-3 text-xs text-stone-400 mt-1">
                  <span>Playing as: <strong className="text-white capitalize">{scenario.player_color}</strong></span>
                  <span>•</span>
                  <span>Opponent Band: <strong className="text-white">{scenario.opponent_rating_band}</strong></span>
                  <span>•</span>
                  <span>Time Class: <strong className="text-white">10+0 Rapid ({scenario.clock_bucket})</strong></span>
                </div>
              </div>
              <button
                onClick={() => {
                  setCurrentFen(scenario.fen);
                  setMoveHistory([]);
                  setIsGameActive(true);
                  setIsPlayerTurn(true);
                  setGameStatus('Scenario reset to initial state.');
                }}
                className="flex items-center gap-1.5 text-xs bg-[#1e1c19] hover:bg-[#363431] text-stone-300 px-3 py-1.5 rounded-lg border border-[#363431] transition"
              >
                <RotateCcw className="w-3.5 h-3.5" />
                Reset Scenario
              </button>
            </div>
          )}

          {/* Clocks & Board Container */}
          <div className="w-full max-w-xl space-y-3">
            {/* Opponent Clock */}
            <ClockWidget
              initialSeconds={engineClock}
              isActive={isGameActive && !isPlayerTurn}
              onTimeOut={() => {
                setIsGameActive(false);
                setGameStatus('Opponent flagged! You win on time.');
              }}
              label={`Stockfish (${targetElo} Elo)`}
            />

            {/* Chessboard View */}
            <div className="relative">
              <ChessBoard
                fen={currentFen}
                playerColor={playerColor}
                isPlayerTurn={isPlayerTurn && isGameActive}
                onMove={handleMove}
                disabled={!isGameActive}
              />
            </div>

            {/* Player Clock */}
            <ClockWidget
              initialSeconds={playerClock}
              isActive={isGameActive && isPlayerTurn}
              onTimeOut={() => {
                setIsGameActive(false);
                setGameStatus('Time out! You flagged under pressure.');
              }}
              label={`You (${username})`}
            />

            {/* Evaluation & Status Bar */}
            <div className="bg-[#262421] border border-[#363431] rounded-xl p-3.5 space-y-2">
              <div className="flex items-center justify-between text-xs font-semibold">
                <span className="text-stone-300 flex items-center gap-1.5">
                  <Bot className="w-4 h-4 text-[#81b64c]" />
                  {gameStatus}
                </span>
                <span className="font-mono text-stone-400">
                  Eval: {evaluation > 0 ? `+${(evaluation / 100).toFixed(2)}` : (evaluation / 100).toFixed(2)}
                </span>
              </div>

              {/* Evaluation Bar Meter */}
              <div className="w-full bg-stone-900 h-2 rounded-full overflow-hidden flex border border-[#363431]">
                <div
                  className="bg-white transition-all duration-300"
                  style={{ width: `${evalBarPercent}%` }}
                />
                <div
                  className="bg-stone-800 transition-all duration-300 flex-1"
                />
              </div>

              {/* Move history */}
              {moveHistory.length > 0 && (
                <div className="text-xs text-stone-400 pt-1 border-t border-[#363431] flex flex-wrap gap-1 font-mono">
                  <span className="text-stone-500">Continuations:</span>
                  {moveHistory.map((m, idx) => (
                    <span key={idx} className="bg-[#1e1c19] px-1.5 py-0.5 rounded text-white">
                      {idx % 2 === 0 ? `${Math.floor(idx / 2) + 1}. ` : ''}{m}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
