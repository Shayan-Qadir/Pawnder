import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Chess } from 'chess.js';
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
  SkipBack,
  SkipForward,
  Sliders,
  Loader2,
} from 'lucide-react';

interface Position {
  fen: string;
  san: string | null;
  evaluation: number;
}

export function App() {
  const [username, setUsername] = useState('');
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [scenario, setScenario] = useState<Scenario | null>(null);
  const [currentFen, setCurrentFen] = useState<string>('');
  const [playerColor, setPlayerColor] = useState<'white' | 'black'>('white');
  const [isPlayerTurn, setIsPlayerTurn] = useState<boolean>(true);
  const [isGameActive, setIsGameActive] = useState<boolean>(false);
  const [playerClock, setPlayerClock] = useState<number>(65);
  const [engineClock, setEngineClock] = useState<number>(65);
  const [targetElo, setTargetElo] = useState<number>(1400); // placeholder until a scenario loads
  const [clockEpoch, setClockEpoch] = useState<number>(0);
  const [evaluation, setEvaluation] = useState<number>(0);
  const [gameStatus, setGameStatus] = useState<string>('Ready to start practice scenario');
  const [useBaseline, setUseBaseline] = useState<boolean>(false);
  const [positions, setPositions] = useState<Position[]>([]);
  const [viewIndex, setViewIndex] = useState<number | null>(null);
  const [isImporting, setIsImporting] = useState<boolean>(false);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);
  const [isMoving, setIsMoving] = useState<boolean>(false);
  const isLoading = isImporting || isGenerating;
  const [moveAnimation, setMoveAnimation] = useState<{
    from: string;
    to: string;
    id: number;
  } | null>(null);
  const animationId = useRef(0);
  const engineDelayTimer = useRef<number | null>(null);

  const clearMoveAnimation = () => {
    const hadPendingEngineReply = engineDelayTimer.current !== null;
    if (engineDelayTimer.current) window.clearTimeout(engineDelayTimer.current);
    engineDelayTimer.current = null;
    setMoveAnimation(null);
    setIsMoving(false);
    // A history arrow or reset can cancel the 500 ms delay before Stockfish's
    // reply is shown. Restore control, otherwise the status stays stuck on
    // "Stockfish is choosing a reply..." and the board is unplayable.
    if (hadPendingEngineReply) {
      setIsPlayerTurn(true);
      setGameStatus('Reviewing the position — make your move.');
    }
  };

  const showMove = (from: string, to: string) => {
    animationId.current += 1;
    setMoveAnimation({ from, to, id: animationId.current });
  };

  // ---- Move-history review (arrow keys) ---------------------------------
  const stepBack = useCallback(() => {
    setViewIndex((i) => {
      if (positions.length < 2) return null;
      if (i === null) return positions.length - 2;
      return Math.max(0, i - 1);
    });
  }, [positions.length]);

  const stepForward = useCallback(() => {
    setViewIndex((i) => {
      if (i === null || positions.length < 2) return null;
      const next = i + 1;
      return next >= positions.length - 1 ? null : next;
    });
  }, [positions.length]);

  useEffect(
    () => () => {
      if (engineDelayTimer.current) window.clearTimeout(engineDelayTimer.current);
    },
    []
  );

  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (!scenario) return;
      const target = e.target as HTMLElement | null;
      if (target && ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName)) return;
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        clearMoveAnimation();
        stepBack();
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        clearMoveAnimation();
        stepForward();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [scenario, stepBack, stepForward]);

  const safeIndex =
    viewIndex !== null && viewIndex >= 0 && viewIndex < positions.length
      ? viewIndex
      : null;
  const isViewingHistory = safeIndex !== null;
  const displayFen = isViewingHistory ? positions[safeIndex].fen : currentFen;
  const displayEval = isViewingHistory ? positions[safeIndex].evaluation : evaluation;

  const handleLoadUser = async (uname: string) => {
    const name = uname.trim().toLowerCase();
    if (!name) return;

    setUsername(name);
    setIsImporting(true);
    setGameStatus('Importing your games and retraining your private model...');
    try {
      const prof = await importUser(name);
      setProfile(prof);
      // Re-import always produces a freshly trained private model, so the
      // scenario generated right after it uses the newest data.
      await handleGenerateScenario(name, prof.player_rating);
    } catch (err) {
      console.error('Error loading user:', err);
      setGameStatus('Import failed — check the username and try again.');
    } finally {
      setIsImporting(false);
    }
  };

  const handleGenerateScenario = async (
    uname = username,
    rating?: number
  ) => {
    if (!uname.trim()) return;
    clearMoveAnimation();
    setIsGenerating(true);
    setGameStatus('Generating a fresh practice scenario...');
    try {
      // No targetElo here: the scenario itself supplies the Stockfish rating.
      const sc = await predictScenario(
        uname,
        rating ?? profile?.player_rating ?? 1300,
        useBaseline
      );
      setScenario(sc);
      setCurrentFen(sc.fen);
      setPlayerColor(sc.player_color);
      setPlayerClock(sc.clock_remaining);
      setEngineClock(sc.clock_remaining);
      setTargetElo(sc.target_elo);
      setClockEpoch((n) => n + 1);
      setIsPlayerTurn(true);
      setIsGameActive(true);
      setPositions([{ fen: sc.fen, san: null, evaluation: 0 }]);
      setViewIndex(null);
      setGameStatus('Scenario loaded! Make your move against Stockfish.');
      setEvaluation(0);
    } catch (err) {
      console.error('Error generating scenario:', err);
      setGameStatus('Could not generate a scenario — try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const recordResult = (result?: string | null) => {
    if (!scenario || !result) return;
    const playerIsWhite = scenario.player_color === 'white';
    const outcome =
      result === '1-0'
        ? playerIsWhite ? 'win' : 'loss'
        : result === '0-1'
        ? playerIsWhite ? 'loss' : 'win'
        : 'draw';
    sendFeedback(scenario.scenario_id, username, outcome);
  };

  const handleMove = async (moveUci: string) => {
    if (!isGameActive) return;

    // If the user moved after stepping back with ←, treat the viewed position
    // as the new live position and branch from there ("play the move again").
    const replayIndex = isViewingHistory && safeIndex !== null ? safeIndex : null;
    const sourceFen = replayIndex !== null ? positions[replayIndex].fen : currentFen;
    if (replayIndex !== null) {
      clearMoveAnimation();
      setCurrentFen(sourceFen);
      setPositions((prev) => prev.slice(0, replayIndex + 1));
      setViewIndex(null);
      setIsPlayerTurn(true);
      setPlayerClock(scenario?.clock_remaining ?? 65);
      setEngineClock(scenario?.clock_remaining ?? 65);
      setClockEpoch((n) => n + 1);
      setGameStatus('Replaying from this position — make your move.');
    }

    setIsPlayerTurn(false);
    setIsMoving(true);
    setGameStatus('Submitting your move to Stockfish...');

    // Human-readable SAN for the replay list.
    let playerSan = moveUci;
    try {
      const boardForSan = new Chess(sourceFen);
      const promo =
        moveUci.length > 4
          ? (moveUci[4] as 'q' | 'r' | 'b' | 'n')
          : undefined;
      const mv = boardForSan.move({
        from: moveUci.slice(0, 2),
        to: moveUci.slice(2, 4),
        ...(promo ? { promotion: promo } : {}),
      });
      if (mv?.san) playerSan = mv.san;
    } catch {
      // keep the UCI string if chess.js cannot parse it
    }

    try {
      const res = await submitMove(sourceFen, moveUci, targetElo);
      if (!res.is_legal) {
        setGameStatus('Illegal move! Try another.');
        setIsMoving(false);
        setIsPlayerTurn(true);
        return;
      }

      const fenAfterPlayer = res.fen_after_player || res.fen_after;
      // 1) Show the player's move immediately, as a sliding piece.
      setCurrentFen(fenAfterPlayer);
      showMove(moveUci.slice(0, 2), moveUci.slice(2, 4));
      setPositions((prev) => [
        ...prev,
        { fen: fenAfterPlayer, san: playerSan, evaluation },
      ]);
      setEvaluation(res.evaluation_cp);

      const engineUci = res.engine_move_uci;
      if (engineUci && engineUci !== '0000') {
        // 2) Pause ~0.5 s, then show Stockfish's reply sliding in.
        setGameStatus('Stockfish is choosing a reply...');
        engineDelayTimer.current = window.setTimeout(() => {
          engineDelayTimer.current = null;
          setIsMoving(false);
          setCurrentFen(res.fen_after);
          showMove(engineUci.slice(0, 2), engineUci.slice(2, 4));
          setPositions((prev) => [
            ...prev,
            {
              fen: res.fen_after,
              san: res.engine_move_san || engineUci,
              evaluation: res.evaluation_cp,
            },
          ]);
          if (res.game_over) {
            setIsGameActive(false);
            setGameStatus(res.status);
            recordResult(res.result);
          } else {
            setIsPlayerTurn(true);
            setGameStatus(`In play: Stockfish played ${res.engine_move_san || engineUci}`);
          }
        }, 500);
      } else {
        // No engine reply (game already over after the player's move).
        setIsMoving(false);
        if (res.game_over) {
          setIsGameActive(false);
          setGameStatus(res.status);
          recordResult(res.result);
        } else {
          setIsPlayerTurn(true);
          setGameStatus(res.status);
        }
      }
    } catch (e) {
      console.error('Move submission failed:', e);
      setGameStatus('Engine connection error');
      setIsMoving(false);
      setIsPlayerTurn(true);
    }
  };

  const evalBarPercent = Math.min(100, Math.max(0, 50 + (displayEval / 1000) * 50));
  const moveChips = positions.slice(1).map((p, i) => ({
    ...p,
    label: `${i % 2 === 0 ? `${Math.floor(i / 2) + 1}. ` : ''}${p.san ?? ''}`,
    index: i + 1,
  }));

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

        {/* Username Import Bar — Chess.com is the main platform */}
        <div className="flex items-center gap-2">
          <span className="hidden sm:inline text-xs text-stone-400 font-medium mr-0.5">Chess.com</span>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                handleLoadUser(username);
              }
            }}
            placeholder="Chess.com Username (press Enter)"
            className="bg-[#262421] border border-[#363431] focus:border-[#81b64c] text-white px-3 py-1.5 rounded-lg text-sm outline-none transition"
          />
          <button
            onClick={() => handleLoadUser(username)}
            disabled={isLoading || !username.trim()}
            className="bg-[#81b64c] hover:bg-[#a3d160] disabled:opacity-50 disabled:cursor-not-allowed text-black font-semibold px-4 py-1.5 rounded-lg text-sm flex items-center gap-1.5 transition shadow"
          >
            {isImporting ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Play className="w-4 h-4 fill-black" />
            )}
            {isImporting ? 'Importing & Training...' : 'Import & Practice'}
          </button>
          {isLoading && (
            <span className="hidden md:inline text-[11px] text-[#81b64c] font-medium animate-pulse max-w-[210px] leading-tight">
              {isImporting
                ? 'Fetching games → Stockfish scan → private retrain'
                : 'Selecting a fresh blunder...'}
            </span>
          )}
        </div>
      </header>

      {/* Main Grid */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 md:p-6 grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Weakness Summary & Controls */}
        <div className="lg:col-span-4 space-y-5">
          <WeaknessSummary
            profile={profile}
            onRefresh={() => handleLoadUser(username)}
            isLoading={isImporting}
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

            {/* Scenario-driven Stockfish Elo */}
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-stone-400">Stockfish Difficulty (from scenario):</span>
                <span className="text-white font-mono font-bold">
                  {scenario ? `${targetElo} Elo` : '—'}
                </span>
              </div>
              <input
                type="range"
                min="1000"
                max="2400"
                step="50"
                value={targetElo}
                disabled={!scenario}
                onChange={(e) => setTargetElo(Number(e.target.value))}
                className="w-full accent-[#81b64c] bg-[#363431] rounded-lg h-2 cursor-pointer disabled:opacity-40"
              />
              <p className="text-[11px] text-stone-500 mt-1 leading-relaxed">
                Each drill sets its own strength from the opponent rating band and clock
                pressure. Adjust here to override it for the current game.
              </p>
            </div>

            <button
              onClick={() => handleGenerateScenario(username, profile?.player_rating)}
              disabled={isLoading || !username.trim()}
              className="w-full bg-[#363431] hover:bg-[#464441] disabled:opacity-40 disabled:cursor-not-allowed text-white font-medium py-2 rounded-lg text-sm flex items-center justify-center gap-2 transition"
            >
              {isGenerating ? (
                <Loader2 className="w-4 h-4 animate-spin text-[#81b64c]" />
              ) : (
                <Zap className="w-4 h-4 text-amber-400" />
              )}
              {isGenerating ? 'Generating Scenario...' : 'Generate New Practice Scenario'}
            </button>
            {isLoading && (
              <p className="text-[11px] text-[#81b64c] flex items-center justify-center gap-1.5">
                <Loader2 className="w-3 h-3 animate-spin" />
                {isImporting ? 'Retraining your private model...' : 'Loading a new position from your blunders...'}
              </p>
            )}
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
                  <span>•</span>
                  <span>Engine: <strong className="text-white">{targetElo} Elo</strong></span>
                </div>
              </div>
              <button
                onClick={() => {
                  clearMoveAnimation();
                  setClockEpoch((n) => n + 1);
                  setCurrentFen(scenario.fen);
                  setPositions([{ fen: scenario.fen, san: null, evaluation: 0 }]);
                  setViewIndex(null);
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

          {scenario && currentFen ? (
            /* Clocks & Board Container */
            <div className="w-full max-w-md space-y-3">
              {/* Opponent Clock */}
              <ClockWidget
                initialSeconds={engineClock}
                resetKey={clockEpoch}
                isActive={isGameActive && !isPlayerTurn && !isViewingHistory}
                onTimeOut={() => {
                  setIsGameActive(false);
                  setGameStatus('Opponent flagged! You win on time.');
                }}
                label={`Stockfish (${targetElo} Elo)`}
              />

              {/* Chessboard View */}
              <div className="relative">
                <ChessBoard
                  fen={displayFen}
                  playerColor={playerColor}
                  isPlayerTurn={isPlayerTurn && isGameActive}
                  onMove={handleMove}
                  disabled={!isGameActive || isLoading}
                  animateMove={isViewingHistory ? null : moveAnimation}
                  evalPercent={evalBarPercent}
                />
                {isLoading && (
                  <div className="absolute inset-0 z-30 flex flex-col items-center justify-center gap-2 rounded-md bg-[#161512]/85 backdrop-blur-[1px] text-center px-4">
                    <Loader2 className="w-8 h-8 animate-spin text-[#81b64c]" />
                    <span className="text-sm font-semibold text-white">
                      {isImporting ? 'Importing your games...' : 'Generating scenario...'}
                    </span>
                    <span className="text-[11px] text-stone-400 leading-relaxed">
                      {isImporting
                        ? 'Stockfish is scanning new games and retraining your private model.'
                        : 'Finding a fresh blunder from your own games.'}
                    </span>
                  </div>
                )}
              </div>

              {/* Player Clock */}
              <ClockWidget
                initialSeconds={playerClock}
                resetKey={clockEpoch}
                isActive={isGameActive && isPlayerTurn && !isViewingHistory}
                onTimeOut={() => {
                  setIsGameActive(false);
                  setGameStatus('Time out! You flagged under pressure.');
                }}
                label={username ? `You (${username})` : 'You'}
              />

              {/* Evaluation & Status Bar */}
              <div className="bg-[#262421] border border-[#363431] rounded-xl p-3.5 space-y-2">
                <div className="flex items-center justify-between text-xs font-semibold">
                  <span className="text-stone-300 flex items-center gap-1.5">
                    {isMoving || isLoading ? (
                      <Loader2 className="w-4 h-4 animate-spin text-[#81b64c]" />
                    ) : (
                      <Bot className="w-4 h-4 text-[#81b64c]" />
                    )}
                    {isViewingHistory
                      ? `Reviewing move ${safeIndex} of ${positions.length - 1} — move a piece to replay from here`
                      : gameStatus}
                  </span>
                  <span className="font-mono text-stone-400">
                    Eval: {displayEval > 0 ? `+${(displayEval / 100).toFixed(2)}` : (displayEval / 100).toFixed(2)}
                  </span>
                </div>

                {/* Replay controls */}
                <div className="flex items-center gap-2 text-xs text-stone-400 border-t border-[#363431] pt-2">
                  <button
                    onClick={stepBack}
                    disabled={!scenario || positions.length < 2 || (safeIndex !== null && safeIndex === 0)}
                    className="p-1 rounded bg-[#1e1c19] hover:bg-[#363431] disabled:opacity-30 text-stone-300"
                    title="Previous move (←)"
                  >
                    <SkipBack className="w-3.5 h-3.5" />
                  </button>
                  <button
                    onClick={stepForward}
                    disabled={!scenario || positions.length < 2 || safeIndex === null}
                    className="p-1 rounded bg-[#1e1c19] hover:bg-[#363431] disabled:opacity-30 text-stone-300"
                    title="Next move (→)"
                  >
                    <SkipForward className="w-3.5 h-3.5" />
                  </button>
                  <span>Use ← / → to replay the game</span>
                  {isViewingHistory && (
                    <button
                      onClick={() => setViewIndex(null)}
                      className="ml-auto text-[#81b64c] hover:underline"
                    >
                      Back to live
                    </button>
                  )}
                </div>

                {/* Move history */}
                {moveChips.length > 0 && (
                  <div className="text-xs text-stone-400 pt-1 border-t border-[#363431] flex flex-wrap gap-1 font-mono">
                    <span className="text-stone-500">Continuations:</span>
                    {moveChips.map((m) => (
                      <button
                        key={m.index}
                        onClick={() => setViewIndex(m.index)}
                        className={`px-1.5 py-0.5 rounded transition ${
                          safeIndex === m.index
                            ? 'bg-[#81b64c] text-black font-bold'
                            : 'bg-[#1e1c19] text-white hover:bg-[#363431]'
                        }`}
                      >
                        {m.label}
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          ) : (
            <div className="w-full max-w-md aspect-square rounded-md border-2 border-dashed border-[#363431] bg-[#1e1c19]/60 flex flex-col items-center justify-center gap-3 text-center px-8">
              {isLoading ? (
                <>
                  <Loader2 className="w-10 h-10 animate-spin text-[#81b64c]" />
                  <p className="text-sm font-semibold text-white">
                    {isImporting ? 'Importing & retraining...' : 'Generating scenario...'}
                  </p>
                  <p className="text-xs text-stone-400 leading-relaxed">
                    {isImporting
                      ? 'Analyzing new games with Stockfish and rebuilding your private model. This can take a moment.'
                      : 'Selecting a fresh blunder from your own games.'}
                  </p>
                </>
              ) : (
                <>
                  <span className="text-5xl opacity-60">♟</span>
                  <p className="text-sm font-medium text-stone-300">No practice scenario yet</p>
                  <p className="text-xs text-stone-500 leading-relaxed">
                    Enter your Chess.com username above and click <strong>Import &amp; Practice</strong> to find a blunder from your own games. The base model is pretrained on Lichess games.
                  </p>
                </>
              )}
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
