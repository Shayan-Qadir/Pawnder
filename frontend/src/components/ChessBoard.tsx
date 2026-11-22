import React, { useState, useEffect, useRef } from 'react';
import { Chess, Square } from 'chess.js';

interface MoveAnimation {
  from: string;
  to: string;
  id: number;
}

interface ChessBoardProps {
  fen: string;
  playerColor: 'white' | 'black';
  isPlayerTurn: boolean;
  onMove: (moveUci: string) => void;
  disabled?: boolean;
  animateMove?: MoveAnimation | null;
  evalPercent?: number;
}

interface ActiveAnimation {
  from: Square;
  to: Square;
  pieceCode: string;
  moving: boolean;
}

const ANIMATION_MS = 160;

/** Infer a single piece move from two positions (used for history replay). */
function inferMove(beforeFen: string, afterFen: string): { from: Square; to: Square } | null {
  try {
    const before = new Chess(beforeFen);
    const after = new Chess(afterFen);
    const removed: { sq: Square; piece: NonNullable<ReturnType<Chess['get']>> }[] = [];
    const added: { sq: Square; piece: NonNullable<ReturnType<Chess['get']>> }[] = [];

    for (const file of 'abcdefgh') {
      for (const rank of '12345678') {
        const sq = `${file}${rank}` as Square;
        const a = before.get(sq);
        const b = after.get(sq);
        if (a && (!b || b.color !== a.color || b.type !== a.type)) {
          removed.push({ sq, piece: a });
        }
        if (b && (!a || a.color !== b.color || a.type !== b.type)) {
          added.push({ sq, piece: b });
        }
      }
    }

    // A capture (or undoing one) leaves one extra square difference: the
    // captured piece disappearing forward / reappearing backward. Match the
    // same piece type+color across two different squares to find the mover.
    const moves: { from: Square; to: Square }[] = [];
    for (const r of removed) {
      for (const a of added) {
        if (
          r.sq !== a.sq &&
          r.piece.color === a.piece.color &&
          r.piece.type === a.piece.type
        ) {
          moves.push({ from: r.sq, to: a.sq });
        }
      }
    }
    if (moves.length === 1 && removed.length <= 2 && added.length <= 2) {
      return moves[0];
    }
  } catch {
    // Ignore invalid FENs; the board will simply not animate this step.
  }
  return null;
}

export const ChessBoard: React.FC<ChessBoardProps> = ({
  fen,
  playerColor,
  isPlayerTurn,
  onMove,
  disabled = false,
  animateMove = null,
  evalPercent = 50,
}) => {
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [legalTargets, setLegalTargets] = useState<Square[]>([]);
  const [game, setGame] = useState<Chess>(new Chess(fen));
  const [anim, setAnim] = useState<ActiveAnimation | null>(null);
  const [dragOverSquare, setDragOverSquare] = useState<Square | null>(null);
  const animTimer = useRef<number | null>(null);
  const prevFenRef = useRef<string | null>(null);
  const inferredAnimId = useRef(0);

  useEffect(() => {
    // Explicit moves (live play) win; otherwise infer the step when the FEN
    // changes during ←/→ history replay or a move-chip jump.
    let requested = animateMove;
    if (!requested && prevFenRef.current && prevFenRef.current !== fen) {
      const inferred = inferMove(prevFenRef.current, fen);
      if (inferred) {
        inferredAnimId.current += 1;
        requested = { ...inferred, id: inferredAnimId.current };
      }
    }
    prevFenRef.current = fen;

    try {
      const next = new Chess(fen);
      setGame(next);
      setSelectedSquare(null);
      setLegalTargets([]);

      if (animTimer.current) window.clearTimeout(animTimer.current);

      if (requested) {
        const piece = next.get(requested.to as Square);
        if (piece) {
          setAnim({
            from: requested.from as Square,
            to: requested.to as Square,
            pieceCode: `${piece.color}${piece.type.toUpperCase()}`,
            moving: false,
          });
          // Paint the piece on the origin square first, then transition it.
          requestAnimationFrame(() =>
            requestAnimationFrame(() =>
              setAnim((current) => (current ? { ...current, moving: true } : current))
            )
          );
          animTimer.current = window.setTimeout(() => setAnim(null), ANIMATION_MS + 40);
        } else {
          setAnim(null);
        }
      } else {
        setAnim(null);
      }
    } catch (e) {
      console.error('Invalid FEN loaded:', fen);
    }
    return () => {
      if (animTimer.current) window.clearTimeout(animTimer.current);
    };
  }, [fen, animateMove?.id]);

  const expectedColor = playerColor === 'white' ? 'w' : 'b';
  const canInteract = !disabled && isPlayerTurn;

  const handleSquareClick = (square: Square) => {
    if (!canInteract) return;

    // If already selected, try to move
    if (selectedSquare) {
      if (selectedSquare === square) {
        setSelectedSquare(null);
        setLegalTargets([]);
        return;
      }

      // Check if clicked square is in legal targets
      if (legalTargets.includes(square)) {
        const moveUci = `${selectedSquare}${square}`;
        // Auto-promote to queen if pawn reaches 8th/1st rank
        const piece = game.get(selectedSquare);
        let finalUci = moveUci;
        if (
          piece?.type === 'p' &&
          ((square[1] === '8' && piece.color === 'w') ||
            (square[1] === '1' && piece.color === 'b'))
        ) {
          finalUci += 'q';
        }

        setSelectedSquare(null);
        setLegalTargets([]);
        onMove(finalUci);
        return;
      }
    }

    // Select piece
    const piece = game.get(square);
    if (piece && piece.color === expectedColor) {
      setSelectedSquare(square);
      const moves = game.moves({ square, verbose: true });
      setLegalTargets(moves.map((m) => m.to as Square));
    } else {
      setSelectedSquare(null);
      setLegalTargets([]);
    }
  };

  // Highlight the piece's legal destinations as soon as a drag begins, so the
  // drag behaves like the existing click-to-move flow.
  const handleDragStart = (e: React.DragEvent, square: Square) => {
    if (!canInteract) {
      e.preventDefault();
      return;
    }
    e.dataTransfer.setData('text/plain', square);
    e.dataTransfer.effectAllowed = 'move';
    setSelectedSquare(square);
    setLegalTargets(game.moves({ square, verbose: true }).map((m) => m.to as Square));
  };

  const clearDrag = () => {
    setDragOverSquare(null);
    setSelectedSquare(null);
    setLegalTargets([]);
  };

  // Dropping onto a legal destination builds the same UCI as a click.
  const handleDrop = (e: React.DragEvent, square: Square) => {
    if (!canInteract) return;
    e.preventDefault();
    const from = e.dataTransfer.getData('text/plain') as Square;
    const piece = from ? game.get(from) : null;
    const legal = piece && piece.color === expectedColor
      ? game.moves({ square: from, verbose: true }).map((m) => m.to as Square)
      : [];
    if (!from || !legal.includes(square)) {
      clearDrag();
      return;
    }
    let uci = `${from}${square}`;
    if (
      piece?.type === 'p' &&
      ((square[1] === '8' && piece.color === 'w') ||
        (square[1] === '1' && piece.color === 'b'))
    ) {
      uci += 'q';  // auto-promote to queen, same as click-to-move
    }
    clearDrag();
    onMove(uci);
  };

  const ranks = playerColor === 'white' ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const files =
    playerColor === 'white'
      ? ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
      : ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'];

  // chess.com-style material advantage, shown from each side's perspective.
  const pieceValues: Record<string, number> = { p: 1, n: 3, b: 3, r: 5, q: 9 };
  let whiteMaterial = 0;
  let blackMaterial = 0;
  for (const row of game.board()) {
    for (const piece of row) {
      if (!piece) continue;
      const value = pieceValues[piece.type] || 0;
      if (piece.color === 'w') whiteMaterial += value;
      else blackMaterial += value;
    }
  }
  const whiteAdvantage = whiteMaterial - blackMaterial;
  const playerAdvantage = playerColor === 'white' ? whiteAdvantage : -whiteAdvantage;
  const opponentAdvantage = -playerAdvantage;
  const formatAdvantage = (n: number) => (n > 0 ? `+${n}` : `${n}`);
  const advantageClass = (n: number) =>
    n > 0 ? 'text-[#81b64c]' : n < 0 ? 'text-stone-500' : 'text-stone-600';

  const animStyle = anim
    ? (() => {
        const target = anim.moving ? anim.to : anim.from;
        const col = Math.max(0, files.indexOf(target[0] as (typeof files)[number]));
        const row = Math.max(0, ranks.indexOf(Number(target[1])));
        return {
          left: `${col * 12.5}%`,
          top: `${row * 12.5}%`,
          width: '12.5%',
          height: '12.5%',
          transition: `left ${ANIMATION_MS}ms ease-in-out, top ${ANIMATION_MS}ms ease-in-out`,
        };
      })()
    : null;

  return (
    <div className="flex flex-col items-center select-none w-full ml-3 sm:ml-3.5">
      {/* Opponent material advantage */}
      <div className="w-full max-w-[320px] sm:max-w-[384px] flex justify-end px-1 pb-1">
        <span title="Opponent material advantage" className={`text-xs font-mono font-bold ${advantageClass(opponentAdvantage)}`}>
          {formatAdvantage(opponentAdvantage)}
        </span>
      </div>

      <div className="relative w-full max-w-[320px] sm:max-w-[384px]">
        {/* Vertical evaluation bar, chess.com style (white strength shown). */}
        <div
          className="absolute -left-3 sm:-left-3.5 top-0 bottom-0 w-2.5 sm:w-3 overflow-hidden rounded-sm border border-[#363431] bg-stone-900 shadow-md"
          title="Evaluation bar — white advantage"
        >
          <div
            className="absolute left-0 w-full bg-white transition-all duration-300"
            style={
              playerColor === 'white'
                ? { bottom: 0, height: `${evalPercent}%` }
                : { top: 0, height: `${evalPercent}%` }
            }
          />
        </div>
        <div className="grid grid-cols-8 w-full aspect-square border-2 border-[#363431] rounded-md shadow-xl overflow-hidden bg-[#779556]">
        {ranks.map((rank, rIdx) =>
          files.map((file, fIdx) => {
            const square = `${file}${rank}` as Square;
            const piece = game.get(square);
            const isLight = (rIdx + fIdx) % 2 === 0;
            const isSelected = selectedSquare === square;
            const isTarget = legalTargets.includes(square);
            const isAnimatingTarget = anim?.to === square;
            const isDropHover = isTarget && dragOverSquare === square;

            const squareBg = isDropHover || isSelected
              ? 'bg-[#f6f669]'
              : isLight
              ? 'bg-[#ebecd0]'
              : 'bg-[#779556]';

            const pieceCode = piece
              ? `${piece.color}${piece.type.toUpperCase()}`
              : null;

            return (
              <div
                key={square}
                onClick={() => handleSquareClick(square)}
                onDragOver={(e) => {
                  if (isTarget) {
                    e.preventDefault();
                    e.dataTransfer.dropEffect = 'move';
                  }
                }}
                onDragEnter={() => { if (isTarget) setDragOverSquare(square); }}
                onDragLeave={() => { if (dragOverSquare === square) setDragOverSquare(null); }}
                onDrop={(e) => handleDrop(e, square)}
                className={`w-full aspect-square flex items-center justify-center relative cursor-pointer transition-colors ${squareBg}`}
              >
                {/* Coordinate markers */}
                {fIdx === 0 && (
                  <span
                    className={`absolute top-0 left-1 text-[9px] font-semibold ${
                      isLight ? 'text-[#779556]' : 'text-[#ebecd0]'
                    }`}
                  >
                    {rank}
                  </span>
                )}
                {rIdx === 7 && (
                  <span
                    className={`absolute bottom-0 right-0.5 text-[9px] font-semibold ${
                      isLight ? 'text-[#779556]' : 'text-[#ebecd0]'
                    }`}
                  >
                    {file}
                  </span>
                )}

                {/* Legal move target dot / ring */}
                {isTarget && (
                  <span
                    className={`absolute rounded-full pointer-events-none ${
                      piece
                        ? 'w-3/4 h-3/4 border-[3px] border-black/30'
                        : 'w-1/4 h-1/4 bg-black/25'
                    }`}
                  />
                )}

                {/* Piece Render — hidden on the destination while it flies in */}
                {pieceCode && !isAnimatingTarget && (
                  <img
                    src={`/pieces/${pieceCode}.png`}
                    alt={`${piece?.color === 'w' ? 'White' : 'Black'} ${piece?.type}`}
                    className={`relative w-[92%] h-[92%] object-contain drop-shadow-sm ${
                      canInteract && piece?.color === expectedColor
                        ? 'cursor-grab active:cursor-grabbing'
                        : 'pointer-events-none'
                    }`}
                    draggable={canInteract && piece?.color === expectedColor}
                    onDragStart={(e) => handleDragStart(e, square)}
                    onDragEnd={clearDrag}
                  />
                )}
              </div>
            );
          })
        )}

          {/* Flying piece overlay for the current move */}
          {anim && animStyle && (
            <div className="absolute z-20 pointer-events-none flex items-center justify-center" style={animStyle}>
              <img
                src={`/pieces/${anim.pieceCode}.png`}
                alt="Moving piece"
                className="w-[92%] h-[92%] object-contain drop-shadow-md"
                draggable={false}
              />
            </div>
          )}
        </div>
      </div>

      {/* Player material advantage */}
      <div className="w-full max-w-[320px] sm:max-w-[384px] flex justify-start px-1 pt-1">
        <span title="Your material advantage" className={`text-xs font-mono font-bold ${advantageClass(playerAdvantage)}`}>
          {formatAdvantage(playerAdvantage)}
        </span>
      </div>
    </div>
  );
};
