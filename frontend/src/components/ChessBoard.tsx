import React, { useState, useEffect } from 'react';
import { Chess, Square } from 'chess.js';

interface ChessBoardProps {
  fen: string;
  playerColor: 'white' | 'black';
  isPlayerTurn: boolean;
  onMove: (moveUci: string) => void;
  disabled?: boolean;
}

const PIECE_UNICODE: Record<string, string> = {
  p: '♟',
  n: '♞',
  b: '♝',
  r: '♜',
  q: '♛',
  k: '♚',
  P: '♙',
  N: '♘',
  B: '♗',
  R: '♖',
  Q: '♕',
  K: '♔',
};

export const ChessBoard: React.FC<ChessBoardProps> = ({
  fen,
  playerColor,
  isPlayerTurn,
  onMove,
  disabled = false,
}) => {
  const [selectedSquare, setSelectedSquare] = useState<Square | null>(null);
  const [legalTargets, setLegalTargets] = useState<Square[]>([]);
  const [game, setGame] = useState<Chess>(new Chess(fen));

  useEffect(() => {
    try {
      setGame(new Chess(fen));
      setSelectedSquare(null);
      setLegalTargets([]);
    } catch (e) {
      console.error('Invalid FEN loaded:', fen);
    }
  }, [fen]);

  const handleSquareClick = (square: Square) => {
    if (disabled || !isPlayerTurn) return;

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
    const expectedColor = playerColor === 'white' ? 'w' : 'b';
    if (piece && piece.color === expectedColor) {
      setSelectedSquare(square);
      const moves = game.moves({ square, verbose: true });
      setLegalTargets(moves.map((m) => m.to as Square));
    } else {
      setSelectedSquare(null);
      setLegalTargets([]);
    }
  };

  const ranks = playerColor === 'white' ? [8, 7, 6, 5, 4, 3, 2, 1] : [1, 2, 3, 4, 5, 6, 7, 8];
  const files =
    playerColor === 'white'
      ? ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
      : ['h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'];

  return (
    <div className="flex flex-col items-center select-none">
      <div className="grid grid-cols-8 border-4 border-[#363431] rounded-lg shadow-2xl overflow-hidden bg-[#779556]">
        {ranks.map((rank, rIdx) =>
          files.map((file, fIdx) => {
            const square = `${file}${rank}` as Square;
            const piece = game.get(square);
            const isLight = (rIdx + fIdx) % 2 === 0;
            const isSelected = selectedSquare === square;
            const isTarget = legalTargets.includes(square);

            const squareBg = isSelected
              ? 'bg-[#bbcb2b]'
              : isLight
              ? 'bg-[#ebecd0]'
              : 'bg-[#779556]';

            return (
              <div
                key={square}
                onClick={() => handleSquareClick(square)}
                className={`w-14 h-14 sm:w-16 sm:h-16 md:w-20 md:h-20 flex items-center justify-center relative cursor-pointer transition-colors ${squareBg}`}
              >
                {/* Coordinate markers */}
                {fIdx === 0 && (
                  <span
                    className={`absolute top-0.5 left-1 text-[10px] sm:text-xs font-semibold ${
                      isLight ? 'text-[#779556]' : 'text-[#ebecd0]'
                    }`}
                  >
                    {rank}
                  </span>
                )}
                {rIdx === 7 && (
                  <span
                    className={`absolute bottom-0.5 right-1 text-[10px] sm:text-xs font-semibold ${
                      isLight ? 'text-[#779556]' : 'text-[#ebecd0]'
                    }`}
                  >
                    {file}
                  </span>
                )}

                {/* Legal move target dot / ring */}
                {isTarget && (
                  <div
                    className={`absolute rounded-full pointer-events-none ${
                      piece
                        ? 'w-10 h-10 sm:w-12 sm:h-12 border-4 border-black/30'
                        : 'w-4 h-4 sm:w-5 sm:h-5 bg-black/25'
                    }`}
                  />
                )}

                {/* Piece Render */}
                {piece && (
                  <span
                    className={`text-4xl sm:text-5xl md:text-6xl font-normal leading-none transform active:scale-95 transition-transform ${
                      piece.color === 'w'
                        ? 'text-white drop-shadow-[0_2px_3px_rgba(0,0,0,0.8)]'
                        : 'text-stone-900 drop-shadow-[0_1px_1px_rgba(255,255,255,0.4)]'
                    }`}
                  >
                    {PIECE_UNICODE[piece.color === 'w' ? piece.type.toUpperCase() : piece.type]}
                  </span>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
