// Generates the chess.com-style "cburnett" PNG piece set used by the board.
// Source: Lichess asset CDN (cburnett SVG set by Colin Burnett, public domain).
// Run with: npm run pieces
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import sharp from 'sharp';

const OUT_DIR = path.resolve('public/pieces');
const SIZE = 120;
const BASE = 'https://lichess1.org/assets/piece/cburnett';

const pieces = [
  'wK', 'wQ', 'wR', 'wB', 'wN', 'wP',
  'bK', 'bQ', 'bR', 'bB', 'bN', 'bP',
];

await mkdir(OUT_DIR, { recursive: true });

for (const name of pieces) {
  const response = await fetch(`${BASE}/${name}.svg`);
  if (!response.ok) throw new Error(`Failed to fetch ${name}.svg: ${response.status}`);
  const svg = Buffer.from(await response.arrayBuffer());
  const png = await sharp(svg, { density: 300 }).resize(SIZE, SIZE).png().toBuffer();
  await writeFile(path.join(OUT_DIR, `${name}.png`), png);
  console.log(`${name}.png`);
}
