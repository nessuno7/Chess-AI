"""
Supervised training on the Lichess puzzle database, and a solve-rate benchmark.

Data: https://database.lichess.org/#puzzles  ->  lichess_db_puzzle.csv.zst
    Either decompress it to .csv, or `pip install zstandard` and pass the .zst directly.
    Columns: PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags

Lichess puzzle format: FEN is the position BEFORE the opponent's last move, and Moves (UCI) starts
with that move. The solver plays moves 1, 3, 5, ... and the opponent's forced replies are 2, 4, ...

Every move of the solution becomes a training sample:
    solver to move      policy target = the solution move, z = +1
    opponent to move    policy target = the reply in the solution, z = -1
Puzzles tagged "equality" (find the move that saves the game) get z = 0 for both sides.

The csv defaults to <repo>/data/lichess_db_puzzle.csv. Weights are saved in <repo>/checkpoints:
    puzzles_latest.pt   after every epoch (to continue from: --init)
    puzzles_best.pt     the epoch with the best held-out solve rate (to start self play from)

Run from python-engine/:
    python puzzles.py --limit 200000 --epochs 5
    python puzzles.py --init ../checkpoints/puzzles_latest.pt --skip 200000 --limit 200000 --epochs 5
    python puzzles.py --eval-only --init ../checkpoints/puzzles_best.pt --eval-sims 100
"""

import argparse
import csv
import io
import random
from dataclasses import dataclass
from pathlib import Path

import chess
import numpy as np
import torch

from board import ChessGame
from monte_carlo_tree_search import MCTSConfig, MonteCarloTS
from network import CHECKPOINT_DIR, REPO_ROOT, ChessNet, NetEvaluator, load_checkpoint, save_checkpoint
from training import make_optimizer, make_sample, train_epoch


@dataclass
class Puzzle:
    puzzle_id: str
    fen: str
    moves: list   # list[chess.Move], moves[0] is the opponent's setup move
    rating: int
    themes: list


def _open_text(path):
    if path.endswith(".zst"):
        try:
            import zstandard
        except ImportError as e:
            raise SystemExit("reading .zst needs `pip install zstandard` (or decompress the file first)") from e
        return io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(open(path, "rb")), encoding="utf-8")
    return open(path, newline="", encoding="utf-8")


def load_puzzles(path, limit=None, min_rating=0, max_rating=10_000, themes=None, skip=0):
    """
    Read up to `limit` puzzles with min_rating <= Rating <= max_rating. `themes`, when given, keeps
    only puzzles carrying at least one of those Lichess theme tags (e.g. {"mateIn1", "fork"}).
    The file is streamed, so `limit` keeps memory bounded even on the full ~5M puzzle file.
    The first `skip` puzzles that pass the filters are passed over, so successive runs can walk
    through the file a chunk at a time.
    """
    out = []
    skipped = 0
    with _open_text(path) as f:
        for row in csv.DictReader(f):
            rating = int(row["Rating"])
            if not min_rating <= rating <= max_rating:
                continue
            tags = row["Themes"].split()
            if themes and not themes.intersection(tags):
                continue
            if skipped < skip:
                skipped += 1
                continue
            out.append(Puzzle(row["PuzzleId"], row["FEN"],
                              [chess.Move.from_uci(u) for u in row["Moves"].split()], rating, tags))
            if limit and len(out) >= limit:
                break
    return out


def puzzle_samples(puzzle, stored_timesteps=1, include_opponent=True):
    """Training samples for one puzzle, see the module docstring for the targets."""
    game = ChessGame(puzzle.fen, stored_timesteps=stored_timesteps)
    game.play_move(puzzle.moves[0])
    z_solver = 0.0 if "equality" in puzzle.themes else 1.0

    samples = []
    for i, move in enumerate(puzzle.moves[1:]):
        solver_to_move = i % 2 == 0
        if solver_to_move or include_opponent:
            z = z_solver if solver_to_move else -z_solver
            samples.append(make_sample(game, [game.hash_move(move)], [1.0], z))
        game.play_move(move)
    return samples


# ---- solving -------------------------------------------------------------------------

def _policy_move(game, evaluator):
    """Most likely legal move under the raw policy, no search."""
    logits, _ = evaluator.evaluate(game)
    moves = game.legal_moves_list()
    return max(moves, key=lambda m: logits[game.hash_move(m)])


def _search_move(game, evaluator, sims):
    tree = MonteCarloTS(game, evaluator, MCTSConfig(num_simulations=sims, add_root_noise=False,
                                                    temperature_moves=0))
    tree.search()
    return tree.select_move(move_number=0)   # argmax visits since temperature_moves = 0


def solve(puzzle, evaluator, sims=0, stored_timesteps=1):
    """
    True when the engine finds the whole solution. Like on Lichess, a different move that
    delivers checkmate also counts as solving the puzzle.
    """
    game = ChessGame(puzzle.fen, stored_timesteps=stored_timesteps)
    game.play_move(puzzle.moves[0])
    for i, expected in enumerate(puzzle.moves[1:]):
        if i % 2 == 0:   # solver's turn
            chosen = _search_move(game, evaluator, sims) if sims else _policy_move(game, evaluator)
            if chosen != expected:
                game.play_move(chosen)
                return game.board.is_checkmate()
        game.play_move(expected)
    return True


def solve_rate(puzzles, evaluator, sims=0, stored_timesteps=1):
    if not puzzles:
        return 0.0
    return sum(solve(p, evaluator, sims, stored_timesteps) for p in puzzles) / len(puzzles)


# ---- main ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="train / evaluate the net on Lichess puzzles")
    ap.add_argument("--csv", default=str(REPO_ROOT / "data" / "lichess_db_puzzle.csv"),
                    help="lichess_db_puzzle.csv (or .csv.zst)")
    ap.add_argument("--limit", type=int, default=100_000, help="puzzles to load")
    ap.add_argument("--skip", type=int, default=0, help="matching puzzles to pass over first")
    ap.add_argument("--min-rating", type=int, default=0)
    ap.add_argument("--max-rating", type=int, default=10_000)
    ap.add_argument("--themes", nargs="*", help="only puzzles with one of these theme tags")
    ap.add_argument("--val-fraction", type=float, default=0.02, help="held out for the solve rate")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--no-opponent", action="store_true", help="train only on the solver's moves")
    ap.add_argument("--eval-sims", type=int, default=0, help="MCTS sims when measuring solve rate (0 = raw policy)")
    ap.add_argument("--eval-size", type=int, default=500, help="max held-out puzzles checked per epoch")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--init", help="checkpoint to start from")
    ap.add_argument("--out-dir", default=str(CHECKPOINT_DIR))
    ap.add_argument("--channels", type=int, default=64, help="only used without --init")
    ap.add_argument("--blocks", type=int, default=6, help="only used without --init")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.init:
        net, _ = load_checkpoint(args.init, args.device)
    else:
        net = ChessNet(channels=args.channels, blocks=args.blocks).to(args.device)
    T = net.stored_timesteps
    evaluator = NetEvaluator(net, args.device)

    puzzles = load_puzzles(args.csv, args.limit, args.min_rating, args.max_rating,
                           set(args.themes) if args.themes else None, args.skip)
    random.shuffle(puzzles)
    n_val = max(1, int(len(puzzles) * args.val_fraction))
    val, train = puzzles[:n_val], puzzles[n_val:]
    val_eval = val[:args.eval_size]
    print(f"{len(puzzles)} puzzles: {len(train)} train, {len(val)} held out")

    if args.eval_only:
        print(f"solve rate: {solve_rate(val_eval, evaluator, args.eval_sims, T):.1%}")
        return

    samples = [s for p in train for s in puzzle_samples(p, T, not args.no_opponent)]
    print(f"{len(samples)} training positions")
    optimizer = make_optimizer(net, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    out = Path(args.out_dir)
    best_rate = -1.0
    for epoch in range(1, args.epochs + 1):
        loss, pl, vl = train_epoch(net, optimizer, samples, args.batch_size, args.device)
        scheduler.step()
        rate = solve_rate(val_eval, evaluator, args.eval_sims, T)
        print(f"epoch {epoch}: loss {loss:.3f} (policy {pl:.3f}, value {vl:.3f}), "
              f"held-out solve rate {rate:.1%}")
        save_checkpoint(out / "puzzles_latest.pt", net, optimizer, epoch=epoch, solve_rate=rate)
        if rate > best_rate:
            best_rate = rate
            save_checkpoint(out / "puzzles_best.pt", net, epoch=epoch, solve_rate=rate)
            print(f"  new best, saved {out / 'puzzles_best.pt'}")


if __name__ == "__main__":
    main()
