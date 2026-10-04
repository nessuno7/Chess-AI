"""
External yardstick: the net against Stockfish at a fixed low skill level.

Self-play results and the arena only ever compare the net with itself, so they can not tell
"both sides got stronger" from "nothing changed". A fixed opponent can: run this every few
iterations (self_play.py --stockfish) and watch the score in <out-dir>/benchmark.csv.

The net searches the way it does in the arena (no root noise, the first few plies sampled from
the visit counts so the games differ), Stockfish plays at --skill with a fixed time per move,
optionally capped at --depth for an even weaker opponent. Colours alternate.

Stockfish Skill Level 0 still beats a young net every game; while the score is stuck at 0% add
--depth 1 (or 2) so there is something to measure.

    python benchmark.py ../checkpoints/selfplay_best.pt C:/tools/stockfish.exe --games 10 --skill 0
"""

import argparse
import random

import chess
import chess.engine
import torch

from arena import arena_config, score
from board import ChessGame
from monte_carlo_tree_search import MonteCarloTS
from network import NetEvaluator, load_checkpoint


def play_vs_engine(evaluator, engine, limit, num_games=10, sims=200, stored_timesteps=1,
                   max_plies=400, verbose=False):
    """
    (wins, losses, draws) for the net against `engine`, anything with python-chess's
    engine.play(board, limit) -> PlayResult. A game still running at max_plies is a draw.
    """
    wins = losses = draws = 0
    for i in range(num_games):
        net_is_white = i % 2 == 0
        tree = MonteCarloTS(ChessGame(stored_timesteps=stored_timesteps), evaluator,
                            arena_config(sims), seed=random.randrange(2**32))
        board = tree.game.board
        ply, value = 0, None
        while value is None and ply < max_plies:
            if (board.turn == chess.WHITE) == net_is_white:
                tree.search()
                move = tree.select_move(ply)
            else:
                move = engine.play(board.copy(), limit).move
            tree.advance(move)
            ply += 1
            value = tree._terminal_value()   # for the side to move

        if not value:   # None (move cap) or 0.0
            draws += 1
        elif (board.turn == chess.WHITE) == net_is_white:   # the net is to move and got mated
            losses += 1
        else:
            wins += 1
        if verbose:
            print(f"  game {i + 1}/{num_games}: net {wins}W {losses}L {draws}D in {ply} plies")
    return wins, losses, draws


def play_vs_stockfish(evaluator, stockfish_path, num_games=10, sims=200, skill=0, depth=None,
                      movetime=0.05, stored_timesteps=1, max_plies=400, verbose=False):
    """play_vs_engine against a Stockfish process at Skill Level `skill`, single threaded."""
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        engine.configure({"Skill Level": skill, "Threads": 1})
        limit = chess.engine.Limit(time=movetime, depth=depth)
        return play_vs_engine(evaluator, engine, limit, num_games, sims, stored_timesteps,
                              max_plies, verbose)
    finally:
        engine.quit()


def append_benchmark(path, iteration, generation, skill, depth, wins, losses, draws):
    """One row of <out-dir>/benchmark.csv, writing the header when the file is new."""
    new_file = not path.exists()
    with open(path, "a") as f:
        if new_file:
            f.write("iteration,generation,skill,depth,wins,losses,draws\n")
        f.write(f"{iteration},{generation},{skill},{depth or ''},{wins},{losses},{draws}\n")


def main():
    ap = argparse.ArgumentParser(description="play a checkpoint against Stockfish")
    ap.add_argument("checkpoint")
    ap.add_argument("stockfish", help="path to the Stockfish binary")
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--sims", type=int, default=200)
    ap.add_argument("--skill", type=int, default=0, help="Stockfish Skill Level (0-20)")
    ap.add_argument("--depth", type=int, default=None, help="cap Stockfish's search depth")
    ap.add_argument("--movetime", type=float, default=0.05, help="seconds per Stockfish move")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    net, _ = load_checkpoint(args.checkpoint, args.device)
    w, l, d = play_vs_stockfish(NetEvaluator(net, args.device), args.stockfish, args.games,
                                args.sims, args.skill, args.depth, args.movetime,
                                net.stored_timesteps, args.max_plies, verbose=True)
    print(f"net vs stockfish skill {args.skill}: {w}W {l}L {d}D, score {score(w, l, d):.1%}")


if __name__ == "__main__":
    main()
