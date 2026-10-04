"""
Matches between two networks: the gate self-play uses to decide whether a newly trained net
replaces the current best one, and a CLI to compare any two checkpoints.

Each game has two players, A and B, and each player owns its own ChessGame copy and its own
MonteCarloTS (two trees can not walk one board). After every move both trees advance, so both
keep their subtree. Like self play, many games run side by side: in each round the trees whose
player is to move are searched together, one batched search_many per network.

Colours alternate (A is white in the even games) and the first `temperature_moves` plies are
sampled from the visit counts, otherwise two deterministic nets would play the same game twice.

    python arena.py ../checkpoints/selfplay_latest.pt ../checkpoints/puzzles.pt --games 20 --sims 50
"""

import argparse
import random

import chess
import torch

from board import ChessGame
from monte_carlo_tree_search import MCTSConfig, MonteCarloTS, search_many
from network import NetEvaluator, load_checkpoint


def arena_config(sims, temperature_moves=8):
    return MCTSConfig(num_simulations=sims, add_root_noise=False, temperature_moves=temperature_moves)


class _ArenaGame:
    def __init__(self, a_is_white, eval_a, eval_b, config, stored_timesteps):
        self.a_is_white = a_is_white
        seed = random.randrange(2**32)
        self.tree_a = MonteCarloTS(ChessGame(stored_timesteps=stored_timesteps), eval_a, config, seed=seed)
        self.tree_b = MonteCarloTS(ChessGame(stored_timesteps=stored_timesteps), eval_b, config, seed=seed + 1)
        self.ply = 0

    def a_to_move(self):
        return (self.tree_a.game.board.turn == chess.WHITE) == self.a_is_white

    def mover(self):
        return self.tree_a if self.a_to_move() else self.tree_b


def play_match(eval_a, eval_b, num_games=20, parallel=10, config=None, stored_timesteps=1,
               max_plies=400, verbose=False):
    """
    (wins, losses, draws) for player A against player B. Both nets must use the same
    stored_timesteps, since they read the same ChessGame encoding.
    """
    config = config if config is not None else arena_config(50)
    wins = losses = draws = 0
    started = 0

    def new_game():
        nonlocal started
        g = _ArenaGame(started % 2 == 0, eval_a, eval_b, config, stored_timesteps)
        started += 1
        return g

    active = [new_game() for _ in range(min(parallel, num_games))]
    while active:
        trees_a = [g.tree_a for g in active if g.a_to_move()]
        trees_b = [g.tree_b for g in active if not g.a_to_move()]
        search_many(trees_a, eval_a)
        search_many(trees_b, eval_b)

        still_running = []
        for g in active:
            mover = g.mover()
            move = mover.select_move(g.ply)
            g.tree_a.advance(move)
            g.tree_b.advance(move)
            g.ply += 1

            value = g.tree_a._terminal_value()   # for the side to move, i.e. the one that did NOT move
            if value is None and g.ply >= max_plies:
                value = 0.0
            if value is None:
                still_running.append(g)
                continue

            if value == 0:
                draws += 1
            elif g.a_to_move():   # A is to move and has value -1: A got mated
                losses += 1
            else:
                wins += 1
            if verbose:
                print(f"  arena game {wins + losses + draws}/{num_games}: A {wins}W {losses}L {draws}D")
            if started < num_games:
                still_running.append(new_game())
        active = still_running

    return wins, losses, draws


def score(wins, losses, draws):
    """fraction of the points A took: a win is 1, a draw 0.5"""
    games = wins + losses + draws
    return (wins + 0.5 * draws) / games if games else 0.0


def main():
    ap = argparse.ArgumentParser(description="play two checkpoints against each other")
    ap.add_argument("a", help="checkpoint of player A")
    ap.add_argument("b", help="checkpoint of player B")
    ap.add_argument("--games", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=10)
    ap.add_argument("--sims", type=int, default=50)
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    net_a, _ = load_checkpoint(args.a, args.device)
    net_b, _ = load_checkpoint(args.b, args.device)
    if net_a.stored_timesteps != net_b.stored_timesteps:
        raise SystemExit("the two nets use a different stored_timesteps, they can not share a board encoding")
    w, l, d = play_match(NetEvaluator(net_a, args.device), NetEvaluator(net_b, args.device),
                         args.games, args.parallel, arena_config(args.sims),
                         net_a.stored_timesteps, args.max_plies, verbose=True)
    print(f"A vs B: {w}W {l}L {d}D, score {score(w, l, d):.1%}")


if __name__ == "__main__":
    main()
