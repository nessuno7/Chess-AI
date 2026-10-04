"""Tests for python-engine/monte_carlo_tree_search.py

Run from the repo root:
    C:/Users/nicol/miniconda3/envs/chessai/python.exe tests/test_mcts.py

Everything runs on UniformEvaluator (uniform priors, value 0), so the only thing that can make
the search prefer a move is a terminal position it actually reaches. That makes the backup sign
directly testable: on a mate in one the mating move has to win the visits with Q near +1, for
white AND for black.
"""

import os
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "python-engine"))

from board import ChessGame
from monte_carlo_tree_search import (POLICY_SIZE, MCTSConfig, MonteCarloTS, Node,
                                     UniformEvaluator, search_many)

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))


def quiet_config(sims=200, **kw):
    """deterministic search: no root noise, argmax move selection"""
    return MCTSConfig(num_simulations=sims, add_root_noise=False, temperature_moves=0, **kw)


def snapshot(g):
    return (g.board.fen(), list(g.board.move_stack), dict(g.positions_map), len(g.history))


MATE_WHITE = "r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4"   # Qxf7#
MATE_BLACK = "rnbqkbnr/pppp1ppp/8/4p3/6P1/5P2/PPPPP2P/RNBQKBNR b KQkq - 0 2"           # Qh4#

# ------------------------------------------------------------------------ Node
print("Node")
n = Node()
check("fresh node is not expanded", not n.expanded)
check("fresh node is not terminal", not n.is_terminal)
n.terminal_value = 0.0
check("terminal_value 0.0 (draw) still counts as terminal", n.is_terminal)

n = Node()
moves = list(chess.Board().legal_moves)[:3]
n.expand(moves, np.array([0.2, 0.5, 0.3], dtype=np.float32))
check("expand allocates N, W, children", n.N.tolist() == [0, 0, 0] and n.W.tolist() == [0, 0, 0]
      and n.children == [None] * 3)
check("first visit follows the highest prior", n.best_child_index(1.5) == 1)
n.N[:] = [2, 0, 0]
n.W[:] = [1.0, 0, 0]
check("q is W/N with fpu on unvisited edges", np.allclose(n.q(fpu=-0.3), [0.5, -0.3, -0.3]))
n2 = Node()
n2.expand(moves, np.full(3, 1 / 3, dtype=np.float32))
check("ties are broken by the lowest index", n2.best_child_index(1.5) == 0)

# --------------------------------------------------------- terminal detection
print("terminal values")


def tv(fen, ucis=()):
    g = ChessGame(fen)
    for u in ucis:
        g.play_move(chess.Move.from_uci(u))
    return MonteCarloTS(g, UniformEvaluator())._terminal_value()


check("start position is not terminal", tv(chess.STARTING_FEN) is None)
check("checkmated side to move scores -1", tv(MATE_WHITE, ["h5f7"]) == -1.0)
check("stalemate scores 0", tv("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1") == 0.0)
check("insufficient material scores 0", tv("8/8/4k3/8/8/4K3/8/8 w - - 0 1") == 0.0)
check("fifty move rule scores 0", tv("8/8/4k3/8/8/4K3/8/R7 w - - 100 80") == 0.0)
check("threefold repetition scores 0",
      tv(chess.STARTING_FEN, ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]) == 0.0)
check("twofold repetition is not terminal",
      tv(chess.STARTING_FEN, ["g1f3", "g8f6", "f3g1", "f6g8"]) is None)

# ------------------------------------------------------- board is restored
print("search leaves the board untouched")
for fen in (chess.STARTING_FEN, MATE_WHITE, MATE_BLACK):
    g = ChessGame(fen, stored_timesteps=4)
    g.play_move(g.legal_moves_list()[0])
    before = snapshot(g)
    enc = g.encode()
    t = MonteCarloTS(g, UniformEvaluator(), MCTSConfig(num_simulations=150), seed=0)
    root = t.search()
    check(f"fen/stack/positions_map/history restored ({fen.split()[0][:12]}..)", snapshot(g) == before)
    check("encode() unchanged after search", np.array_equal(g.encode(), enc))
    check("root visits == simulations", int(root.N.sum()) == 150, f"got {root.N.sum()}")

# ----------------------------------------------------------- mate in one
print("mate in one (backup sign)")
for name, fen, mate in (("white", MATE_WHITE, "h5f7"), ("black", MATE_BLACK, "d8h4")):
    g = ChessGame(fen)
    t = MonteCarloTS(g, UniformEvaluator(), quiet_config(400))
    root = t.search()
    best = t.select_move(move_number=99)
    k = root.moves.index(chess.Move.from_uci(mate))
    check(f"{name}: mating move is most visited", best.uci() == mate, f"chose {best.uci()}")
    check(f"{name}: mating move has Q == +1", abs(root.q()[k] - 1.0) < 1e-6, f"Q={root.q()[k]:.3f}")

# --------------------------------------------------------- policy target
print("policy target")
g = ChessGame()
t = MonteCarloTS(g, UniformEvaluator(), MCTSConfig(num_simulations=100), seed=1)
t.search()
pi = t.policy_target()
mask = g.create_legal_moves_mask()
check("pi has POLICY_SIZE entries", pi.shape == (POLICY_SIZE,))
check("pi sums to 1", abs(pi.sum() - 1) < 1e-5)
check("pi is zero on every illegal index", np.all(pi[mask == 0] == 0))
idx, p = t.policy_target_sparse()
dense = np.zeros(POLICY_SIZE, dtype=np.float32)
dense[idx] = p
check("sparse target matches dense", np.allclose(dense, pi))
pi0 = t.policy_target(temperature=0)
check("temperature 0 is one-hot", pi0.max() == 1.0 and pi0.sum() == 1.0)

print("root noise")
t = MonteCarloTS(ChessGame(), UniformEvaluator(), MCTSConfig(num_simulations=10), seed=3)
t.search()
p1 = t.root.priors.copy()
check("noise makes root priors non-uniform", p1.std() > 0)
t.search()
check("noise is applied once per root, not once per search() call", np.array_equal(p1, t.root.priors))

# ---------------------------------------------------------------- advance
print("advance / reset")
g = ChessGame()
t = MonteCarloTS(g, UniformEvaluator(), quiet_config(200))
t.search()
k = int(np.argmax(t.root.N))
move, child = t.root.moves[k], t.root.children[k]
t.advance(move)
check("advance plays the move", g.board.move_stack[-1] == move)
check("advance keeps the child subtree as the root", t.root is child and child.expanded)
t.search(50)
check("search after advance adds onto the reused visits",
      int(t.root.N.sum()) == int(child.N.sum()) and t.root.N.sum() >= 50)
unvisited = [m for m, c in zip(t.root.moves, t.root.children) if c is None]
if unvisited:
    t.advance(unvisited[0])
    check("advance along an unvisited edge gives a fresh root", not t.root.expanded)
t.reset()
check("reset drops the tree", not t.root.expanded)

# ------------------------------------------------------------ search_many
print("search_many")
fens = [chess.STARTING_FEN, MATE_WHITE, MATE_BLACK]
solo = []
for fen in fens:
    t = MonteCarloTS(ChessGame(fen), UniformEvaluator(), quiet_config(120))
    t.search()
    solo.append(t.root.N.copy())

games = [ChessGame(fen) for fen in fens]
before = [snapshot(g) for g in games]
trees = [MonteCarloTS(g, UniformEvaluator(), quiet_config(120)) for g in games]
search_many(trees, UniformEvaluator())
check("every board restored", [snapshot(g) for g in games] == before)
check("every tree got its simulations", all(int(t.root.N.sum()) == 120 for t in trees))
check("batched search == one search per tree (deterministic evaluator)",
      all(np.array_equal(t.root.N, s) for t, s in zip(trees, solo)))

g = ChessGame(MATE_WHITE)
g.play_move(chess.Move.from_uci("h5f7"))
t = MonteCarloTS(g, UniformEvaluator(), quiet_config(10))
search_many([t], UniformEvaluator())
check("a finished game is resolved as terminal, not expanded", t.root.is_terminal and not t.root.expanded)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
