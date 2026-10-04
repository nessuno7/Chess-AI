"""Tests for python-engine/network.py, training.py, self_play.py, arena.py and puzzles.py

Run from the repo root:
    C:/Users/nicol/miniconda3/envs/chessai/python.exe tests/test_training_pipeline.py

Small and fast on purpose (tiny net, a handful of simulations): these check that the pieces fit
together and that the targets are right, not that anything plays well. The last section runs
the real command line scripts (puzzles.py, self_play.py incl. --resume and --workers, arena.py)
end to end in a temporary checkpoint directory.
"""

import os
import random
import subprocess
import sys
import tempfile

import chess
import numpy as np
import torch

ENGINE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "python-engine")
sys.path.insert(0, ENGINE)

import chess.engine

from arena import arena_config, play_match, score
from benchmark import append_benchmark, play_vs_engine, stockfish_from_env
from board import ChessGame
from monte_carlo_tree_search import POLICY_SIZE, Evaluator, MCTSConfig, UniformEvaluator
from network import ChessNet, NetEvaluator, load_checkpoint, save_checkpoint
from puzzles import load_puzzles, puzzle_samples, solve, solve_rate
from self_play import play_games
from training import collate, load_samples, make_optimizer, save_samples, train_epoch

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))


random.seed(0)
torch.manual_seed(0)

# --------------------------------------------------------------------- network
print("network")
for T in (1, 3):
    net = ChessNet(stored_timesteps=T, channels=16, blocks=1)
    games = [ChessGame(stored_timesteps=T), ChessGame("8/8/4k3/8/8/4K3/8/R7 b - - 0 1", stored_timesteps=T)]
    out = NetEvaluator(net).evaluate_batch(games)
    check(f"T={T}: one (logits, value) per game", len(out) == 2)
    check(f"T={T}: logits are POLICY_SIZE wide", all(o[0].shape == (POLICY_SIZE,) for o in out))
    check(f"T={T}: values in [-1, 1]", all(-1 <= o[1] <= 1 for o in out))

# the policy layout: logit (rank*8+file)*73 + type must come from policy plane `type` at (rank, file)
net = ChessNet(channels=8, blocks=1)
captured = {}
net.policy_head.register_forward_hook(lambda m, i, o: captured.setdefault("planes", o))
logits, _ = net(torch.from_numpy(ChessGame().encode()[None]))
planes = captured["planes"][0]
sq, typ = chess.E2, 7
check("policy index = square * 73 + move type",
      torch.isclose(logits[0, sq * 73 + typ], planes[typ, chess.square_rank(sq), chess.square_file(sq)]).item())

with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "net.pt")
    net = ChessNet(stored_timesteps=2, channels=8, blocks=2)
    save_checkpoint(path, net, make_optimizer(net), epoch=3)
    net2, ckpt = load_checkpoint(path)
    x = torch.from_numpy(ChessGame(stored_timesteps=2).encode()[None])
    net.eval(); net2.eval()
    check("checkpoint round trip rebuilds the same net",
          net2.hparams() == net.hparams() and torch.allclose(net(x)[0], net2(x)[0]) and ckpt["epoch"] == 3)

# ------------------------------------------------------------------- self play
print("self play")
cfg = MCTSConfig(num_simulations=8, temperature_moves=10)
samples, results, _ = play_games(UniformEvaluator(), num_games=5, parallel=3, config=cfg,
                                 max_plies=30, verbose=False)
check("every requested game is played", len(results) == 5, f"{results}")
check("results are well formed", set(results) <= {"1-0", "0-1", "1/2-1/2"})
check("one sample per ply, at most max_plies per game", 0 < len(samples) <= 5 * 30)
check("pi sums to 1 on every sample", all(abs(s[2].sum() - 1) < 1e-5 for s in samples))
check("z is in {-1, 0, 1}", all(s[3] in (-1.0, 0.0, 1.0) for s in samples))
check("states have the encode() shape", all(s[0].shape == (21, 8, 8) for s in samples))


class MateFinder(Evaluator):
    """uniform evaluator, but a decisive game is needed to test the z signs: play fool's mate"""
    LINE = ["f2f3", "e7e5", "g2g4", "d8h4"]

    def evaluate(self, game):
        logits = np.full(POLICY_SIZE, -20.0, dtype=np.float32)
        ply = len(game.board.move_stack)
        if ply < len(self.LINE):
            logits[game.hash_move(chess.Move.from_uci(self.LINE[ply]))] = 20.0
        return logits, 0.0


samples, results, _ = play_games(MateFinder(), 1, 1, MCTSConfig(num_simulations=4, add_root_noise=False,
                                                                temperature_moves=0), verbose=False)
check("fool's mate is played and scored 0-1", results == ["0-1"], f"{results}")
check("z = -1 for white's positions, +1 for black's",
      [s[3] for s in samples] == [-1.0, 1.0, -1.0, 1.0], f"{[s[3] for s in samples]}")


class WhiteIsLost(Evaluator):
    """uniform policy, value -1 whenever white is to move (+1 for black): white wants to resign"""

    def evaluate(self, game):
        return np.zeros(POLICY_SIZE, dtype=np.float32), -1.0 if game.board.turn == chess.WHITE else 1.0


print("resignation")
rcfg = MCTSConfig(num_simulations=8, add_root_noise=False)
samples, results, stats = play_games(WhiteIsLost(), 3, 3, rcfg, verbose=False, no_resign_frac=0.0)
check("white resigns on its first move, scored 0-1", results == ["0-1"] * 3, f"{results}")
check("a resigned game keeps its position, z = -1 for white", [s[3] for s in samples] == [-1.0] * 3)
check("no resign checks when every game may resign", stats == {"resign_checked": 0, "resign_false_pos": 0})

samples, results, stats = play_games(WhiteIsLost(), 2, 2, rcfg, max_plies=6, verbose=False, no_resign_frac=1.0)
check("no-resign games are played out", results == ["1/2-1/2"] * 2, f"{results}")
check("they count every would-be resignation that did not lose as a false positive",
      stats == {"resign_checked": 2, "resign_false_pos": 2}, f"{stats}")

samples, results, stats = play_games(WhiteIsLost(), 2, 2, rcfg, max_plies=6, verbose=False,
                                     resign_threshold=-2.0, no_resign_frac=0.0)
check("a threshold below -1 disables resigning", results == ["1/2-1/2"] * 2 and stats["resign_checked"] == 0,
      f"{results} {stats}")

# --------------------------------------------------------------------- puzzles
print("puzzles")
CSV = """PuzzleId,FEN,Moves,Rating,RatingDeviation,Popularity,NbPlays,Themes,GameUrl,OpeningTags
00008,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,f2g3 e6e7 b2b1 b3c1 b1c1 h6c1,1913,75,94,6230,crushing hangingPiece long middlegame,https://lichess.org/787zsVup/black#47,
0000D,5rk1/1p3ppp/pq3b2/8/8/1P1Q1N2/P4PPP/3R2K1 w - - 2 27,d3d6 f8d8 d6d8 f6d8,1580,74,96,3000,advantage endgame short,https://lichess.org/F8M8OS71#53,
mate1,r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR b KQkq - 3 3,a7a6 f3f7,800,80,90,100,mate mateIn1 oneMove opening,https://lichess.org/x,
eq001,r6k/pp2r2p/4Rp1Q/3p4/8/1N1P2R1/PqP2bPP/7K b - - 0 24,f2g3 e6e7,2400,75,94,10,equality,https://lichess.org/x,
"""
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "puzzles.csv")
    with open(path, "w", newline="") as f:
        f.write(CSV)
    puzzles = load_puzzles(path)
    check("all puzzles load", len(puzzles) == 4)
    check("rating filter", [p.puzzle_id for p in load_puzzles(path, min_rating=1600, max_rating=2000)] == ["00008"])
    check("theme filter", [p.puzzle_id for p in load_puzzles(path, themes={"mateIn1"})] == ["mate1"])
    check("limit", len(load_puzzles(path, limit=2)) == 2)
    check("skip passes over the first matching puzzles",
          [p.puzzle_id for p in load_puzzles(path, skip=1, limit=2)] == ["0000D", "mate1"])

p8, pD, pm, peq = puzzles
s = puzzle_samples(p8)
check("one sample per solution move after the setup move", len(s) == 5)
check("solver positions z=+1, opponent positions z=-1", [x[3] for x in s] == [1, -1, 1, -1, 1])
g = ChessGame(p8.fen)
g.play_move(p8.moves[0])
check("first target is the first solver move, hashed in the solver's view",
      s[0][1].tolist() == [g.hash_move(p8.moves[1])] and s[0][2].tolist() == [1.0])
check("--no-opponent keeps only the solver's moves", len(puzzle_samples(p8, include_opponent=False)) == 3)
check("equality puzzles get z = 0", [x[3] for x in puzzle_samples(peq)] == [0.0])


class Oracle(Evaluator):
    """puts all the policy on the next solution move"""

    def __init__(self, puzzle):
        self.p = puzzle

    def evaluate(self, game):
        logits = np.zeros(POLICY_SIZE, dtype=np.float32)
        ply = len(game.board.move_stack)
        if ply < len(self.p.moves):
            logits[game.hash_move(self.p.moves[ply])] = 50.0
        return logits, 0.0


check("an oracle solves every puzzle (raw policy)", all(solve(p, Oracle(p)) for p in puzzles))
check("an oracle solves every puzzle (with search)", all(solve(p, Oracle(p), sims=20) for p in (p8, pD)))
check("search finds a mate in one with no knowledge", solve(pm, UniformEvaluator(), sims=200))
check("uniform policy, no search, misses a multi-move puzzle", not solve(p8, UniformEvaluator()))

# ------------------------------------------------------------------- training
print("training")
net = ChessNet(channels=16, blocks=1)
opt = make_optimizer(net, lr=3e-3)
data = [x for p in puzzles for x in puzzle_samples(p)]
states, pis, zs = collate(data)
check("collate: dense pi rows sum to 1", torch.allclose(pis.sum(1), torch.ones(len(data))))
first = train_epoch(net, opt, data, batch_size=4)
for _ in range(60):
    last = train_epoch(net, opt, data, batch_size=4)
check("loss goes down when overfitting a few puzzles", last[0] < first[0] * 0.5,
      f"{first[0]:.3f} -> {last[0]:.3f}")
rate = solve_rate(puzzles, NetEvaluator(net))
check("the overfit net solves the puzzles it memorised", rate == 1.0, f"{rate:.0%}")

# ------------------------------------------------------------- replay buffer
print("replay buffer on disk")
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "sub", "buffer.npz")
    save_samples(path, data)
    back = load_samples(path)
    check("save/load round trip keeps every sample",
          len(back) == len(data) and all(
              np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
              and a[3] == b[3] for a, b in zip(data, back)))
    check("no temporary file left behind", os.listdir(os.path.dirname(path)) == ["buffer.npz"])

# ---------------------------------------------------------------------- arena
print("arena")
# both players follow the fool's mate line, so black always wins: A wins exactly its black games.
# 1 simulation: with more, white looks ahead, sees g4 loses to Qh4# and (rightly) avoids it
w, l, d_ = play_match(MateFinder(), MateFinder(), num_games=4, parallel=4, config=arena_config(1))
check("colours alternate and wins are credited to the right player", (w, l, d_) == (2, 2, 0),
      f"{(w, l, d_)}")
w, l, d_ = play_match(UniformEvaluator(), UniformEvaluator(), 3, 2, arena_config(4), max_plies=12)
check("every arena game is counted", w + l + d_ == 3)
check("score: win 1, draw 0.5", score(3, 1, 2) == 4 / 6 and score(0, 0, 0) == 0.0)

# ------------------------------------------------------------------ benchmark
print("benchmark")


class FoolsMateEngine:
    """stands in for Stockfish: follows the fool's mate line for either colour"""

    def play(self, board, limit):
        return chess.engine.PlayResult(chess.Move.from_uci(MateFinder.LINE[len(board.move_stack)]), None)


# the net is white in game 0 (walks into the mate) and black in game 1 (delivers it)
w, l, d_ = play_vs_engine(MateFinder(), FoolsMateEngine(), chess.engine.Limit(time=0.01), num_games=2, sims=1)
check("benchmark: colours alternate and results are credited to the net", (w, l, d_) == (1, 1, 0),
      f"{(w, l, d_)}")


class FirstMoveEngine:
    def play(self, board, limit):
        return chess.engine.PlayResult(next(iter(board.legal_moves)), None)


w, l, d_ = play_vs_engine(UniformEvaluator(), FirstMoveEngine(), None, num_games=2, sims=2, max_plies=8)
check("benchmark: a game at max_plies is a draw", (w, l, d_) == (0, 0, 2), f"{(w, l, d_)}")
with tempfile.TemporaryDirectory() as d:
    from pathlib import Path
    path = Path(d) / "benchmark.csv"
    append_benchmark(path, 5, 2, 0, None, 1, 8, 1)
    append_benchmark(path, 10, 4, 0, 1, 2, 7, 1)
    check("benchmark.csv: one header, one row per call",
          path.read_text().splitlines() == ["iteration,generation,skill,depth,wins,losses,draws",
                                            "5,2,0,,1,8,1", "10,4,0,1,2,7,1"])

saved_env = os.environ.pop("STOCKFISH_PATH", None)
try:
    with tempfile.TemporaryDirectory() as d:
        from pathlib import Path
        env_file = Path(d) / ".env"
        check("stockfish_from_env: no variable, no .env -> None", stockfish_from_env(env_file) is None)
        env_file.write_text('# comment\nOTHER=x\nSTOCKFISH_PATH = "C:/sf/stockfish.exe"\n')
        check("stockfish_from_env: read from .env (spaces and quotes stripped)",
              stockfish_from_env(env_file) == "C:/sf/stockfish.exe", repr(stockfish_from_env(env_file)))
        env_file.write_text("STOCKFISH_PATH=\n")
        check("stockfish_from_env: empty value -> None", stockfish_from_env(env_file) is None)
        os.environ["STOCKFISH_PATH"] = "D:/env/stockfish.exe"
        env_file.write_text("STOCKFISH_PATH=C:/sf/stockfish.exe\n")
        check("stockfish_from_env: the environment variable wins over .env",
              stockfish_from_env(env_file) == "D:/env/stockfish.exe")
finally:
    os.environ.pop("STOCKFISH_PATH", None)
    if saved_env is not None:
        os.environ["STOCKFISH_PATH"] = saved_env

# ------------------------------------------------------- command line scripts
print("command line scripts")


def run(script, *args):
    r = subprocess.run([sys.executable, script, *map(str, args)], cwd=ENGINE,
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(r.stdout[-2000:], r.stderr[-2000:])
    return r


TINY = ["--channels", 8, "--blocks", 1]
SP = ["--games-per-iter", 4, "--parallel", 2, "--sims", 4, "--max-plies", 10, "--train-steps", 2,
      "--batch-size", 8, "--arena-games", 2, "--arena-every", 1, "--arena-sims", 4, "--buffer-save-every", 1]
with tempfile.TemporaryDirectory() as d:
    csv_path = os.path.join(d, "puzzles.csv")
    with open(csv_path, "w", newline="") as f:
        f.write(CSV)
    ck = os.path.join(d, "ck")

    r = run("puzzles.py", "--csv", csv_path, "--out-dir", ck, "--epochs", 2, "--batch-size", 4,
            "--val-fraction", 0.25, *TINY)
    check("puzzles.py trains", r.returncode == 0)
    check("puzzles.py writes latest and best checkpoints",
          os.path.exists(os.path.join(ck, "puzzles_latest.pt")) and os.path.exists(os.path.join(ck, "puzzles_best.pt")))
    r = run("puzzles.py", "--csv", csv_path, "--out-dir", ck, "--eval-only",
            "--init", os.path.join(ck, "puzzles_best.pt"), "--eval-sims", 4)
    check("puzzles.py --eval-only", r.returncode == 0 and "solve rate" in r.stdout)

    r = run("self_play.py", "--init", os.path.join(ck, "puzzles_best.pt"), "--out-dir", ck,
            "--iterations", 1, *SP)
    check("self_play.py runs an iteration with an arena", r.returncode == 0 and "arena" in r.stdout)
    files = sorted(os.listdir(ck))
    check("self_play.py writes latest, best and buffer",
          {"selfplay_latest.pt", "selfplay_best.pt", "selfplay_buffer.npz"} <= set(files), f"{files}")
    r = run("self_play.py", "--init", os.path.join(ck, "puzzles_best.pt"), "--out-dir", ck, "--iterations", 1, *SP)
    check("a new run refuses to overwrite an existing one", r.returncode != 0)

    n_before = len(load_samples(os.path.join(ck, "selfplay_buffer.npz")))
    r = run("self_play.py", "--resume", "--out-dir", ck, "--iterations", 2, "--workers", 2, *SP)
    check("--resume --workers 2 continues the run", r.returncode == 0 and "resumed at iteration 2" in r.stdout)
    _, ckpt = load_checkpoint(os.path.join(ck, "selfplay_latest.pt"))
    check("iteration counter carried over", ckpt["iteration"] == 2)
    check("optimizer and scheduler state are saved", ckpt["optimizer"] is not None and ckpt["scheduler"] is not None)
    check("the resumed buffer kept the old positions",
          len(load_samples(os.path.join(ck, "selfplay_buffer.npz"))) > n_before)

    gen_before = ckpt["generation"]
    r = run("self_play.py", "--resume", "--out-dir", ck, "--iterations", 3, *SP, "--arena-games", 0)
    _, ckpt = load_checkpoint(os.path.join(ck, "selfplay_latest.pt"))
    check("--arena-games 0: no arena, the candidate is promoted every iteration",
          r.returncode == 0 and "arena" not in r.stdout and ckpt["generation"] == gen_before + 1)

    r = run("arena.py", os.path.join(ck, "selfplay_latest.pt"), os.path.join(ck, "selfplay_best.pt"),
            "--games", 2, "--sims", 4, "--max-plies", 10)
    check("arena.py compares two checkpoints", r.returncode == 0 and "score" in r.stdout)

print()
if FAILURES:
    print(f"{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1)
print("all passed")
