"""
Self play: N games (10 by default) played side by side, in lockstep, sharing one network.

Every game has its own ChessGame and its own MonteCarloTS. A move is made in all of them at the
same time: search_many runs the simulations round by round, and in every round one leaf from each
tree goes through the net in a single batch of up to N positions. The trees reuse their subtree
after every move (MonteCarloTS.advance). A finished game drops out and, while more games are
still owed, a fresh one takes its slot, so the batch stays full.

With --workers W > 1 the games of an iteration are split over W processes, each running its own
lockstep group of --parallel games on its own copy of the net. The search is plain python and
uses one core, so this is where a multi-core CPU pays off.

Two nets are kept:
    best        plays the self-play games
    candidate   is trained on them. By default (--arena-games 0, the AlphaZero scheme) it becomes
                best after every iteration. With --arena-games > 0 it is gated instead (AlphaGo
                Zero): it replaces best only after scoring >= --gate-threshold against it in an
                --arena-games match every --arena-every iterations. Use 100+ games if gating:
                with ~80% draws a 20 game match is mostly noise.

Resignation: a side whose best root move has Q < --resign-threshold resigns, and the game is
scored as a loss for it. In a --no-resign-frac share of the games resigning is disabled, and
those games measure the false positive rate (a side that would have resigned but did not lose),
printed every iteration; lower the threshold if it climbs above ~5%.

With --stockfish PATH the best net plays --bench-games against Stockfish at --bench-skill every
--bench-every iterations, appended to <out-dir>/benchmark.csv (see benchmark.py).

Training loop:
    repeat:  best plays --games-per-iter games  ->  replay buffer  ->  --train-steps steps on
             the candidate  ->  (arena)  ->  save checkpoints

Everything lands in --out-dir (default <repo>/checkpoints):
    selfplay_latest.pt      candidate + optimizer + lr schedule + iteration counter
    selfplay_best.pt        the current best net
    selfplay_buffer.npz     the replay buffer, every --buffer-save-every iterations
--resume picks all three back up and carries on where the run stopped.

    python self_play.py --init ../checkpoints/puzzles_best.pt --workers 8 --stockfish C:/tools/stockfish.exe
    python self_play.py --resume
"""

import argparse
import copy
import multiprocessing as mp
import random
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import chess
import torch

from arena import arena_config, play_match, score
from benchmark import append_benchmark, play_vs_stockfish
from board import ChessGame
from monte_carlo_tree_search import MCTSConfig, MonteCarloTS, search_many
from network import CHECKPOINT_DIR, ChessNet, NetEvaluator, load_checkpoint, save_checkpoint
from training import load_samples, make_optimizer, make_sample, save_samples, train_steps


class SelfPlayGame:
    """One game in progress: its board, its tree, and the samples it has produced so far."""

    def __init__(self, evaluator, config, stored_timesteps, seed=None, can_resign=True):
        self.game = ChessGame(stored_timesteps=stored_timesteps)
        self.tree = MonteCarloTS(self.game, evaluator, config, seed=seed)
        self.history = []   # (sample without z, side to move) per ply
        self.ply = 0
        self.can_resign = can_resign
        self.would_have_resigned = set()   # colours that crossed the threshold in a no-resign game

    def root_value(self):
        """Q of the most visited root move, for the side to move: what the search makes of the position."""
        root = self.tree.root
        k = int(root.N.argmax())
        return float(root.W[k] / max(root.N[k], 1))

    def result(self):
        """
        None while the game is going, else the value for the side to move. Uses the same rules
        as the search (_terminal_value), so threefold / fifty moves end the game as a draw.
        """
        if self.tree.root.is_terminal:
            return self.tree.root.terminal_value
        return self.tree._terminal_value()

    def finish(self, value_for_side_to_move):
        """Assign z to every stored position and return the finished samples."""
        last_turn = self.game.board.turn
        samples = []
        for (state, idx, p, _), turn in self.history:
            z = value_for_side_to_move if turn == last_turn else -value_for_side_to_move
            samples.append((state, idx, p, z))
        return samples


def play_games(evaluator, num_games, parallel=10, config=None, stored_timesteps=1,
               max_plies=400, verbose=True, resign_threshold=-0.9, no_resign_frac=0.2):
    """
    Play `num_games` self-play games, `parallel` at a time, and return (samples, results, stats):
    results holds one of "1-0", "0-1", "1/2-1/2" per game, stats counts the resign checks
    ("resign_checked": would-be resignations in no-resign games, "resign_false_pos": how many of
    those sides did not go on to lose). A resign_threshold below -1 disables resigning.
    """
    config = config if config is not None else MCTSConfig()
    samples, results = [], []
    stats = {"resign_checked": 0, "resign_false_pos": 0}
    started = 0

    def new_game():
        nonlocal started
        started += 1
        return SelfPlayGame(evaluator, config, stored_timesteps, seed=random.randrange(2**32),
                            can_resign=random.random() >= no_resign_frac)

    active = [new_game() for _ in range(min(parallel, num_games))]
    t0 = time.time()

    while active:
        search_many([g.tree for g in active], evaluator)

        still_running = []
        for g in active:
            idx, p = g.tree.policy_target_sparse(temperature=1.0)
            g.history.append((make_sample(g.game, idx, p, 0.0), g.game.board.turn))

            value = None
            if g.root_value() < resign_threshold:
                if g.can_resign:
                    value = -1.0   # the side to move resigns, finish() scores the game from here
                else:
                    g.would_have_resigned.add(g.game.board.turn)
            if value is None:
                move = g.tree.select_move(g.ply)
                g.tree.advance(move)
                g.ply += 1
                value = g.result()
                if value is None and g.ply >= max_plies:
                    value = 0.0   # adjudicated draw, keeps a shuffling game from running forever
            if value is None:
                still_running.append(g)
                continue

            samples.extend(g.finish(value))
            results.append(_result_string(g.game.board.turn, value))
            for colour in g.would_have_resigned:
                stats["resign_checked"] += 1
                stats["resign_false_pos"] += results[-1] != ("0-1" if colour == chess.WHITE else "1-0")
            if verbose:
                print(f"  game {len(results)}/{num_games}: {results[-1]} in {g.ply} plies "
                      f"({time.time() - t0:.0f}s)")
            if started < num_games:
                still_running.append(new_game())

        active = still_running

    return samples, results, stats


def _result_string(side_to_move, value):
    if value == 0:
        return "1/2-1/2"
    white_won = (value > 0) == (side_to_move == chess.WHITE)
    return "1-0" if white_won else "0-1"


# ---- worker processes ----------------------------------------------------------------

def _worker_init():
    torch.set_num_threads(1)   # W processes each with a full thread pool would just fight


def _worker_play(hparams, state_dict, num_games, parallel, config, max_plies, seed,
                 resign_threshold, no_resign_frac):
    """Runs in a worker process: rebuild the net from its weights and play `num_games`."""
    random.seed(seed)
    net = ChessNet(**hparams)
    net.load_state_dict(state_dict)
    return play_games(NetEvaluator(net), num_games, parallel, config, hparams["stored_timesteps"],
                      max_plies, verbose=False, resign_threshold=resign_threshold,
                      no_resign_frac=no_resign_frac)


def generate(net, args, config, pool):
    """(samples, results, stats) for one iteration, in this process or spread over the worker pool."""
    if pool is None:
        return play_games(NetEvaluator(net, args.device), args.games_per_iter, args.parallel,
                          config, net.stored_timesteps, args.max_plies,
                          resign_threshold=args.resign_threshold, no_resign_frac=args.no_resign_frac)

    cpu_state = {k: v.detach().cpu() for k, v in net.state_dict().items()}
    share = [args.games_per_iter // args.workers + (i < args.games_per_iter % args.workers)
             for i in range(args.workers)]
    futures = [pool.submit(_worker_play, net.hparams(), cpu_state, n, args.parallel, config,
                           args.max_plies, random.randrange(2**32), args.resign_threshold,
                           args.no_resign_frac)
               for n in share if n > 0]
    samples, results = [], []
    stats = {"resign_checked": 0, "resign_false_pos": 0}
    for f in futures:
        s, r, st = f.result()
        samples.extend(s)
        results.extend(r)
        for k in stats:
            stats[k] += st[k]
    return samples, results, stats


# ---- main loop -------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="AlphaZero style self-play training")
    ap.add_argument("--init", help="checkpoint to start a NEW run from (e.g. the puzzle-trained net)")
    ap.add_argument("--resume", action="store_true", help="continue the run saved in --out-dir")
    ap.add_argument("--out-dir", default=str(CHECKPOINT_DIR))
    ap.add_argument("--iterations", type=int, default=100, help="total iterations, counting resumed ones")
    ap.add_argument("--games-per-iter", type=int, default=20)
    ap.add_argument("--parallel", type=int, default=10, help="games played side by side, per process")
    ap.add_argument("--workers", type=int, default=1, help="processes playing games")
    ap.add_argument("--sims", type=int, default=400, help="MCTS simulations per move")
    ap.add_argument("--max-plies", type=int, default=400)
    ap.add_argument("--buffer-size", type=int, default=200_000, help="replay buffer, in positions")
    ap.add_argument("--buffer-save-every", type=int, default=5, help="iterations between buffer saves")
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr-milestones", type=int, nargs="*", default=[],
                    help="iterations at which the lr is divided by 10")
    ap.add_argument("--resign-threshold", type=float, default=-0.9,
                    help="resign below this root Q; anything below -1 disables resigning")
    ap.add_argument("--no-resign-frac", type=float, default=0.2,
                    help="share of games with resigning off, to measure false positives")
    ap.add_argument("--arena-games", type=int, default=0,
                    help="0 = no gate, candidate becomes best every iteration (AlphaZero); if gating, use 100+")
    ap.add_argument("--arena-every", type=int, default=5)
    ap.add_argument("--arena-sims", type=int, default=None, help="default: --sims")
    ap.add_argument("--gate-threshold", type=float, default=0.55)
    ap.add_argument("--stockfish", help="path to a Stockfish binary, turns the benchmark on")
    ap.add_argument("--bench-every", type=int, default=5, help="iterations between Stockfish benchmarks")
    ap.add_argument("--bench-games", type=int, default=10)
    ap.add_argument("--bench-skill", type=int, default=0, help="Stockfish Skill Level (0-20)")
    ap.add_argument("--bench-depth", type=int, default=None,
                    help="cap Stockfish's search depth (e.g. 1) for an even weaker opponent")
    ap.add_argument("--bench-sims", type=int, default=None, help="default: --sims")
    ap.add_argument("--channels", type=int, default=64, help="only for a new run without --init")
    ap.add_argument("--blocks", type=int, default=6, help="only for a new run without --init")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    out = Path(args.out_dir)
    latest_path = out / "selfplay_latest.pt"
    best_path = out / "selfplay_best.pt"
    buffer_path = out / "selfplay_buffer.npz"

    # ---- nets, optimizer, schedule, buffer: fresh or resumed
    start, ckpt = 1, None
    buffer = deque(maxlen=args.buffer_size)
    if args.resume:
        if not latest_path.exists():
            raise SystemExit(f"nothing to resume: {latest_path} does not exist")
        candidate, ckpt = load_checkpoint(latest_path, args.device)
        best = load_checkpoint(best_path, args.device)[0] if best_path.exists() else copy.deepcopy(candidate)
        start = ckpt.get("iteration", 0) + 1
        if buffer_path.exists():
            buffer.extend(load_samples(buffer_path))
        print(f"resumed at iteration {start}, buffer {len(buffer)} positions")
    else:
        if latest_path.exists():
            raise SystemExit(f"{latest_path} already exists: pass --resume, or another --out-dir")
        if args.init:
            candidate, _ = load_checkpoint(args.init, args.device)
        else:
            candidate = ChessNet(channels=args.channels, blocks=args.blocks).to(args.device)
        best = copy.deepcopy(candidate)

    optimizer = make_optimizer(candidate, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)
    generation = 0
    if ckpt is not None:
        if ckpt.get("optimizer"):
            optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scheduler"):
            scheduler.load_state_dict(ckpt["scheduler"])
        generation = ckpt.get("generation", 0)

    config = MCTSConfig(num_simulations=args.sims)
    pool = None
    if args.workers > 1:
        pool = ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn"), initializer=_worker_init)

    try:
        for it in range(start, args.iterations + 1):
            t0 = time.time()
            print(f"iteration {it}: best net (generation {generation}) plays {args.games_per_iter} games")
            samples, results, stats = generate(best, args, config, pool)
            buffer.extend(samples)
            w, b, d = results.count("1-0"), results.count("0-1"), results.count("1/2-1/2")
            print(f"  +{len(samples)} positions (buffer {len(buffer)}), W/B/D = {w}/{b}/{d}, "
                  f"{time.time() - t0:.0f}s")
            if stats["resign_checked"]:
                fp, n = stats["resign_false_pos"], stats["resign_checked"]
                print(f"  resign check: {fp}/{n} would-be resignations were false positives "
                      f"({fp / n:.0%}, keep under ~5%)")

            loss, pl, vl = train_steps(candidate, optimizer, list(buffer), args.train_steps,
                                       args.batch_size, args.device)
            scheduler.step()
            print(f"  loss {loss:.3f} (policy {pl:.3f}, value {vl:.3f}), lr {scheduler.get_last_lr()[0]:.1e}")

            if args.arena_games <= 0:
                best.load_state_dict(candidate.state_dict())
                generation += 1
            elif it % args.arena_every == 0:
                cw, cl, cd = play_match(NetEvaluator(candidate, args.device), NetEvaluator(best, args.device),
                                        args.arena_games, args.parallel,
                                        arena_config(args.arena_sims or args.sims),
                                        candidate.stored_timesteps, args.max_plies)
                s = score(cw, cl, cd)
                promoted = s >= args.gate_threshold
                if promoted:
                    best.load_state_dict(candidate.state_dict())
                    generation += 1
                print(f"  arena candidate vs best: {cw}W {cl}L {cd}D = {s:.1%} -> "
                      + (f"promoted to generation {generation}" if promoted else "kept the old best"))

            if args.stockfish and it % args.bench_every == 0:
                bw, bl, bd = play_vs_stockfish(NetEvaluator(best, args.device), args.stockfish,
                                               args.bench_games, args.bench_sims or args.sims,
                                               args.bench_skill, depth=args.bench_depth,
                                               stored_timesteps=best.stored_timesteps,
                                               max_plies=args.max_plies)
                print(f"  vs stockfish skill {args.bench_skill}"
                      + (f" depth {args.bench_depth}" if args.bench_depth else "")
                      + f": {bw}W {bl}L {bd}D = {score(bw, bl, bd):.1%}")
                append_benchmark(out / "benchmark.csv", it, generation, args.bench_skill,
                                 args.bench_depth, bw, bl, bd)

            save_checkpoint(latest_path, candidate, optimizer, iteration=it, generation=generation,
                            scheduler=scheduler.state_dict())
            save_checkpoint(best_path, best, generation=generation)
            if it % args.buffer_save_every == 0 or it == args.iterations:
                save_samples(buffer_path, list(buffer))
                print(f"  buffer saved ({len(buffer)} positions)")
    finally:
        if pool is not None:
            pool.shutdown()


if __name__ == "__main__":
    main()
