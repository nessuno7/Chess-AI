"""Tests for ChessGame.undo_move() in python-engine/board.py

Run from the repo root:
    C:/Users/nicol/miniconda3/envs/chessai/python.exe tests/test_undo_move.py

undo_move() exists so a search (MCTS) can walk the tree with play_move/undo_move instead of
deep-copying the whole game at every node. That only works if undo is an exact inverse of
play_move, so these tests all check the same thing from different angles:

    a game that played moves and took some back must be INDISTINGUISHABLE from a fresh game
    that only played the moves that are left.

Indistinguishable means all four pieces of state agree:

    board           the fen, including castling rights, ep square and the halfmove clock
    positions_map   the repetition counter, with no stale zero-count entries left behind
    history         the stored (white_view, black_view) frames
    encode()        the tensor the network actually sees

Section 5 is the one that motivated the change from a bounded deque to a full list: a frame
that has scrolled out of the stored_timesteps window still has to come back when the moves
after it are taken back.
"""

import os
import random
import sys

import chess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "python-engine"))

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))


def replay(ucis, fen=None, stored_timesteps=1):
    """a fresh game that has played exactly these moves -- the reference to compare against"""
    g = ChessGame(fen=fen, stored_timesteps=stored_timesteps)
    for u in ucis:
        g.play_move(chess.Move.from_uci(u))
    return g


def same_state(a, b):
    """every piece of state of two ChessGames, as a list of (what, ok) pairs"""
    frames_match = (len(a.history) == len(b.history) and
                    all(np.array_equal(fa[0], fb[0]) and np.array_equal(fa[1], fb[1])
                        for fa, fb in zip(a.history, b.history)))
    return [
        ("fen", a.board.fen() == b.board.fen()),
        ("move_stack", a.board.move_stack == b.board.move_stack),
        ("positions_map", a.positions_map == b.positions_map),
        ("history", frames_match),
        ("encode", np.array_equal(a.encode(), b.encode())),
    ]


def diff(a, b):
    """the names of the parts that do not match, for the failure message"""
    return ", ".join(what for what, ok in same_state(a, b) if not ok) or "none"


# --------------------------------------------------------------------- import
print("=" * 72)
print("0. Import python-engine/board.py")
print("=" * 72)
try:
    from board import ChessGame
    check("import ChessGame", True)
except Exception as exc:
    check("import ChessGame", False, f"{type(exc).__name__}: {exc}")
    print("\ncannot continue without the import")
    sys.exit(1)


# ------------------------------------------------------------- one move, back
print()
print("=" * 72)
print("1. One move, taken straight back")
print("=" * 72)
try:
    g = ChessGame()
    before = g.encode()
    g.play_move(chess.Move.from_uci("e2e4"))
    returned = g.undo_move()

    check("undo_move returns the move it took back", returned == chess.Move.from_uci("e2e4"),
          f"got {returned}")
    check("board is back to the starting position", g.board.fen() == chess.STARTING_FEN,
          g.board.fen())
    check("encode() is bit-for-bit what it was before the move", np.array_equal(g.encode(), before))
    check("history is back to one frame", len(g.history) == 1, f"got {len(g.history)}")
    check("state matches a fresh game", all(ok for _, ok in same_state(g, ChessGame())),
          diff(g, ChessGame()))
except Exception as exc:
    check("one move round trip", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------ undo at the root
print()
print("=" * 72)
print("2. Undoing past the first position is refused")
print("=" * 72)
try:
    g = ChessGame()
    g.undo_move()
    check("undo on a fresh game raises IndexError", False, "no exception raised")
except Exception as exc:
    check("undo on a fresh game raises IndexError", type(exc).__name__ == "IndexError",
          f"raised {type(exc).__name__}: {exc}")

# a game built from a fen starts AT that fen -- the moves that led to it are not ours to take back
MID = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"
try:
    g = ChessGame(fen=MID)
    g.play_move(chess.Move.from_uci("e1g1"))
    g.undo_move()
    check("a fen game can be undone back to its own fen", g.board.fen() == MID, g.board.fen())
    g.undo_move()
    check("undoing past the fen raises IndexError", False, "no exception raised")
except Exception as exc:
    check("undoing past the fen raises IndexError", type(exc).__name__ == "IndexError",
          f"raised {type(exc).__name__}: {exc}")

# a refused undo must not have half-done anything
try:
    g = ChessGame()
    try:
        g.undo_move()
    except IndexError:
        pass
    check("a refused undo leaves the game untouched",
          all(ok for _, ok in same_state(g, ChessGame())), diff(g, ChessGame()))
except Exception as exc:
    check("refused undo leaves the game untouched", False, f"{type(exc).__name__}: {exc}")


# -------------------------------------------------------- unwind a whole game
print()
print("=" * 72)
print("3. Unwinding a game ply by ply retraces the states it came through")
print("=" * 72)
# Italian game, then a queen trade and castling, so the halfmove clock and castling rights move
GAME = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "e1g1", "f8c5",
        "d2d3", "d7d6", "c1g5", "h7h6", "g5f6", "d8f6"]
try:
    g = ChessGame(stored_timesteps=2)
    snapshots = [(g.board.fen(), g.encode(), dict(g.positions_map))]
    for u in GAME:
        g.play_move(chess.Move.from_uci(u))
        snapshots.append((g.board.fen(), g.encode(), dict(g.positions_map)))

    bad = []
    for ply in range(len(GAME) - 1, -1, -1):
        got = g.undo_move()
        want_move = chess.Move.from_uci(GAME[ply])
        want_fen, want_x, want_map = snapshots[ply]
        if got != want_move:
            bad.append(f"ply {ply}: returned {got}, expected {want_move}")
        if g.board.fen() != want_fen:
            bad.append(f"ply {ply}: fen {g.board.fen()}, expected {want_fen}")
        if not np.array_equal(g.encode(), want_x):
            bad.append(f"ply {ply}: encode() differs")
        if g.positions_map != want_map:
            bad.append(f"ply {ply}: positions_map differs")

    check(f"all {len(GAME)} plies unwind to the exact state they came from", not bad,
          f"{len(bad)} problem(s), first: {bad[:2]}")
    check("back at the starting position", g.board.fen() == chess.STARTING_FEN, g.board.fen())
    check("history is back to one frame", len(g.history) == 1, f"got {len(g.history)}")
except Exception as exc:
    check("unwinding a whole game", False, f"{type(exc).__name__}: {exc}")

# and replaying forward after a full unwind rebuilds the same states again
try:
    for ply, u in enumerate(GAME):
        g.play_move(chess.Move.from_uci(u))
        if not np.array_equal(g.encode(), snapshots[ply + 1][1]):
            raise AssertionError(f"replay diverged at ply {ply}")
    check("replaying the game after unwinding gives the same tensors back", True)
except Exception as exc:
    check("replaying after unwinding", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------- special moves
print()
print("=" * 72)
print("4. Undoing moves that touch more than one square")
print("=" * 72)
# castling (two pieces move), en passant (the captured pawn is not on the target square),
# promotion (the piece that comes back is not the piece that moved), and a capture whose
# piece has to reappear. Each one is played from a fen and taken straight back.
SPECIAL = [
    ("castling, kingside",  "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1g1"),
    ("castling, queenside", "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1", "e1c1"),
    ("castling, black",     "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", "e8c8"),
    ("en passant",          "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 2",                  "e5d6"),
    ("promotion to queen",  "8/1P2k3/8/8/8/8/4K3/8 w - - 0 1",                    "b7b8q"),
    ("underpromotion",      "8/1P2k3/8/8/8/8/4K3/8 w - - 0 1",                    "b7b8n"),
    ("capture",             "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2", "e4d5"),
    ("rook move losing castling rights", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "h1h2"),
]
bad = []
for name, fen, uci in SPECIAL:
    try:
        g = ChessGame(fen=fen, stored_timesteps=2)
        before = g.encode()
        g.play_move(chess.Move.from_uci(uci))
        g.undo_move()
        ref = ChessGame(fen=fen, stored_timesteps=2)
        broken = [what for what, ok in same_state(g, ref) if not ok]
        if not np.array_equal(g.encode(), before):
            broken.append("encode vs before")
        if broken:
            bad.append(f"{name}: {broken}")
    except Exception as exc:
        bad.append(f"{name}: {type(exc).__name__}: {exc}")
check("every special move is restored exactly", not bad, f"first: {bad[:2]}")

# the legal moves themselves have to come back, not just the tensor
try:
    g = ChessGame(fen="r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1")
    before = g.create_legal_moves_mask()
    n_before = len(g.legal_moves_list())
    g.play_move(chess.Move.from_uci("e1g1"))
    g.undo_move()
    check("the legal move mask is the same after castling is taken back",
          np.array_equal(g.create_legal_moves_mask(), before) and
          len(g.legal_moves_list()) == n_before,
          f"{n_before} legal moves before, {len(g.legal_moves_list())} after")
except Exception as exc:
    check("legal move mask after undo", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------- frames outside the window
print()
print("=" * 72)
print("5. Frames that scrolled out of the encoded window come back")
print("=" * 72)
# stored_timesteps=2 means encode() only shows the last 2 frames, but history keeps all of them:
# undoing 4 plies has to bring back frames the tensor had long since stopped showing.
# a bounded deque cannot do this -- the dropped frames are gone -- so this is the regression test
# for history being a full list.
try:
    T = 2
    g = ChessGame(stored_timesteps=T)
    xs = [g.encode()]
    for u in GAME[:8]:
        g.play_move(chess.Move.from_uci(u))
        xs.append(g.encode())

    check("history keeps every frame, not just the window",
          len(g.history) == 9, f"{len(g.history)} frames kept, window is {T}")

    bad = []
    for ply in range(8, 4, -1):              # undo 4 plies, well past the 2-frame window
        g.undo_move()
        if not np.array_equal(g.encode(), xs[ply - 1]):
            bad.append(f"ply {ply - 1}")
    check("encode() after undoing 4 plies still shows the out-of-window frames", not bad,
          f"mismatched at {bad}")
    check("state matches a fresh game replayed to that ply",
          all(ok for _, ok in same_state(g, replay(GAME[:4], stored_timesteps=T))),
          diff(g, replay(GAME[:4], stored_timesteps=T)))

    # the frames really are the old positions, not zeros: with T=2 at ply 4, frame 1 is ply 3.
    # ply 4 has white to move, so the tensor shows the white view of that ply-3 frame -- the ply-3
    # game itself has black to move and would render the same frame flipped, so the comparison is
    # against the stored white view, not against that game's encode()
    ref_frame = replay(GAME[:3], stored_timesteps=T).history[-1][0]
    check("the restored older frame is the real position, not zero padding",
          np.array_equal(g.encode()[7 + 14:7 + 28], ref_frame) and ref_frame.any())
except Exception as exc:
    check("out-of-window frames", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------- repetitions
print()
print("=" * 72)
print("6. positions_map and the repetition planes are rolled back")
print("=" * 72)
# Nf3 Nf6 Ng1 Ng8 walks back to the starting position, so it is seen twice and plane 12 lights up
SHUFFLE = ["g1f3", "g8f6", "f3g1", "f6g8"]
try:
    g = ChessGame()
    start_key = g.board._transposition_key()
    for u in SHUFFLE:
        g.play_move(chess.Move.from_uci(u))

    check("the starting position is counted twice after the shuffle",
          g.positions_map[start_key] == 2, f"count is {g.positions_map.get(start_key)}")
    check("repetition plane 12 is set on the repeated position",
          g.encode()[7 + 12].min() == 1.0)

    for _ in SHUFFLE:
        g.undo_move()

    check("the count is back to 1 after unwinding", g.positions_map[start_key] == 1,
          f"count is {g.positions_map.get(start_key)}")
    check("repetition plane 12 is clear again", g.encode()[7 + 12].max() == 0.0)
    check("positions_map matches a fresh game exactly", g.positions_map == ChessGame().positions_map,
          f"{len(g.positions_map)} entries vs {len(ChessGame().positions_map)}")
    check("no zero-count entries are left behind",
          all(v > 0 for v in g.positions_map.values()),
          f"{[k for k, v in g.positions_map.items() if v <= 0][:2]}")
except Exception as exc:
    check("repetition rollback", False, f"{type(exc).__name__}: {exc}")

# undo, then play something else: the repetition planes must be recomputed for the NEW line,
# not inherited from the line that was taken back
try:
    g = ChessGame()
    for u in SHUFFLE:
        g.play_move(chess.Move.from_uci(u))
    g.undo_move()                                   # take back Ng8, black to move
    g.play_move(chess.Move.from_uci("d7d5"))        # a position never seen before
    check("a different move after undo gets clear repetition planes",
          g.encode()[7 + 12].max() == 0.0 and g.encode()[7 + 13].max() == 0.0)
    check("that line matches a fresh game that played it",
          all(ok for _, ok in same_state(g, replay(SHUFFLE[:3] + ["d7d5"]))),
          diff(g, replay(SHUFFLE[:3] + ["d7d5"])))
except Exception as exc:
    check("recomputed repetition planes", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------ random stress
print()
print("=" * 72)
print("7. Random walks: play and undo in any order, always match a fresh replay")
print("=" * 72)
# this is the search-like access pattern: descend a few plies, back up, descend a different way.
# after every single step the game is compared against a fresh game built from the move list,
# which is the invariant undo_move has to hold for MCTS to be able to reuse one ChessGame.
try:
    rng = random.Random(20260917)
    bad = []
    steps = 0
    for walk in range(40):
        T = rng.choice([1, 2, 4, 8])
        g = ChessGame(stored_timesteps=T)
        played = []
        for _ in range(60):
            # undo more often when deep, so the walk goes up and down instead of just forward
            go_back = played and (rng.random() < 0.35 or not g.legal_moves_list())
            if go_back:
                got = g.undo_move()
                want = chess.Move.from_uci(played.pop())
                if got != want:
                    bad.append(f"walk {walk}: undo returned {got}, expected {want}")
            else:
                moves = g.legal_moves_list()
                if not moves:
                    break
                mv = rng.choice(moves)
                g.play_move(mv)
                played.append(mv.uci())

            steps += 1
            ref = replay(played, stored_timesteps=T)
            broken = [what for what, ok in same_state(g, ref) if not ok]
            if broken:
                bad.append(f"walk {walk}, depth {len(played)}, T={T}: {broken}")
                break

    check(f"{steps} random play/undo steps all match a fresh replay", not bad,
          f"{len(bad)} problem(s), first: {bad[:2]}")
except Exception as exc:
    check("random walk stress", False, f"{type(exc).__name__}: {exc}")

# undoing all the way back from deep inside a random game returns the starting position
try:
    rng = random.Random(7)
    g = ChessGame(stored_timesteps=4)
    n = 0
    for _ in range(80):
        moves = g.legal_moves_list()
        if not moves:
            break
        g.play_move(rng.choice(moves))
        n += 1
    while g.board.move_stack:
        g.undo_move()
    check(f"a {n}-ply random game unwinds completely to the start",
          all(ok for _, ok in same_state(g, ChessGame(stored_timesteps=4))),
          diff(g, ChessGame(stored_timesteps=4)))
except Exception as exc:
    check("full unwind of a random game", False, f"{type(exc).__name__}: {exc}")


print()
print("=" * 72)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL CHECKS PASSED")
