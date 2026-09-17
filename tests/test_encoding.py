"""Tests for ChessGame.encode() in python-engine/board.py

Run from the repo root:
    C:/Users/nicol/miniconda3/envs/chessai/python.exe tests/test_encoding.py

Layout under test (from the comments + the history-shift arithmetic in encode):

    shape = (14*T + 7, 8, 8)

    globals, planes 0-6:
        0     player to move (the only absolute plane)
        1-4   castling  our K, our Q, their K, their Q
        5     en passant (rank flipped when black is to move)
        6     halfmove clock / 100

    every square-based plane is seen from the side to move: when black is to
    move, ranks are flipped (r -> 7-r) and files are kept

    then T frames of 14 planes each, frame f at base = 7 + 14*f:
        +0..+5    our   P,N,B,R,Q,K
        +6..+11   their P,N,B,R,Q,K
        +12       1 if the position has occurred once before
        +13       1 if it has occurred twice or more before
"""

import os
import sys
import traceback

import chess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "python-engine"))

N_GLOBALS = 7
N_FRAME = 14

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))


def frame(x, f=0):
    base = N_GLOBALS + f * N_FRAME
    return x[base:base + N_FRAME]


# --------------------------------------------------------------------- import
print("=" * 72)
print("0. Import python-engine/board.py")
print("=" * 72)
try:
    from board import ChessGame  # noqa: E402
    print("  [ok  ] imported ChessGame")
except Exception:
    print("  [FAIL] could not import board.py\n")
    traceback.print_exc()
    print("\n" + "=" * 72)
    print("BLOCKED: fix the import error above before the rest can run.")
    sys.exit(1)


# ------------------------------------------------------------- constructor
print()
print("=" * 72)
print("1. Constructor")
print("=" * 72)
try:
    g = ChessGame()
    check("ChessGame() gives the standard start position",
          g.board.board_fen() == chess.Board().board_fen(),
          f"got {g.board.board_fen()!r}")
    check("ChessGame() board has 32 pieces", len(g.board.piece_map()) == 32,
          f"got {len(g.board.piece_map())}")
except Exception as exc:
    check("ChessGame() constructs", False, f"{type(exc).__name__}: {exc}")

FEN = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
try:
    g = ChessGame(fen=FEN)
    # board.fen() drops the ep square when no capture is legal (python-chess
    # default en_passant="legal"), so ask for it unconditionally
    check("ChessGame(fen=...) honours the FEN",
          g.board.fen(en_passant="fen") == FEN,
          f"got {g.board.fen(en_passant='fen')!r}")
except Exception as exc:
    check("ChessGame(fen=...) constructs", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------- shape
print()
print("=" * 72)
print("2. Tensor shape scales with stored_timesteps")
print("=" * 72)
for T in (1, 2, 4, 8):
    try:
        g = ChessGame(stored_timesteps=T)
        x = g.encode()
        want = (N_FRAME * T + N_GLOBALS, 8, 8)
        check(f"T={T}: shape == {want}", x.shape == want, f"got {x.shape}")
    except Exception as exc:
        check(f"T={T}: encode() runs", False, f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------------- globals
print()
print("=" * 72)
print("3. Global planes 0-6")
print("=" * 72)
try:
    g = ChessGame(stored_timesteps=1)
    x = g.encode()
    check("plane 0 == 1.0 at the start (white to move)", x[0].min() == 1.0,
          f"got {x[0].min()}")
    check("planes 1-4 all 1.0 (all castling rights at start)",
          all(x[i].min() == 1.0 for i in range(1, 5)),
          f"got {[float(x[i].max()) for i in range(1, 5)]}")
    check("plane 5 (en passant) empty at start", x[5].sum() == 0.0,
          f"got sum {x[5].sum()}")
    check("plane 6 (halfmove clock) == 0.0 at start", x[6].max() == 0.0,
          f"got {x[6].max()}")
except Exception as exc:
    check("globals encode", False, f"{type(exc).__name__}: {exc}")

# en passant must land on its own plane, not on top of a castling plane
try:
    g = ChessGame(fen=FEN)          # black to move, ep square e3
    x = g.encode()
    ep = g.board.ep_square
    check("ep square is set in this FEN", ep is not None, f"ep_square={ep}")
    if ep is not None:
        # black to move -> the ep square is seen with ranks flipped (e3 -> e6)
        r, c = 7 - chess.square_rank(ep), chess.square_file(ep)
        check("plane 5 marks the ep square (black view)", x[5, r, c] == 1.0,
              f"x[5][{r}][{c}]={x[5, r, c]}")
        check("plane 5 marks ONLY that square", x[5].sum() == 1.0,
              f"sum={x[5].sum()}")
        check("ep did not overwrite castling plane 4 (their Q right)",
              x[4].min() == 1.0,
              f"plane4 min={x[4].min()} max={x[4].max()} "
              f"-- white still has queenside rights in this FEN")
except Exception as exc:
    check("en passant encoding", False, f"{type(exc).__name__}: {exc}")

# halfmove clock
try:
    g = ChessGame(stored_timesteps=1)
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        g.play_move(chess.Move.from_uci(uci))
    x = g.encode()
    check("plane 6 tracks halfmove clock (4 reversible moves -> 0.04)",
          abs(float(x[6].max()) - 0.04) < 1e-6,
          f"got {float(x[6].max())}, board clock={g.board.halfmove_clock}")
except Exception as exc:
    check("halfmove clock encoding", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------ piece planes
print()
print("=" * 72)
print("4. Piece planes (frame 0 = planes 7-18)")
print("=" * 72)
try:
    g = ChessGame(stored_timesteps=1)
    x = g.encode()
    f0 = frame(x, 0)
    check("frame 0 piece planes sum to 32 at the start", f0[:12].sum() == 32.0,
          f"got {f0[:12].sum()}")

    # our king (piece plane 5) must be a single square
    check("our-king plane has exactly one square", f0[5].sum() == 1.0,
          f"got {f0[5].sum()}")
    check("their-king plane has exactly one square", f0[11].sum() == 1.0,
          f"got {f0[11].sum()}")

    # white to move at the start -> 'our' king is the white king on e1
    r, c = chess.square_rank(chess.E1), chess.square_file(chess.E1)
    check("our king on e1 when white to move", f0[5, r, c] == 1.0,
          f"f0[5] nonzero at {list(zip(*np.nonzero(f0[5])))}")
except Exception as exc:
    check("piece planes", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------ repetition planes
print()
print("=" * 72)
print("5. Repetition planes (frame 0 = planes 19, 20)")
print("=" * 72)
try:
    g = ChessGame(stored_timesteps=1)
    f0 = frame(g.encode(), 0)
    check("fresh start position: both rep planes 0",
          f0[12].max() == 0.0 and f0[13].max() == 0.0,
          f"rep1={f0[12].max()} rep2={f0[13].max()}")

    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        g.play_move(chess.Move.from_uci(uci))
    f0 = frame(g.encode(), 0)
    check("2nd occurrence: rep plane 1 set, rep plane 2 clear",
          f0[12].max() == 1.0 and f0[13].max() == 0.0,
          f"rep1={f0[12].max()} rep2={f0[13].max()}, "
          f"is_repetition(2)={g.board.is_repetition(2)}")

    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        g.play_move(chess.Move.from_uci(uci))
    f0 = frame(g.encode(), 0)
    check("3rd occurrence: both rep planes set",
          f0[12].max() == 1.0 and f0[13].max() == 1.0,
          f"rep1={f0[12].max()} rep2={f0[13].max()}, "
          f"is_repetition(3)={g.board.is_repetition(3)}")
except Exception as exc:
    check("repetition planes", False, f"{type(exc).__name__}: {exc}")


# --------------------------------------------------------------- history
print()
print("=" * 72)
print("6. History: frame f must hold the position f plies ago")
print("=" * 72)
try:
    g = ChessGame(stored_timesteps=3)
    x0 = g.encode()
    check("T=3 start: frames 1 and 2 are zero-padded",
          frame(x0, 1).sum() == 0.0 and frame(x0, 2).sum() == 0.0,
          f"f1={frame(x0, 1).sum()} f2={frame(x0, 2).sum()}")

    g.play_move(chess.Move.from_uci("e2e4"))
    x1 = g.encode()
    check("after 1 move: frame 1 is populated (the previous position)",
          frame(x1, 1)[:12].sum() == 32.0,
          f"frame1 piece sum = {frame(x1, 1)[:12].sum()}")
    check("after 1 move: frame 2 still zero",
          frame(x1, 2).sum() == 0.0,
          f"frame2 sum = {frame(x1, 2).sum()}")

    g.play_move(chess.Move.from_uci("e7e5"))
    x2 = g.encode()
    check("after 2 moves: frame 2 is populated",
          frame(x2, 2)[:12].sum() == 32.0,
          f"frame2 piece sum = {frame(x2, 2)[:12].sum()}")
    check("frames are not all identical (history really shifts)",
          not np.array_equal(frame(x2, 0), frame(x2, 1)))
except Exception as exc:
    check("history shifting", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------ orientation
print()
print("=" * 72)
print("6b. Orientation: everything relative to the side to move")
print("=" * 72)
# A position and its colour-flipped mirror are the same position from the
# mover's point of view, so they must encode identically except plane 0.
MIRROR_FEN = "r3k2r/pppq1ppp/2n2n2/3pp3/1b1PP3/2N2N2/PPPQ1PPP/R3K2R w Kq - 4 8"
try:
    a = ChessGame(fen=MIRROR_FEN).encode()
    b = ChessGame(fen=chess.Board(MIRROR_FEN).mirror().fen()).encode()
    diff = [i for i in range(a.shape[0]) if not np.array_equal(a[i], b[i])]
    check("position vs board.mirror(): only plane 0 differs", diff == [0],
          f"differing planes {diff}")
except Exception as exc:
    check("mirror invariance", False, f"{type(exc).__name__}: {exc}")

# every history frame must use the CURRENT mover's perspective, so 'our pawns'
# means the same colour in frame 0 and frame 1
try:
    g = ChessGame(stored_timesteps=2)
    g.play_move(chess.Move.from_uci("g1f3"))    # black to move now
    x = g.encode()
    f0, f1 = frame(x, 0), frame(x, 1)
    check("frame 0 and frame 1 agree on which side is 'our' pawns",
          np.array_equal(f0[0], f1[0]),
          "frame 1 was built from the previous mover's perspective")
except Exception as exc:
    check("history perspective", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------ flip, square by square
print()
print("=" * 72)
print("6c. Flip: exact squares when black is to move")
print("=" * 72)
# planes are indexed [plane, rank, file]; black view means rank -> 7-rank, file kept
try:
    g = ChessGame(stored_timesteps=2)
    g.play_move(chess.Move.from_uci("e2e4"))    # black to move, ep square e3
    x = g.encode()
    f0, f1 = frame(x, 0), frame(x, 1)

    # frame 0: black pieces are 'ours' and sit at the bottom
    assert f0[5, 0, 4] == 1.0, "our king (black e8) should be at rank 0, file e"
    assert f0[4, 0, 3] == 1.0, "our queen (black d8) should be at rank 0, file d"
    assert f0[4, 0, 4] == 0.0, "queen landed on file e -> files were flipped (rotation, not mirror)"
    assert f0[0, 1].sum() == 8.0, "our 8 pawns (black rank 7) should be on rank 1"
    assert f0[11, 7, 4] == 1.0, "their king (white e1) should be at rank 7, file e"
    assert f0[6, 4, 4] == 1.0, "their e4 pawn should be at rank 4, file e"
    assert f0[6, 6, 4] == 0.0, "their e2 square should be empty"
    assert f0[6, 6].sum() == 7.0, "their other 7 pawns should be on rank 6"

    # frame 1 (start position, white moved from it) is also shown from black's side
    assert f1[0, 1].sum() == 8.0, "frame 1: our pawns (black) should be on rank 1"
    assert f1[6, 6].sum() == 8.0, "frame 1: their pawns (white) should be on rank 6"
    assert f1[5, 0, 4] == 1.0, "frame 1: our king (black) should be at rank 0, file e"

    # globals: ep e3 -> rank 5, file e
    assert x[5, 5, 4] == 1.0, "ep square e3 should be at rank 5, file e"
    assert x[5].sum() == 1.0, "ep plane should mark one square"
    assert x[0].max() == 0.0, "plane 0 should be 0 when black is to move"

    # stored pair: black view is exactly the colour-swapped, rank-flipped white view
    white_view, black_view = g.history[-1]     # history is oldest first, [-1] is the current position
    perm = np.r_[6:12, 0:6, 12:14]
    assert np.array_equal(black_view, white_view[perm, ::-1, :]), "stored black view != flipped white view"
    assert np.array_equal(black_view[perm, ::-1, :], white_view), "flipping twice should give the white view back"
    assert np.array_equal(f0, black_view), "encode() should copy the black view when black is to move"

    # white to move: encode() copies the white view untouched
    g.play_move(chess.Move.from_uci("e7e5"))
    assert np.array_equal(frame(g.encode(), 0), g.history[-1][0]), "encode() should copy the white view when white is to move"

    # castling relative to the mover: black to move with black q, white K -> [our K, our Q, their K, their Q]
    x = ChessGame(fen="r3k2r/8/8/8/8/8/8/R3K2R b Kq - 0 1").encode()
    assert [float(x[i].max()) for i in range(1, 5)] == [0.0, 1.0, 1.0, 0.0], "castling planes not relative to black"

    check("all flip assertions hold", True)
except AssertionError as exc:
    check("all flip assertions hold", False, str(exc))
except Exception as exc:
    check("flip assertions run", False, f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------- legality API
print()
print("=" * 72)
print("7. play_move / legal_moves_list")
print("=" * 72)
try:
    g = ChessGame(stored_timesteps=1)
    check("legal_moves_list() returns 20 moves at the start",
          len(g.legal_moves_list()) == 20, f"got {len(g.legal_moves_list())}")
except Exception as exc:
    check("legal_moves_list()", False, f"{type(exc).__name__}: {exc}")

try:
    g = ChessGame(stored_timesteps=1)
    g.play_move(chess.Move.from_uci("e2e4"))
    check("play_move applies a legal move", g.board.piece_at(chess.E4) is not None)
except Exception as exc:
    check("play_move legal", False, f"{type(exc).__name__}: {exc}")

# illegal move FROM AN OCCUPIED square
try:
    g = ChessGame(stored_timesteps=1)
    g.play_move(chess.Move.from_uci("e2e5"))        # pawn cannot jump 3
    check("illegal move raises MoveNotLegalExcpetion", False, "no exception raised")
except Exception as exc:
    check("illegal move raises MoveNotLegalExcpetion",
          type(exc).__name__ == "MoveNotLegalExcpetion",
          f"raised {type(exc).__name__}: {exc}")

# illegal move FROM AN EMPTY square -- piece_at() returns None
try:
    g = ChessGame(stored_timesteps=1)
    g.play_move(chess.Move.from_uci("e4e5"))        # nothing on e4
    check("illegal move from an empty square raises MoveNotLegalExcpetion",
          False, "no exception raised")
except Exception as exc:
    check("illegal move from an empty square raises MoveNotLegalExcpetion",
          type(exc).__name__ == "MoveNotLegalExcpetion",
          f"raised {type(exc).__name__}: {exc}")


print()
print("=" * 72)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL CHECKS PASSED")
