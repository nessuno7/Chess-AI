"""Tests for ChessGame.hash_move() in python-engine/board.py

Run from the repo root:
    C:/Users/nicol/miniconda3/envs/chessai/python.exe tests/test_move_hashing.py

Layout under test - the AlphaZero 8x8x73 = 4672 policy space:

    index = 73*from_square + plane          (from_square is the CANONICAL one,
                                             i.e. mirrored when black is to move)

    plane 0-55   queen moves:     (distance-1) + (direction-1)*7
    plane 56-63  knight moves:    55 + KNIGHT_LOOKUP[(x,y)]
    plane 64-72  underpromotions: 63 + 3*(piece-2) + direction
                                  piece N,B,R = 2,3,4, direction N/NE/NW = 1/2/3
                                  (a promotion can only go N, NE or NW, so this
                                  lands in 64-72 exactly)

The direction numbers are board.py's own, from QUEEN_LOOKUP and KNIGHT_LOOKUP:

    QUEEN_LOOKUP   N:1  NE:2  NW:3  SW:4  E:5  S:6  SE:7  W:8
    block base     (direction-1)*7, so
                   N:0  NE:7  NW:14 SW:21 E:28 S:35 SE:42 W:49

    KNIGHT_LOOKUP  (2,1):1  (-2,1):2  (2,-1):3  (-2,-1):4
                   (1,2):5  (1,-2):6  (-1,2):7  (-1,-2):8
                   keyed by (file_delta, rank_delta), plane = 55 + value

Queen promotions are NOT in 64-72 on purpose: a pawn reaching the last rank
through a normal queen-move plane is assumed to promote to a queen.

Sections 3, 4 and 5 pin these two orders down with exact numbers. Renumber
either lookup table and the expected values there have to move with it.
"""

import os
import random
import sys
import traceback

import chess
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "python-engine"))

N_PLANES = 73
N_MOVES = 64 * N_PLANES         # 4672

FAILURES = []


def check(name, cond, extra=""):
    if not cond:
        FAILURES.append(name)
    print(f"  [{'ok  ' if cond else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))


def plane_of(index):
    return index % N_PLANES


def hashes(game, ucis):
    """hash a list of uci strings from the game's current position"""
    return [game.hash_move(chess.Move.from_uci(u)) for u in ucis]


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
    print("BLOCKED: fix the error above before the rest can run.")
    sys.exit(1)


# ------------------------------------------------------------------- blocks
print()
print("=" * 72)
print("1. Every legal move lands in the right block")
print("=" * 72)
try:
    g = ChessGame()
    idx = hashes(g, ["e2e4", "d2d3", "a2a4"])
    check("pawn pushes are queen moves (plane 0-55)",
          all(0 <= plane_of(i) < 56 for i in idx),
          f"planes {[plane_of(i) for i in idx]}")

    idx = hashes(g, ["g1f3", "g1h3", "b1c3", "b1a3"])
    check("knight moves use the knight planes (56-63)",
          all(56 <= plane_of(i) < 64 for i in idx),
          f"planes {[plane_of(i) for i in idx]}")

    # a position where the same piece can slide a long way
    g = ChessGame(fen="4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1")
    idx = hashes(g, ["d4d8", "d4h8", "d4h4", "d4g1", "d4d1", "d4a1", "d4a4", "d4a7"])
    check("queen slides in all 8 directions stay in 0-55",
          all(0 <= plane_of(i) < 56 for i in idx),
          f"planes {[plane_of(i) for i in idx]}")
    check("queen slides in 8 directions give 8 different planes",
          len(set(plane_of(i) for i in idx)) == 8,
          f"planes {[plane_of(i) for i in idx]}")
except Exception as exc:
    check("block assignment", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------- from-square split
print()
print("=" * 72)
print("2. index splits into from-square and plane")
print("=" * 72)
try:
    g = ChessGame()
    for uci in ("e2e4", "g1f3", "b1c3"):
        i = g.hash_move(chess.Move.from_uci(uci))
        want = chess.parse_square(uci[:2])
        check(f"{uci}: index // 73 == from square {chess.square_name(want)}",
              i // N_PLANES == want,
              f"got square {chess.square_name(i // N_PLANES)} (index {i})")
except Exception as exc:
    check("from-square split", False, f"{type(exc).__name__}: {exc}")


# ----------------------------------------------------------- exact indices
print()
print("=" * 72)
print("3. Exact indices for the current QUEEN_LOOKUP order")
print("=" * 72)
try:
    g = ChessGame()
    # e2 = square 12, N -> direction 1 -> block 0; distance 2 -> plane 1
    check("e2e4 == 73*12 + 1 (N, distance 2)",
          g.hash_move(chess.Move.from_uci("e2e4")) == 73*12 + 1,
          f"got {g.hash_move(chess.Move.from_uci('e2e4'))}")
    # same direction, distance 1 -> plane 0
    check("e2e3 == 73*12 + 0 (N, distance 1)",
          g.hash_move(chess.Move.from_uci("e2e3")) == 73*12 + 0,
          f"got {g.hash_move(chess.Move.from_uci('e2e3'))}")

    # a lone queen on d4 (square 27) reaches every direction
    g = ChessGame(fen="4k3/8/8/8/3Q4/8/8/4K3 w - - 0 1")
    wanted = {
        "d4d8": 0 + 3,      # N  block 0,  distance 4
        "d4h8": 7 + 3,      # NE block 7,  distance 4
        "d4a7": 14 + 2,     # NW block 14, distance 3
        "d4a1": 21 + 2,     # SW block 21, distance 3
        "d4e4": 28 + 0,     # E  block 28, distance 1
        "d4h4": 28 + 3,     # E  block 28, distance 4
        "d4d1": 35 + 2,     # S  block 35, distance 3
        "d4g1": 42 + 2,     # SE block 42, distance 3
        "d4a4": 49 + 2,     # W  block 49, distance 3
    }
    for uci, plane in wanted.items():
        got = g.hash_move(chess.Move.from_uci(uci))
        check(f"{uci} == 73*27 + {plane}", got == 73*27 + plane,
              f"got {got} (plane {plane_of(got)}, expected plane {plane})")

    # castling is a normal king move in python-chess: e1 -> g1 is E, distance 2
    g = ChessGame(fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    check("e1g1 (castling) == 73*4 + 29 (E, distance 2)",
          g.hash_move(chess.Move.from_uci("e1g1")) == 73*4 + 29,
          f"got {g.hash_move(chess.Move.from_uci('e1g1'))}")
    check("e1c1 (castling) == 73*4 + 50 (W, distance 2)",
          g.hash_move(chess.Move.from_uci("e1c1")) == 73*4 + 50,
          f"got {g.hash_move(chess.Move.from_uci('e1c1'))}")
except Exception as exc:
    check("exact indices", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------- knight planes
print()
print("=" * 72)
print("4. All 8 knight destinations get their own plane")
print("=" * 72)
try:
    g = ChessGame(fen="4k3/8/8/8/3N4/8/8/4K3 w - - 0 1")
    # d4 = square 27; plane = 55 + KNIGHT_LOOKUP[(file_delta, rank_delta)]
    wanted = {
        "d4f5": 55 + 1,     # ( 2,  1)
        "d4b5": 55 + 2,     # (-2,  1)
        "d4f3": 55 + 3,     # ( 2, -1)
        "d4b3": 55 + 4,     # (-2, -1)
        "d4e6": 55 + 5,     # ( 1,  2)
        "d4e2": 55 + 6,     # ( 1, -2)
        "d4c6": 55 + 7,     # (-1,  2)
        "d4c2": 55 + 8,     # (-1, -2)
    }
    for uci, plane in wanted.items():
        got = g.hash_move(chess.Move.from_uci(uci))
        check(f"{uci} == 73*27 + {plane}", got == 73*27 + plane,
              f"got {got} (plane {plane_of(got)}, expected plane {plane})")

    planes = [plane_of(i) for i in hashes(g, list(wanted))]
    check("the 8 knight moves cover planes 56-63 exactly",
          sorted(planes) == list(range(56, 64)), f"planes {sorted(planes)}")
except Exception as exc:
    check("knight planes", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------ promotions
print()
print("=" * 72)
print("5. Underpromotions (9 planes) and queen promotions (queen block)")
print("=" * 72)
try:
    # white pawn on b7 (square 49), black rooks on a8 and c8 -> push, capture left, right
    g = ChessGame(fen="r1r3k1/1P6/8/8/8/8/8/6K1 w - - 0 1")
    # plane = 63 + 3*(piece-2) + direction, direction N:1 (push) NE:2 (right) NW:3 (left)
    wanted = {
        "b7b8n": 63 + 0 + 1, "b7c8n": 63 + 0 + 2, "b7a8n": 63 + 0 + 3,
        "b7b8b": 63 + 3 + 1, "b7c8b": 63 + 3 + 2, "b7a8b": 63 + 3 + 3,
        "b7b8r": 63 + 6 + 1, "b7c8r": 63 + 6 + 2, "b7a8r": 63 + 6 + 3,
    }
    for uci, plane in wanted.items():
        got = g.hash_move(chess.Move.from_uci(uci))
        check(f"{uci} == 73*49 + {plane}", got == 73*49 + plane,
              f"got {got} (plane {plane_of(got)}, expected plane {plane})")

    planes = [plane_of(i) for i in hashes(g, list(wanted))]
    check("the 9 underpromotions cover planes 64-72 exactly",
          sorted(planes) == list(range(64, 73)), f"planes {sorted(planes)}")

    # queen promotions reuse the queen block: N block 0, NE block 7, NW block 14,
    # all at distance 1
    qwanted = {"b7b8q": 0, "b7c8q": 7, "b7a8q": 14}
    for uci, plane in qwanted.items():
        got = g.hash_move(chess.Move.from_uci(uci))
        check(f"{uci} == 73*49 + {plane} (queen block, distance 1)",
              got == 73*49 + plane,
              f"got {got} (plane {plane_of(got)}, expected plane {plane})")

    qplanes = [plane_of(i) for i in hashes(g, list(qwanted))]
    check("queen promotion planes disjoint from underpromotion planes",
          not (set(qplanes) & set(planes)))
except Exception as exc:
    check("promotion planes", False, f"{type(exc).__name__}: {exc}")


# -------------------------------------------------------------- symmetry
print()
print("=" * 72)
print("6. Black is mirrored: the same move hashes the same for both colours")
print("=" * 72)
try:
    w = ChessGame()
    b = ChessGame(fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1")
    for white_uci, black_uci in (("e2e4", "e7e5"), ("g1f3", "g8f6"), ("b1c3", "b8c6")):
        check(f"white {white_uci} and black {black_uci} hash the same",
              w.hash_move(chess.Move.from_uci(white_uci))
              == b.hash_move(chess.Move.from_uci(black_uci)),
              f"{w.hash_move(chess.Move.from_uci(white_uci))} vs "
              f"{b.hash_move(chess.Move.from_uci(black_uci))}")

    wc = ChessGame(fen="r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
    bc = ChessGame(fen="r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1")
    check("white e1g1 and black e8g8 (castling) hash the same",
          wc.hash_move(chess.Move.from_uci("e1g1"))
          == bc.hash_move(chess.Move.from_uci("e8g8")),
          f"{wc.hash_move(chess.Move.from_uci('e1g1'))} vs "
          f"{bc.hash_move(chess.Move.from_uci('e8g8'))}")

    # black underpromotion mirrors onto the same plane as white's
    wp = ChessGame(fen="r1r3k1/1P6/8/8/8/8/8/6K1 w - - 0 1")
    bp = ChessGame(fen="6K1/8/8/8/8/8/1p6/R1R3k1 b - - 0 1")
    check("white b7a8=N and black b2a1=N hash the same",
          wp.hash_move(chess.Move.from_uci("b7a8n"))
          == bp.hash_move(chess.Move.from_uci("b2a1n")),
          f"{wp.hash_move(chess.Move.from_uci('b7a8n'))} vs "
          f"{bp.hash_move(chess.Move.from_uci('b2a1n'))}")
except Exception as exc:
    check("colour symmetry", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------ the real test
print()
print("=" * 72)
print("7. Random games: in range, and never two legal moves in one slot")
print("=" * 72)
try:
    random.seed(0)
    out_of_range, collisions, n_moves = [], [], 0

    for _ in range(60):
        g = ChessGame()
        while not g.board.is_game_over() and g.board.ply() < 120:
            moves = g.legal_moves_list()
            slots = {}
            for mv in moves:
                n_moves += 1
                i = g.hash_move(mv)
                if not (0 <= i < N_MOVES):
                    out_of_range.append((g.board.fen(), mv.uci(), i))
                slots.setdefault(i, []).append(mv.uci())
            for i, ucis in slots.items():
                if len(ucis) > 1:
                    collisions.append((g.board.fen(), i, sorted(ucis)))
            g.play_move(random.choice(moves))

    check(f"all {n_moves} hashed moves are in 0..{N_MOVES - 1}",
          not out_of_range,
          f"{len(out_of_range)} bad, first: {out_of_range[:2]}")
    check("no two legal moves in one position share a slot",
          not collisions,
          f"{len(collisions)} collisions, first: {collisions[:2]}")
except Exception as exc:
    check("random game sweep", False, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------- null move
print()
print("=" * 72)
print("8. A move that does not move is rejected")
print("=" * 72)
try:
    g = ChessGame()
    g.hash_move(chess.Move(chess.E2, chess.E2))
    check("from == to raises MoveNotLegalExcpetion", False, "no exception raised")
except Exception as exc:
    check("from == to raises MoveNotLegalExcpetion",
          type(exc).__name__ == "MoveNotLegalExcpetion",
          f"raised {type(exc).__name__}: {exc}")


# ---------------------------------------------------------- legal move mask
print()
print("=" * 72)
print("9. create_legal_moves_mask: the hashes of exactly the legal moves")
print("=" * 72)
try:
    g = ChessGame(fen="r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    mask = g.create_legal_moves_mask()
    want = {g.hash_move(mv) for mv in g.legal_moves_list()}
    if mask is None:
        raise AssertionError("create_legal_moves_mask() returned None -- missing `return mask`")
    got = set(np.nonzero(mask)[0].tolist())
    check("mask marks exactly the hashes of the legal moves", got == want,
          f"{len(want)} legal moves, {len(got)} marked, "
          f"missing {sorted(want - got)[:4]}, extra {sorted(got - want)[:4]}")
except Exception as exc:
    check("create_legal_moves_mask", False, f"{type(exc).__name__}: {exc}")


print()
print("=" * 72)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for name in FAILURES:
        print(f"  - {name}")
    sys.exit(1)
print("ALL CHECKS PASSED")
