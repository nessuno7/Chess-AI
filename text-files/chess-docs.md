# chess-docs.md

Working reference for the pure-Python Chess AI. Every API call below was verified against **`chess` 1.11.2 / Python 3.10.19** in the `chessai` conda env.

---

## 0. Environment

```bash
conda activate chessai
pip install chess          # PyPI name is `chess`, project name is python-chess
```

Already in `chessai`: `python 3.10.19`, `pytorch 2.7.0` (**CPU/MKL build - no CUDA**), `numpy 2.2.6`, `chess 1.11.2`.

> If you later want GPU training you must reinstall PyTorch with a CUDA build; the current one will silently run on CPU forever.

Gotcha: `conda run -n chessai python -c "..."` **fails** if the code contains newlines (`AssertionError: Support for scripts where arguments contain newlines not implemented`). Either write a file, or call the interpreter directly:

```bash
C:/Users/nicol/miniconda3/envs/chessai/python.exe script.py
```

---

## 1. The four core objects

| Object | What it is | Notes |
|---|---|---|
| `chess.Board` | Full position + move stack | Mutable. `push`/`pop` is the hot path. |
| `chess.Move` | `from_square`, `to_square`, `promotion` | Immutable, hashable, no board context. |
| `chess.Piece` | `piece_type` (1-6), `color` (bool) | `chess.PAWN=1 ... chess.KING=6` |
| square | plain `int` 0-63 | `A1=0`, `H8=63`. Rank-major: `sq = rank*8 + file`. |

Colors are **booleans**: `chess.WHITE is True`, `chess.BLACK is False`. This matters - `if board.turn:` means "white to move".

```python
import chess
board = chess.Board()                       # standard start position
board = chess.Board(fen)                    # from FEN
board = chess.Board(None)                   # empty board
```

---

## 2. Function cheat-sheet (all verified)

### Move generation & legality

```python
board.legal_moves                 # generator-like; iterate it
board.legal_moves.count()         # 20 at start - cheaper than len(list(...))
move in board.legal_moves         # membership test works
board.push(move)                  # apply (mutates)
board.pop()                       # undo, returns the Move
board.push_uci("e2e4")            # parse + apply
board.parse_san("Nf3")            # SAN -> Move
board.san(move)                   # Move -> SAN   (call BEFORE push)
```

`legal_moves` already excludes moves that leave your king in check, and already *includes* castling, en passant and promotions. **You never write this logic yourself.**

### Squares & coordinates

```python
chess.parse_square("e4")     # -> 28
chess.square_name(28)        # -> "e4"
chess.square(file, rank)     # (4,3) -> 28     both 0-indexed
chess.square_file(28)        # -> 4
chess.square_rank(28)        # -> 3
chess.square_mirror(28)      # vertical flip
chess.SQUARES                # all 64 ints
chess.PIECE_TYPES            # [1,2,3,4,5,6]
```

### Reading the position

```python
board.piece_at(chess.E1)          # Piece or None
board.piece_map()                 # {square: Piece} for occupied squares only
board.turn                        # True = white to move
board.halfmove_clock              # 50-move-rule counter
board.fullmove_number             # starts at 1
board.ply()                       # halfmoves since start
board.ep_square                   # en passant target square or None
board.fen() / board.board_fen() / board.epd()
board.attackers(chess.WHITE, chess.E4)   # -> SquareSet
board.is_attacked_by(chess.WHITE, chess.E4)
board.is_check()
board.has_kingside_castling_rights(chess.WHITE)
board.has_queenside_castling_rights(chess.WHITE)
```

### Move classification (call **before** `push`)

```python
board.is_capture(move)
board.gives_check(move)
board.is_en_passant(move)
board.is_castling(move)
```

### Game termination

```python
board.is_game_over()              # bool
board.outcome()                   # None, or Outcome(termination=..., winner=...)
board.result()                    # "1-0" / "0-1" / "1/2-1/2" / "*"
board.is_checkmate()
board.is_stalemate()
board.is_insufficient_material()
board.is_repetition(3)
board.can_claim_draw()            # threefold or 50-move
```

`outcome().winner` is `True` (white), `False` (black), or `None` (draw).

**Careful:** `winner=False` means *black won*, not "no winner". Test draws with `is None`.

```python
oc = board.outcome()
if oc: print(oc.termination, oc.winner)
# Outcome(termination=<Termination.CHECKMATE: 1>, winner=False)
```

### Bitboards (fast bulk access)

```python
board.pawns, board.knights, board.bishops, board.rooks, board.queens, board.kings
board.occupied_co[chess.WHITE]    # 0xffff at start
board.occupied
chess.scan_reversed(bb)           # iterate set bits -> square ints
chess.msb(bb), chess.lsb(bb)
```

`white_pawns = board.pawns & board.occupied_co[chess.WHITE]`

### Copying, hashing, mirroring

```python
board.copy(stack=False)              # MUCH faster - drops move history
board._transposition_key()           # exact, hashable, 1.6 us  <- USE THIS
chess.polyglot.zobrist_hash(board)   # 64-bit Polyglot key, but 85 us (recomputed!)
board.mirror()                       # flip colors + ranks (data augmentation)
```

`copy(stack=False)` is the one to use in search/MCTS. The default copy duplicates the whole move stack and is a real cost in a hot loop.

### Rendering & engines

```python
import chess.svg
chess.svg.board(board)               # SVG string; renders inline in Jupyter
chess.svg.board(board, lastmove=mv, check=sq, arrows=[...])

import chess.engine
eng = chess.engine.SimpleEngine.popen_uci(r"C:\path\to\stockfish.exe")
info = eng.analyse(board, chess.engine.Limit(depth=15))
info["score"], info["pv"]
res = eng.play(board, chess.engine.Limit(time=0.1)); res.move
eng.quit()                           # always - it's a subprocess
```

---

## 3. Board -> tensor (network input)

Implementation: **`python-engine/encoding.py`**, covered by `python-engine/test_encoding.py`.

Follows the `plan.txt` design: a reduced AlphaZero encoding with the number of history timesteps `T` as a **hyperparameter**, and the global constants at the **front** of the tensor.

```
[ 8 global planes ][ frame t=0 ][ frame t=1 ] ... [ frame t=T-1 ]
                     current      1 ply ago        T-1 plies ago
```

| Plane | Content |
|---|---|
| 0 | colour (1 = white to move) - **absolute**, see below |
| 1-2 | **our** castling rights K/Q |
| 3-4 | **their** castling rights K/Q |
| 5 | no-progress counter (`halfmove_clock / 100`) |
| 6 | move count (`fullmove_number / 200`) |
| 7 | en passant target square |
| *then, per frame (13 planes)* | |
| +0-5 | our P,N,B,R,Q,K |
| +6-11 | their P,N,B,R,Q,K |
| +12 | repetition count of that position (0 / 0.5 / 1.0) |

`n_channels(T) = 8 + 13*T` -> `T=1`: 21, `T=4`: 60, **`T=8`: 112**.

```python
from encoding import encode, n_channels
x = encode(board, history=8)        # (112, 8, 8) float32
```

Deviations from AlphaZero, deliberately: **1 repetition plane per frame instead of 2** (a saturating scalar carries the same information more compactly), and **en passant as an 8th global**. AlphaZero omits ep because 8 history frames let the network infer it - but since `T` is tunable here and `T=1` has no history at all, ep must be explicit or the encoding is not Markov.

### Orientation - the rule that makes it correct

Everything is **relative to the side to move**:

- piece planes are our/their, not white/black
- the board is **mirrored vertically when black is to move**, so "forward" is always up
- castling planes are ordered `(us-K, us-Q, them-K, them-Q)`
- **all history frames use the *current* player's orientation**, so frames are directly comparable to each other

The invariant that catches mistakes, asserted in the tests: a position and its `board.mirror()` must produce **identical tensors except plane 0**. Plane 0 is deliberately absolute - white has a real first-move advantage, and AlphaZero kept a colour plane too.

> **Bug this replaces.** The previous version of this section relativized piece *type* (our/their) but indexed squares **absolutely**. That is only half-relative and fails the mirror test with a piece-plane difference of 64. It also listed castling as absolute `K/Q/k/q`, which flips under mirror (`[1,0,0,1]` vs `[0,1,1,0]`). Both are fixed, and both are kept as regression tests in section A of the test file.

### History and zero-padding

Frames with no corresponding position (the opening plies) are left **zero**. Do not repeat frame 0 into them - the network must be able to distinguish "no history" from "position unchanged".

### Repetition planes

Each frame carries its own repetition count, saturating at 2 (`0.0 / 0.5 / 1.0`). Not optional: without it a drawn-by-repetition position is indistinguishable from a winning one and the value head cannot be correct.

Counting scans only the **reversible window** (`min(len(move_stack), halfmove_clock)`) - a position before a pawn move or capture can never recur. Verified to agree with `board.is_repetition(2)` at every ply of a repeating line.

### Cost

Measured mid-game (stack depth 60):

| history `T` | us/call | channels |
|---|---|---|
| 1 | 748 | 21 |
| 4 | 990 | 60 |
| 8 | 1,592 | 112 |
| 16 | 2,907 | 216 |

~1.6 ms per position at `T=8`. Fine for supervised training if you encode inside DataLoader workers; **too slow for an MCTS inner loop** - there, cache encoded frames and shift the stack rather than re-encoding the full history at every node.

---

## 4. Move -> index (policy head)

The policy head outputs a fixed-size vector; you need a stable `Move <-> int` bijection.

### Option A - 4096 + underpromotions (simple, recommended to start)

```python
def move_to_index(move: chess.Move) -> int:
    if move.promotion and move.promotion != chess.QUEEN:
        # underpromotion: N=2, B=3, R=4 -> offsets 0,1,2
        ff = chess.square_file(move.from_square)
        tf = chess.square_file(move.to_square)
        return 4096 + (move.promotion - 2) * 24 + ff * 3 + (tf - ff + 1)
    return move.from_square * 64 + move.to_square     # 0..4095
# vector size: 4096 + 72 = 4168
```

Queen promotion is the default and shares the plain `from*64+to` slot - safe, because a pawn reaching the last rank *must* promote, so the move is unambiguous.

Underpromotions need their own block because `from*64+to` cannot distinguish `a7a8=N` from `b7a8=N`. The `ff * 3 + (tf - ff + 1)` term encodes *from-file* (0-7) and *direction* (capture-left / push / capture-right, as 0/1/2), which is unique. **Verified collision-free** over all 88 geometrically possible promotion moves, and against the full 4096 plain-move range.

This assumes the side-to-move-relative orientation from §3, so promotions always travel "up" the board. If you use absolute orientation, you need to double this block for black.

### Option B - AlphaZero 8x8x73 = 4672

56 "queen" moves (8 directions x 7 distances) + 8 knight moves + 9 underpromotions (3 pieces x 3 directions), per from-square. More structured, better inductive bias, more code. Move to it if A plateaus.

### Legal-move masking is mandatory

Never softmax over the raw head:

```python
logits = net(x)                                  # (4168,)
mask = torch.full_like(logits, float("-inf"))
for mv in board.legal_moves:
    mask[move_to_index(mv)] = 0.0
probs = torch.softmax(logits + mask, dim=-1)
```

Without the mask the net wastes capacity learning legality - which `board.legal_moves` already gives you for free, exactly.

---

## 5. Lichess puzzle pipeline

Re-download: <https://database.lichess.org/#puzzles> (`lichess_db_puzzle.csv.zst`, ~1 GB decompressed). It is **not** in git and never was.

Columns:

```
PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags
```

### THE gotcha

`Moves` is a UCI sequence where **the first move is the opponent's**. The puzzle position is the one *after* that move. Training on `moves[0]` teaches the net to predict the wrong side's move.

```python
import csv, chess

def puzzle_positions(path):
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            board = chess.Board(row["FEN"])
            moves = row["Moves"].split()
            board.push_uci(moves[0])              # opponent's setup move
            for i, uci in enumerate(moves[1:]):
                mv = chess.Move.from_uci(uci)
                if i % 2 == 0:                    # solver's move -> a label
                    yield board.copy(stack=False), mv, int(row["Rating"])
                board.push(mv)
```

Only every *other* move is a solver move - the ones in between are the opponent's forced replies. Yielding all of them mixes both sides into your labels.

Practical notes:

- Stream with `csv.DictReader`; never `pd.read_csv` the whole 1 GB.
- Filter by `Rating` to build a curriculum (start ~1000-1500).
- `Themes` (e.g. `mateIn2`, `fork`, `pin`) makes good eval slices.
- Sanity-assert `mv in board.legal_moves` while parsing. A failure means your move-index or FEN handling is off - very likely the source of the old `NoKingPiece` exception on `lichess-training`.

---

## 6. Search

Supervised puzzle-solving needs no search. Playing strength does.

### Alpha-beta (classical, easy win)

```python
def negamax(board, depth, alpha, beta):
    if depth == 0 or board.is_game_over():
        return evaluate(board)
    best = -inf
    for mv in board.legal_moves:            # order captures first!
        board.push(mv)
        score = -negamax(board, depth - 1, -beta, -alpha)
        board.pop()
        best = max(best, score)
        alpha = max(alpha, score)
        if alpha >= beta:
            break                           # cutoff
    return best
```

Move ordering matters more than depth: try `board.is_capture(mv)` and `board.gives_check(mv)` first and you roughly double effective depth.

### MCTS + policy/value net (AlphaZero shape)

Four steps per simulation: **select** (PUCT), **expand** (one net eval -> priors + value), **backup**, then move by visit count. Cache net evals by `board._transposition_key()` (**not** `zobrist_hash`: 1.6 us vs 85 us); batch leaf evaluations or the GPU sits idle.

---

## 7. Performance - the real numbers

Measured on this machine, `chessai` env:

```
PERFT(4) = 197,281 nodes in 6.39s  ->  ~30,900 nodes/sec
```

(197,281 is the known-correct perft(4) value - a good correctness self-test.)

**That is slow**, and it is a hard ceiling on pure-Python `push`/`pop`. Implications:

- Supervised learning from puzzles: **fine**. You're NN-bound, not movegen-bound.
- Alpha-beta to depth 5-6: seconds per move. Playable, not strong.
- Full RL self-play: this **is** your bottleneck. Millions of games are not happening at 30k nodes/sec.

Mitigations in order of effort: `copy(stack=False)`; cache by `_transposition_key()`; parallelize games across processes (`multiprocessing`, not threads - the GIL); batch NN evals; and only then consider a native-backed movegen.

Do not optimize this now. Get correctness first.

---

## 8. Pitfalls

1. **`winner=False` means black won.** Test draws with `winner is None`.
2. **Colors are bools.** `piece.color == chess.WHITE`, never `== 1` in a dict key you also index with ints.
3. **`board.san(move)` must be called before `board.push(move)`** - SAN depends on the pre-move position.
4. **Mixing absolute and side-to-move orientation** between encoder and move labels. Assert it.
5. **Forgetting the legal-move mask** at inference - the net will happily emit illegal moves with high confidence.
6. **`board.copy()` in a search loop** - use `stack=False`.
7. **Not calling `engine.quit()`** - leaves orphaned Stockfish processes.
8. **The Lichess first-move offset** (§5). Worth repeating.
