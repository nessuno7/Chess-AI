"""
Play against a trained checkpoint in the terminal.

Moves are typed in SAN (Nf3, exd5, O-O, e8=Q) or UCI (g1f3). Other commands:
    undo    take back your last move (and the engine's reply)
    hint    let the engine suggest a move for you
    quit

The engine searches with its root noise off and always plays its most visited move, and keeps
its tree between moves (MonteCarloTS.advance), so it gets the opponent's reply's subtree for free.

    python play.py                                   # newest self-play best, else puzzles best
    python play.py --checkpoint ../checkpoints/puzzles_best.pt --color black --sims 400
"""

import argparse

import chess
import torch

from board import ChessGame
from monte_carlo_tree_search import MCTSConfig, MonteCarloTS
from network import CHECKPOINT_DIR, NetEvaluator, load_checkpoint


def default_checkpoint():
    for name in ("selfplay_best.pt", "puzzles_best.pt", "puzzles_latest.pt"):
        if (CHECKPOINT_DIR / name).exists():
            return CHECKPOINT_DIR / name
    raise SystemExit(f"no checkpoint found in {CHECKPOINT_DIR}, train one first (puzzles.py)")


def parse_move(board, text):
    for parse in (board.parse_san, board.parse_uci):
        try:
            return parse(text)
        except ValueError:
            pass
    return None


def describe_result(board, tree):
    value = tree._terminal_value()
    if value is None:
        return None
    if value == -1:
        return f"checkmate, {'black' if board.turn == chess.WHITE else 'white'} wins"
    return "draw"


def main():
    ap = argparse.ArgumentParser(description="play against the engine")
    ap.add_argument("--checkpoint", help="defaults to the best net in checkpoints/")
    ap.add_argument("--color", choices=("white", "black"), default="white", help="your colour")
    ap.add_argument("--sims", type=int, default=200, help="engine simulations per move")
    ap.add_argument("--fen", help="start from this position instead")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    path = args.checkpoint or default_checkpoint()
    net, _ = load_checkpoint(path, args.device)
    print(f"loaded {path}")
    game = ChessGame(args.fen, stored_timesteps=net.stored_timesteps)
    config = MCTSConfig(num_simulations=args.sims, add_root_noise=False, temperature_moves=0)
    tree = MonteCarloTS(game, NetEvaluator(net, args.device), config)
    human = chess.WHITE if args.color == "white" else chess.BLACK

    while True:
        print()
        print(game.board.unicode(invert_color=True, borders=True, orientation=human))
        result = describe_result(game.board, tree)
        if result:
            print(result)
            return

        if game.board.turn != human:
            root = tree.search()
            move = tree.select_move(move_number=0)
            k = root.moves.index(move)
            print(f"engine plays {game.board.san(move)}  "
                  f"(visits {root.N[k]}/{root.N.sum()}, eval {root.q()[k]:+.2f} for the engine)")
            tree.advance(move)
            continue

        text = input("your move: ").strip()
        if text == "quit":
            return
        if text == "undo":
            if len(game.board.move_stack) >= 2:
                game.undo_move()
                game.undo_move()
                tree.reset()
            continue
        if text == "hint":
            tree.search()
            print(f"hint: {game.board.san(tree.select_move(move_number=0))}")
            continue
        move = parse_move(game.board, text)
        if move is None or move not in game.board.legal_moves:
            print("not a legal move")
            continue
        tree.advance(move)


if __name__ == "__main__":
    main()
