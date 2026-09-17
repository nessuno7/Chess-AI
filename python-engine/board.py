from collections import deque

import chess
import numpy as np

N_GLOBALS = 7   # planes 0-6: side to move, 4 castling rights, en passant, halfmove clock
N_FRAME = 14    # per position: 12 piece planes + 2 repetition planes

# reorders a white-view frame into a black-view frame: black pieces become "our" planes 0-5,
# white pieces become "their" planes 6-11, repetition planes 12-13 stay where they are
BLACK_VIEW_PERM = np.r_[6:12, 0:6, 12:14]

KNIGHT_LOOKUP = {(2,1):1, (-2,1):2, (2, -1):3, (-2, -1):4, (1,2):5, (1, -2):6, (-1,2):7, (-1,-2):8}
QUEEN_LOOKUP = {(0,1):1, (1,1,):2, (-1,1):3, (1,0):5, (-1,0):8, (-1,-1):4, (0,-1):6, (1,-1):7, }


class MoveNotLegalExcpetion(Exception):
    """
    Exception raise when a not legal move is played on the ChessGame class
    """ 

class ChessGame:
    def __init__(self, fen: str | None = None, stored_timesteps: int = 1): # default fen to None
        if fen is None:
            self.board = chess.Board()
        else:
            self.board = chess.Board(fen)
        self.stored_timesteps = stored_timesteps
        self.positions_map: dict[int, int] = {}

        # newest first: history[f] is the position f plies ago.
        # maxlen makes appendleft drop the oldest frame for us
        # each entry is a (white_view, black_view) pair, see _position_frame for why both are stored
        self.history: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=self.stored_timesteps)

        # hashing the intial board first position
        key = self.board._transposition_key()
        self.positions_map[key] = self.positions_map.get(key, 0) + 1
        self.history.appendleft(self._position_frame())

    def legal_moves_list(self):
        r_ = []

        for move in self.board.legal_moves:
            r_.append(move)

        return r_

    def play_move(self, move):
        if move in self.board.legal_moves:
            self.board.push(move)
            key = self.board._transposition_key()
            self.positions_map[key] = self.positions_map.get(key, 0) + 1 # updating the ferqeuncy table of baord occurences
            self.history.appendleft(self._position_frame()) # the new position becomes frame 0, the rest shift down
        else:
            from_sq    = chess.square_name(move.from_square)   # 'e2'
            to_sq      = chess.square_name(move.to_square)     # 'e4'    

            raise MoveNotLegalExcpetion(f"Illegal move: no piece on {from_sq} ({from_sq} -> {to_sq})")

    def canonical_move(self, move, turn):
        piece = self.board.piece_type_at(move.from_square)

        if turn == chess.WHITE:
            return piece, move
        return piece, chess.Move(
            chess.square_mirror(move.from_square),
            chess.square_mirror(move.to_square),
            promotion=move.promotion,   # unchanged
            drop=move.drop,
        )

    def hash_move(self, move):
        turn = self.board.turn

        piece, move = self.canonical_move(move, turn)
        first_part = 73*move.from_square

        second_part = 0
        """
        order of moves is:
        N: 1
        S: 2
        E: 3
        W: 4
        NE: 5
        SE: 6
        NW: 7
        SW: 8   (8 possible directions) + distance 


        r8  56 57 58 59 60 61 62 63
        r7  48 49 50 51 52 53 54 55
        r6  40 41 42 43 44 45 46 47
        r5  32 33 34 35 36 37 38 39
        r4  24 25 26 27 28 29 30 31
        r3  16 17 18 19 20 21 22 23
        r2   8  9 10 11 12 13 14 15
        r1   0  1  2  3  4  5  6  7
        a  b  c  d  e  f  g  h


        from 0-55: we have normal queen moves
        from 56-63: knight moves
        from 64-72: underpomotions
        """

        from_ = move.from_square
        file_from = chess.square_file(from_)
        rank_from = chess.square_rank(from_)

        to_ = move.to_square
        file_to = chess.square_file(to_)
        rank_to = chess.square_rank(to_)

        if piece == chess.KNIGHT:
            x = file_to - file_from
            y = rank_to - rank_from

            second_part = 55+KNIGHT_LOOKUP[(x,y)] # ensure all result are whithin 55+1 and 55+8 : 56-63

        else:
            if rank_to>rank_from: 
                y = 1
            elif rank_to<rank_from:
                y = -1 
            else:
                y=0

            if file_to>file_from:
                x =1
            elif file_to<file_from:
                x=-1
            else:
                x=0

            if (x,y) == (0,0):
                raise MoveNotLegalExcpetion("Move cannot move from cell to the same cell")

            direction = QUEEN_LOOKUP[(x,y)]

            if move.promotion and move.promotion < 5: # if the promotion is not a queen
                extra = 3*(move.promotion-2) + direction
                second_part = 63+extra # ensure all result are whithin 63+1 and 63+9: 64 and 72
            else:
                distance = max(abs(file_from- file_to), abs(rank_from-rank_to))
                second_part = (distance-1)+(direction-1)*7 # all result whithin 0 to 7*7+6: 0 to 55  

        return first_part+second_part

    

    def create_legal_moves_mask(self):

        mask = np.zeros((4672,), dtype = np.float32)

        for move in self.board.legal_moves:
            hash = self.hash_move(move)
            mask[hash] = 1

        return mask

    def _position_frame(self):
        """
        The 14 planes that describe the current position on its own:
        0-5 our pieces P,N,B,R,Q,K; 6-11 their pieces; 12-13 the repetition planes.
        Built once per position, when the position is reached, and then kept in self.history.
        Returns (white_view, black_view): the same frame seen by white and by black.
        """
        f = np.zeros((N_FRAME, 8, 8), dtype=np.float32)

        # planes 0-11; pieces positions, built from white's point of view
        for sq, piece in self.board.piece_map().items():
            plane = (piece.piece_type - 1) + (0 if piece.color == chess.WHITE else 6)
            f[plane, chess.square_rank(sq), chess.square_file(sq)] = 1.0

        # planes 12-13
        # plane 12 is 1 if one repetiton has occured, plane 13 is 1 if the second rep has occured, else both planes to 0
        # positions_map counts the current position too, so the first visit is repeats == 1
        key = self.board._transposition_key()
        repeats = self.positions_map.get(key, 0)
        f[12] = float(repeats >= 2)
        f[13] = float(repeats >= 3)

        # every frame in the tensor has to be seen from the CURRENT player's side, and the player to move
        # alternates every ply, so the same stored frame is needed in both views over its lifetime.
        # we store the black view once here (colours swapped + ranks flipped, files kept since chess is
        # not left-right symmetric) instead of re-flipping every frame on every encode() call:
        # ~3.5KB extra per frame, and encode() becomes a plain copy
        black_view = f[BLACK_VIEW_PERM, ::-1, :].copy()

        return f, black_view

    def encode(self):
        """
        The way the board will be encoded is by defining planes.
        Pure read: the globals come from the current board, the frames from self.history,
        so calling it twice on the same position gives the same tensor.
        """

        x = np.zeros((N_FRAME*self.stored_timesteps + N_GLOBALS, 8, 8), dtype=np.float32)

        us, them = self.board.turn, not self.board.turn

        # i: 0; player to move, the only absolute plane (white has a real first move advantage)
        x[0] = float(us == chess.WHITE)

        # i: 1-4
        # castling rights, relative to the player to move: our K, our Q, their K, their Q
        rights = [
            self.board.has_kingside_castling_rights(us),
            self.board.has_queenside_castling_rights(us),
            self.board.has_kingside_castling_rights(them),
            self.board.has_queenside_castling_rights(them),
        ]
        for i, right in enumerate(rights):
            x[1 + i] = float(right)

        # i: 5; enpassant, rank flipped when black is to move so it matches the frames
        if self.board.ep_square is not None:
            ep_sq = self.board.ep_square if us == chess.WHITE else chess.square_mirror(self.board.ep_square)
            x[5, chess.square_rank(ep_sq), chess.square_file(ep_sq)] = 1.0

        # i: 6; halfmove counter
        x[6] = self.board.halfmove_clock / 100.0

        # planes i: 7 onwards; one 14-plane frame per stored timestep, newest first.
        # frames we do not have yet (early in the game) stay zero-padded
        # all frames use the current player's view, picked from the pair stored in history
        view = 0 if us == chess.WHITE else 1
        for i, frames in enumerate(self.history):
            base = N_GLOBALS + N_FRAME*i
            x[base:base + N_FRAME] = frames[view]

        return x


    