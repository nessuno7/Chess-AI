import pandas as pd
from sympy.multipledispatch.dispatcher import RaiseNotImplementedError

CSV_PATH = "lichess_db_puzzle.csv"

class ProblemIterator:
    def __init__(self, skip):
        self.file_iter = pd.read_csv(CSV_PATH, chunksize=50_000, skiprows=skip)
        self.chunk = None
        self.row_iter = None
        pass

    def __iter__(self):
        return self

    def __next__(self): #if iterator return None twice the iterator is stopped
        count = 0
        while count < 2:
            if self.row_iter is None:
                self.chunk = next(self.file_iter)
                self.chunk = self.chunk[["FEN", "Moves", "Rating"]]
                self.row_iter = self.chunk.iterrows()

            try:
                i, row = next(self.row_iter)
                fen = row["FEN"]
                moves_list = row["Moves"].split()
                rating = int(row["Rating"])
                return i, fen, moves_list, rating
            except StopIteration:
                self.row_iter = None
                count = count + 1

        raise StopIteration()

def moveToInt(move):
    int1 = ord(move[0]) - ord('a')
    int2 = 7 - int(move[2])
    square_from = 8*int2 + int1
    int1 = ord(move[2]) - ord('a')
    int2 = 7 - int(move[3])
    square_to = 8*int2 + int1
    if len(move) >= 5:
        if move[4] == "q":
            promotion = 1
        elif move[4] == "r":
            promotion = 2
        elif move[4] == "b":
            promotion = 3
        elif move[4] == "n":
            promotion = 4
        else:
            raise IndexError()
    else:
        promotion = 0
    return square_from, square_to, promotion

def intToCoord(coord):
    string = []
    string.append(chr(coord%8 + ord('a')))
    string.append(str(int((coord - (coord%8))/8)))
    return ''.join(string)

def intToMove(square_from, square_to, promotion):
    string = []
    string.append(intToCoord(square_from))
    string.append(intToCoord(square_to))
    if promotion != 0:
        if promotion == 1:
            string.append("q")
        elif promotion == 2:
            string.append("r")
        elif promotion == 3:
            string.append("b")
        elif promotion == 4:
            string.append("n")
    return ''.join(string)

def formatMovesList(moves):
    moves_list = []
    for move in moves:
        moves_list.append(moveToInt(move))
    return moves_list

if __name__ == "__main__":
    iterator = ProblemIterator()
    i, fen, moves, rating = next(iterator)

    for move in moves:
        print(move)

    print(i, fen, rating)
