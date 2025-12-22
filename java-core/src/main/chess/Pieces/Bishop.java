package main.chess.Pieces;

import main.chess.Coordinate;

import java.util.TreeSet;

public class Bishop extends Piece {
    public Bishop(Coordinate coordinate, boolean white) {
        super(coordinate, white);
        pieceType = PieceType.BISHOP;
    }

    public Bishop(Bishop copy) {
        super(copy);
        pieceType = PieceType.BISHOP;
    }

    @Override
    public Piece copy() {
        return new Bishop(this);
    }

    @Override
    public TreeSet<Coordinate> returnScope(ChessBoard currentGameState) {
        TreeSet<Coordinate> threatenedCoordinates = new TreeSet<>();
        for (int i = 0; i < 9; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow() + i, this.coordinate.getColumn() + i
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        for (int i = 0; i < 9; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow() + i, this.coordinate.getColumn() - i
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        for (int i = 0; i < 9; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow() - i, this.coordinate.getColumn() + i
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        for (int i = 0; i < 9; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow() - i, this.coordinate.getColumn() - i
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        return threatenedCoordinates;
    }

}
