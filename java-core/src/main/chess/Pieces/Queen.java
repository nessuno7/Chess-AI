package main.chess.Pieces;

import main.chess.Coordinate;

import java.util.TreeSet;

public class Queen extends Piece {
    public Queen(Coordinate coordinate, boolean white) {
        super(coordinate, white);
        pieceType = PieceType.QUEEN;
    }

    public Queen(Queen copy) {
        super(copy);
        pieceType = PieceType.QUEEN;
    }

    @Override
    public Piece copy() {
        return new Queen(this);
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
        for (int i = 0; i < 9; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow() + i, this.coordinate.getColumn()
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        for (int i = 0; i < 9; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow(), this.coordinate.getColumn() + i
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        for (int i = 0; i < 9; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow(), this.coordinate.getColumn() - i
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        for (int i = 0; i < 9; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow() - i, this.coordinate.getColumn()
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        return threatenedCoordinates;
    }

}
