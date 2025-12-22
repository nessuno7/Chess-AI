package main.chess.Pieces;

import main.chess.Coordinate;

import java.util.TreeSet;

public class Knight extends Piece {
    public Knight(Coordinate coordinate, boolean white) {
        super(coordinate, white);
        pieceType = PieceType.KNIGHT;
    }

    public Knight(Knight copy) {
        super(copy);
        pieceType = PieceType.KNIGHT;
    }

    @Override
    public Piece copy() {
        return new Knight(this);
    }

    private boolean knightMoveCondition(Coordinate newCoord, ChessBoard currentGameState) {
        if (newCoord.isValid()) {
            if (!newCoord.equals(this.coordinate)) {
                if (currentGameState.isTherePieceAtCoordinate(newCoord)) {
                    if (currentGameState.returnColorPieceAtCoordinate(newCoord) != this.white) {
                        return true;
                    }
                } else {
                    return true;
                }
            }
        }
        return false;
    }

    @Override
    public TreeSet<Coordinate> returnScope(ChessBoard currentGameState) {
        TreeSet<Coordinate> threatenedCoordinates = new TreeSet<>();
        Coordinate newCoord = new Coordinate(this.coordinate);

        newCoord.setRow(this.coordinate.getRow() + 2);
        newCoord.setColumn(this.coordinate.getColumn() + 1);
        if(newCoord.isValid()) {
            if (knightMoveCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        newCoord.setRow(this.coordinate.getRow() + 2);
        newCoord.setColumn(this.coordinate.getColumn() - 1);
        if(newCoord.isValid()) {
            if (knightMoveCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        newCoord.setRow(this.coordinate.getRow() - 2);
        newCoord.setColumn(this.coordinate.getColumn() + 1);
        if(newCoord.isValid()) {
            if (knightMoveCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        newCoord.setRow(this.coordinate.getRow() - 2);
        newCoord.setColumn(this.coordinate.getColumn() - 1);
        if(newCoord.isValid()) {
            if (knightMoveCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        newCoord.setRow(this.coordinate.getRow() + 1);
        newCoord.setColumn(this.coordinate.getColumn() + 2);
        if(newCoord.isValid()) {
            if (knightMoveCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        newCoord.setRow(this.coordinate.getRow() - 1);
        newCoord.setColumn(this.coordinate.getColumn() + 2);
        if(newCoord.isValid()) {
            if (knightMoveCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        newCoord.setRow(this.coordinate.getRow() + 1);
        newCoord.setColumn(this.coordinate.getColumn() - 2);
        if(newCoord.isValid()) {
            if (knightMoveCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        newCoord.setRow(this.coordinate.getRow() - 1);
        newCoord.setColumn(this.coordinate.getColumn() - 2);
        if(newCoord.isValid()) {
            if (knightMoveCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        return threatenedCoordinates;
    }
}
