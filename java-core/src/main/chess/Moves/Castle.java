package main.chess.Moves;

import main.chess.Coordinate;
import main.chess.Pieces.Piece;

public class Castle extends Move {
    private final Coordinate initCoordinateRook;
    private final Coordinate finalCoordinateRook;

    public Castle(
            Coordinate initCoordinate, Coordinate finalCoordinate, Piece.PieceType pieceType,
            boolean white, Coordinate initCoordinateRook, Coordinate finalCoordinateRook
    ) {
        super(initCoordinate, finalCoordinate, pieceType, white);
        this.finalCoordinateRook = new Coordinate(finalCoordinateRook);
        this.initCoordinateRook = new Coordinate(initCoordinateRook);
    }

    public Castle(Castle copy) {
        super(copy);
        this.initCoordinateRook = copy.getInitCoordinateRook();
        this.finalCoordinateRook = copy.getFinalCoordinateRook();
    }

    @Override
    public Castle copy() {
        return new Castle(this);
    }

    public Coordinate getInitCoordinateRook() {
        return new Coordinate(initCoordinateRook);
    }

    public Coordinate getFinalCoordinateRook() {
        return new Coordinate(finalCoordinateRook);
    }

    @Override
    public void rotateMove() {
        intialCoordinate.rotate();
        finalCoordinate.rotate();
        initCoordinateRook.rotate();
        finalCoordinateRook.rotate();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        sb.append(initCoordinateRook.toString());
        sb.append(finalCoordinateRook.toString());
        return sb.toString();
    }
}