package main.chess.Moves;

import main.chess.Coordinate;
import main.chess.Pieces.Piece;

public class EnPassant extends Move {
    private Coordinate pawnTakenCoordinate;

    public EnPassant(
            Coordinate initCoordinate, Coordinate finalCoordinate, Piece.PieceType pieceType,
            boolean white, Coordinate pawnTakenCoordinate
    ) {
        super(initCoordinate, finalCoordinate, pieceType, white);
        this.pawnTakenCoordinate = new Coordinate(pawnTakenCoordinate);
    }

    public EnPassant(EnPassant copy) {
        super(copy);
        this.pawnTakenCoordinate = copy.getPawnTakenCoordinate();
    }

    @Override
    public EnPassant copy() {
        return new EnPassant(this);
    }

    public Coordinate getPawnTakenCoordinate() {
        return new Coordinate(this.pawnTakenCoordinate);
    }

    @Override
    public void rotateMove() {
        intialCoordinate.rotate();
        finalCoordinate.rotate();
        pawnTakenCoordinate.rotate();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        sb.append(pawnTakenCoordinate.toString());
        return sb.toString();
    }
}
