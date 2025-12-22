package main.chess.Moves;

import main.chess.Coordinate;
import main.chess.Pieces.Piece;

public class Promotion extends Move {
    private Piece.PieceType newPieceType;

    public Promotion(
            Coordinate initCoordinate, Coordinate finalCoordinate, Piece.PieceType pieceType,
            boolean white, Piece.PieceType newType
    ) {
        super(initCoordinate, finalCoordinate, pieceType, white);
        this.newPieceType = newType;
    }

    public Promotion(Promotion copy) {
        super(copy);
        this.newPieceType = copy.getNewPieceType();
    }

    public void setNewPieceType(Piece.PieceType newPieceType) {
        this.newPieceType = newPieceType;
    }

    @Override
    public Promotion copy() {
        return new Promotion(this);
    }

    public Piece.PieceType getNewPieceType() {
        return newPieceType;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        sb.append(super.toString());
        sb.append(newPieceType.toChar());
        return sb.toString();
    }
}
