package main.chess.Moves;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import main.chess.Coordinate;
import main.chess.Pieces.Piece;

public class Move {
    protected final Coordinate intialCoordinate;
    protected final Coordinate finalCoordinate;
    private final Piece.PieceType pieceType;
    private final boolean white;

    public Move(
            Coordinate initCoordinate, Coordinate finalCoordinate, Piece.PieceType pieceType,
            boolean white
    ) {
        this.intialCoordinate = new Coordinate(initCoordinate);
        this.finalCoordinate = new Coordinate(finalCoordinate);
        this.pieceType = pieceType;
        this.white = white;
    }

    public Move(Move copy) {
        this.intialCoordinate = new Coordinate(copy.getIntialCoordinate());
        this.finalCoordinate = new Coordinate(copy.getFinalCoordinate());
        this.pieceType = copy.getPieceType();
        this.white = copy.isWhite();
    }

    public Move copy() {
        return new Move(this);
    }

    public Coordinate getIntialCoordinate() {
        return new Coordinate(this.intialCoordinate);
    }

    public Coordinate getFinalCoordinate() {
        return new Coordinate(this.finalCoordinate);
    }

    public Piece.PieceType getPieceType() {
        return pieceType;
    }

    public boolean isWhite() {
        return white;
    }

    public void rotateMove() {
        intialCoordinate.rotate();
        finalCoordinate.rotate();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Move move = (Move) o;
        return intialCoordinate.equals(move.getIntialCoordinate())
                && finalCoordinate.equals(move.getFinalCoordinate());
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        if(white){
            sb.append('W');
        }
        else{
            sb.append('B');
        }
        sb.append(pieceType.toChar());

        sb.append(intialCoordinate.toString());
        sb.append(finalCoordinate.toString());
        return sb.toString();
    }
}
