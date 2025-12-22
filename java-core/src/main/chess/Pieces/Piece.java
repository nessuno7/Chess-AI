package main.chess.Pieces;

import main.chess.Coordinate;
import main.chess.Moves.Move;

import java.util.LinkedList;
import java.util.TreeSet;

public abstract class Piece {
    protected Coordinate coordinate;
    protected final boolean white;
    protected PieceType pieceType; // not final because pawn can promote

    public enum PieceType {
        KING(0),
        QUEEN(1),
        ROOK(2),
        BISHOP(3),
        KNIGHT(4),
        PAWN(5);

        private final int pieceId;

        PieceType(int pieceId) {
            this.pieceId = pieceId;
        }

        public int getPieceId() {
            return pieceId;
        }

        public char toChar(){
            switch (this){
                case KING -> {
                    return 'K';
                }
                case QUEEN -> {
                    return 'Q';
                }
                case ROOK -> {
                    return 'R';
                }
                case BISHOP -> {
                    return 'B';
                }
                case KNIGHT -> {
                    return 'N';
                }
                case PAWN -> {
                    return 'P';
                }
                default -> {
                    throw new IllegalArgumentException("Invalid piece type");
                }
            }
        }
    }

    public Piece(Coordinate coordinate, boolean white) {
        this.coordinate = new Coordinate(coordinate);
        this.white = white;
    }

    public Piece(Piece copy) {
        this.coordinate = new Coordinate(copy.getPosition());
        this.white = copy.isWhite();
    }

    public abstract Piece copy();

    public void movePiece(Move move) { // assumes move is always legal
        this.coordinate.setCoordinate(move.getFinalCoordinate());
    }

    public void movePiece(Coordinate finalCoordinate) {
        this.coordinate.setCoordinate(finalCoordinate);
    }

    public Coordinate getPosition() {
        return new Coordinate(this.coordinate);
    }

    public boolean isWhite() {
        return white;
    }

    public PieceType getPieceType() {
        return pieceType;
    }

    public abstract TreeSet<Coordinate> returnScope(ChessBoard currentGameState);

    public LinkedList<Move> getAllLegalMoves(ChessBoard board) {
        LinkedList<Move> legalMoves = new LinkedList<>();
        TreeSet<Coordinate> scopeOfPieces = this.returnScope(board);

        for (Coordinate coord : scopeOfPieces) {
            Move newMove = new Move(this.coordinate, coord, this.pieceType, this.white);
            ChessBoard simulatedBoard = new ChessBoard(board);
            if (simulatedBoard.simulateMove(newMove)) {
                legalMoves.add(newMove);
            }
        }
        return legalMoves;
    }

    protected boolean standardLoopCondition(
            Coordinate newCoord, ChessBoard currentGameState,
            TreeSet<Coordinate> threatenedCoordinates
    ) {
        if (newCoord.isValid()) {
            if (!newCoord.equals(this.coordinate)) {
                if (currentGameState.isTherePieceAtCoordinate(newCoord)) {
                    if (currentGameState.returnColorPieceAtCoordinate(newCoord) != this.white) {
                        threatenedCoordinates.add(newCoord);
                    }
                    return false;
                } else {
                    threatenedCoordinates.add(newCoord);
                    return true;
                }
            }
            return true;
        } else {
            return false;
        }
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
        sb.append(coordinate.toString());

        return sb.toString();
    }

}
