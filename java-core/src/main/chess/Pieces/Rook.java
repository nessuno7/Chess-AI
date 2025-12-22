package main.chess.Pieces;

import main.chess.Coordinate;
import main.chess.Moves.Move;

import java.util.TreeSet;

public class Rook extends Piece {
    private boolean hasMovedAlready;
    private final boolean isLeftRookVal;

    public Rook(Coordinate coordinate, boolean white, boolean left) {
        super(coordinate, white);
        pieceType = PieceType.ROOK;
        this.hasMovedAlready = false;
        this.isLeftRookVal = left;
    }

    public Rook(Rook copy) {
        super(copy);
        pieceType = PieceType.ROOK;
        this.hasMovedAlready = copy.getHasMovedAlready();
        this.isLeftRookVal = copy.isLeftRookFun();
    }

    @Override
    public Piece copy() {
        return new Rook(this);
    }

    public boolean isLeftRookFun() {
        return isLeftRookVal;
    }

    public boolean getHasMovedAlready() {
        return this.hasMovedAlready;
    }

    @Override
    public void movePiece(Move move) {
        this.coordinate.setCoordinate(move.getFinalCoordinate());
        this.hasMovedAlready = true;
    }

    @Override
    public TreeSet<Coordinate> returnScope(ChessBoard currentGameState) {
        TreeSet<Coordinate> threatenedCoordinates = new TreeSet<>();
        for (int i = 0; i < 8; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow() + i, this.coordinate.getColumn()
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        for (int i = 0; i < 8; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow(), this.coordinate.getColumn() + i
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        for (int i = 0; i < 8; i++) {
            Coordinate newCoord = new Coordinate(
                    this.coordinate.getRow(), this.coordinate.getColumn() - i
            );
            if (!standardLoopCondition(newCoord, currentGameState, threatenedCoordinates)) {
                break;
            }
        }
        for (int i = 0; i < 8; i++) {
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
