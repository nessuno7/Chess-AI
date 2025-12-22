package main.chess.Pieces;

import main.chess.Coordinate;
import main.chess.Moves.EnPassant;
import main.chess.Moves.Move;
import main.chess.Moves.Promotion;

import java.util.LinkedList;
import java.util.NoSuchElementException;
import java.util.TreeSet;

public class Pawn extends Piece {

    public Pawn(Coordinate coordinate, boolean white) {
        super(coordinate, white);
        pieceType = PieceType.PAWN;
        // hasMoveAlready = false;
    }

    public Pawn(Pawn copy) {
        super(copy);
        pieceType = PieceType.PAWN;
    }

    @Override
    public Piece copy() {
        return new Pawn(this);
    }

    private boolean pawnTakeCondition(Coordinate newCoord, ChessBoard currentGameState) {
        if (newCoord.isValid()) {
            if (!newCoord.equals(this.coordinate)) {
                if (currentGameState.isTherePieceAtCoordinate(newCoord)) {
                    return currentGameState.returnColorPieceAtCoordinate(newCoord) != this.white;
                }
            }
        }
        return false;
    }

    @Override
    public TreeSet<Coordinate> returnScope(ChessBoard currentGameState) {
        TreeSet<Coordinate> threatenedCoordinates = new TreeSet<>();

        Coordinate newCoord = new Coordinate(0, 0);
        if(newCoord.isValid()) {
            if (this.white) {
                newCoord.setRow(this.coordinate.getRow() + 1);
            } else {
                newCoord.setRow(this.coordinate.getRow() - 1);
            }
            newCoord.setColumn(this.coordinate.getColumn() + 1);
            if (this.pawnTakeCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        if(newCoord.isValid()) {
            if (this.white) {
                newCoord.setRow(this.coordinate.getRow() + 1);
            } else {
                newCoord.setRow(this.coordinate.getRow() - 1);
            }
            newCoord.setColumn(this.coordinate.getColumn() - 1);
            if (this.pawnTakeCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        return threatenedCoordinates;
    }

    private boolean isEnPassantAllowed(ChessBoard currentGameState, boolean right) {
        Coordinate newCoord;
        if (right) {
            newCoord = new Coordinate(
                    this.coordinate.getRow() + 1, this.coordinate.getColumn() + 1
            );
        } else {
            newCoord = new Coordinate(
                    this.coordinate.getRow() + 1, this.coordinate.getColumn() - 1
            );
        }
        if(newCoord.isValid()) {
            if (!currentGameState.isTherePieceAtCoordinate(newCoord)) {
                try{
                    if (currentGameState.lastMove().getPieceType() == PieceType.PAWN
                            && (currentGameState.lastMove().isWhite() != this.white)) {
                        int initRow = currentGameState.lastMove().getIntialCoordinate().getRow();
                        int finalRow = currentGameState.lastMove().getFinalCoordinate().getRow();
                        newCoord.setRow(this.coordinate.getRow());
                        if ((finalRow - initRow) == -2) {
                            return currentGameState.lastMove().getFinalCoordinate().equals(newCoord);
                        }
                    }
                } catch (NoSuchElementException e){}
            }
        }

        return false;
    }

    public TreeSet<Coordinate> legalMovesHelper(ChessBoard currentGameState) {
        TreeSet<Coordinate> threatenedCoordinates = new TreeSet<>();

        Coordinate newCoord = new Coordinate(0, 0);
        newCoord.setRow(this.coordinate.getRow() + 1);
        newCoord.setColumn(this.coordinate.getColumn() + 1);
        if(newCoord.isValid()) {
            if (this.pawnTakeCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        newCoord.setRow(this.coordinate.getRow() + 1);
        newCoord.setColumn(this.coordinate.getColumn() - 1);
        if(newCoord.isValid()) {
            if (this.pawnTakeCondition(newCoord, currentGameState)) {
                threatenedCoordinates.add(new Coordinate(newCoord));
            }
        }

        return threatenedCoordinates;
    }

    @Override
    public LinkedList<Move> getAllLegalMoves(ChessBoard board) {
        LinkedList<Move> legalMoves = new LinkedList<>();
        TreeSet<Coordinate> scopeOfPieces = this.legalMovesHelper(board);
        for (Coordinate coord : scopeOfPieces) {
            Move newMove;
            if (coord.getRow() == 7) {
                newMove = new Promotion(
                        this.coordinate, coord, this.pieceType, this.white, PieceType.QUEEN
                );
            } else {
                newMove = new Move(this.coordinate, coord, this.pieceType, this.white);
            }

            ChessBoard simulatedBoard = new ChessBoard(board);
            if (simulatedBoard.simulateMove(newMove)) {
                legalMoves.add(newMove);
            }
        }

        Coordinate newCoord = new Coordinate(this.coordinate.getRow(), this.coordinate.getColumn());
        newCoord.setColumn(this.coordinate.getColumn());
        newCoord.setRow(this.coordinate.getRow() + 1);

        if(newCoord.isValid()) {
            if (newCoord.isValid() && !board.isTherePieceAtCoordinate(newCoord)) {
                Move newMove;
                if (newCoord.getRow() == 7) {
                    newMove = new Promotion(
                            this.coordinate, newCoord, this.pieceType, this.white, PieceType.QUEEN
                    );
                } else {
                    newMove = new Move(this.coordinate, newCoord, this.pieceType, this.white);
                }
                ChessBoard simulatedBoard = new ChessBoard(board);
                if (simulatedBoard.simulateMove(newMove)) {
                    legalMoves.add(newMove);
                }
            }
        }

        if (this.getPosition().getRow() == 1) { // this for the double pawn start
            newCoord.setColumn(this.coordinate.getColumn());
            newCoord.setRow(this.coordinate.getRow() + 2);
            if (newCoord.isValid() && !board.isTherePieceAtCoordinate(newCoord)) {
                Move newMove = new Move(this.coordinate, newCoord, this.pieceType, this.white);
                ChessBoard simulatedBoard = new ChessBoard(board);
                if (simulatedBoard.simulateMove(newMove)) {
                    legalMoves.add(newMove);
                }
            }
        }

        // this is for the enpassant
        if (isEnPassantAllowed(board, true)) {
            EnPassant newMove = new EnPassant(
                    this.coordinate,
                    new Coordinate(this.coordinate.getRow() + 1, this.coordinate.getColumn() + 1),
                    this.pieceType, this.white,
                    new Coordinate(this.coordinate.getRow(), this.coordinate.getColumn() + 1)
            );
            ChessBoard simulatedBoard = new ChessBoard(board);
            if (simulatedBoard.simulateMove(newMove)) {
                legalMoves.add(newMove);
            }
        }
        if (isEnPassantAllowed(board, false)) {
            EnPassant newMove = new EnPassant(
                    this.coordinate,
                    new Coordinate(this.coordinate.getRow() + 1, this.coordinate.getColumn() - 1),
                    this.pieceType, this.white,
                    new Coordinate(this.coordinate.getRow(), this.coordinate.getColumn() - 1)
            );
            ChessBoard simulatedBoard = new ChessBoard(board);
            if (simulatedBoard.simulateMove(newMove)) {
                legalMoves.add(newMove);
            }
        }

        return legalMoves;
    }
}
