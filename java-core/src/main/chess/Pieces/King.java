package main.chess.Pieces;

import main.chess.Coordinate;
import main.chess.Moves.Castle;
import main.chess.Moves.Move;

import java.util.LinkedList;
import java.util.TreeSet;

public class King extends Piece {
    private boolean isLongCastleAllowed;
    private boolean isShortCastleAllowed;

    public King(Coordinate coordinate, boolean white) {
        super(coordinate, white);
        pieceType = PieceType.KING;
        isLongCastleAllowed = true;
        isShortCastleAllowed = true;
    }

    public King(King copy) {
        super(copy);
        pieceType = PieceType.KING;
        isShortCastleAllowed = copy.getIsShortCastleAllowedVar();
        isLongCastleAllowed = copy.getIsLongCastleAllowedVar();
    }

    @Override
    public Piece copy() {
        return new King(this);
    }

    @Override
    public void movePiece(Move move) {
        this.coordinate.setCoordinate(move.getFinalCoordinate());
        this.isLongCastleAllowed = false;
        this.isShortCastleAllowed = false;
    }

    public boolean getIsLongCastleAllowedVar() {
        return isLongCastleAllowed;
    }

    public boolean getIsShortCastleAllowedVar() {
        return isShortCastleAllowed;
    }

    @Override
    public TreeSet<Coordinate> returnScope(ChessBoard currentGameState) {
        TreeSet<Coordinate> threatenedCoordinates = new TreeSet<>();
        for (int i = -1; i < 2; i++) {
            for (int j = -1; j < 2; j++) {
                Coordinate newCoord = new Coordinate(
                        this.coordinate.getRow() + i, this.coordinate.getColumn() + j
                );
                if (newCoord.isValid() && !newCoord.equals(this.coordinate)) {
                    if (currentGameState.isTherePieceAtCoordinate(newCoord)) {
                        if (currentGameState.returnColorPieceAtCoordinate(newCoord) != this.white) {
                            threatenedCoordinates.add(newCoord);
                        }
                    } else {
                        threatenedCoordinates.add(newCoord);
                    }
                }
            }
        }
        return threatenedCoordinates;
    }

    public boolean isInCheck(ChessBoard currentGameState) {
        TreeSet<Coordinate> scopeOppPieces = currentGameState.getAllThreatendCoords(!this.white);
        for (Coordinate coordinate : scopeOppPieces) {
            if (coordinate.equals(this.coordinate)) {
                return true;
            }
        }
        return false;
    }

    private boolean getIsLongCastleAllowed(ChessBoard currentGameState) {
        TreeSet<Coordinate> scopeOppPieces = currentGameState.getAllThreatendCoords(!this.white);
        if (this.isLongCastleAllowed) {
            if (!this.isInCheck(currentGameState)) {
                for (Piece piece : currentGameState.getPiecesOnBoard()) {
                    if (piece.getPieceType() == PieceType.ROOK && (piece.isWhite() == this.white)) {
                        Rook rookPiece = (Rook) piece;
                        if (rookPiece.isLeftRookFun()) {
                            if (!rookPiece.getHasMovedAlready()) {
                                // check that the path is not obstructed
                                Coordinate[] coordList;
                                if (!piece.isWhite()) {
                                    coordList = new Coordinate[] { new Coordinate(0, 4),
                                            new Coordinate(0, 5), new Coordinate(0, 6) };
                                } else {
                                    coordList = new Coordinate[] { new Coordinate(0, 1),
                                            new Coordinate(0, 2), new Coordinate(0, 3) };
                                }

                                for (Coordinate coord : coordList) {
                                    if (currentGameState.isTherePieceAtCoordinate(coord)
                                            || scopeOppPieces.contains(coord)) {
                                        return false;
                                    }
                                }
                                return !rookPiece.getHasMovedAlready();
                            } else {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }

    private boolean getIsShortCastleAllowed(ChessBoard currentGameState) {
        TreeSet<Coordinate> scopeOppPieces = currentGameState.getAllThreatendCoords(!this.white);
        if (this.isShortCastleAllowed) {
            if (!this.isInCheck(currentGameState)) {
                for (Piece piece : currentGameState.getPiecesOnBoard()) {
                    if (piece.getPieceType() == PieceType.ROOK && (piece.isWhite() == this.white)) {
                        if (!((Rook)piece).isLeftRookFun()) {
                            if (!((Rook)piece).getHasMovedAlready()) {
                                // check that the path is not obstructed
                                Coordinate[] coordList;
                                if (!piece.isWhite()) {
                                    coordList = new Coordinate[] { new Coordinate(0, 1),
                                            new Coordinate(0, 2) };
                                } else {
                                    coordList = new Coordinate[] { new Coordinate(0, 5),
                                            new Coordinate(0, 6) };
                                }

                                for (Coordinate coord : coordList) {
                                    if (currentGameState.isTherePieceAtCoordinate(coord)
                                            || scopeOppPieces.contains(coord)) {
                                        return false;
                                    }
                                }
                                return !((Rook)piece).getHasMovedAlready();
                            } else {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        return false;
    }

    @Override
    public LinkedList<Move> getAllLegalMoves(ChessBoard board) {
        LinkedList<Move> legalMoves = new LinkedList<>();
        TreeSet<Coordinate> scopeOfPiece = this.returnScope(board);

        for (Coordinate coord : scopeOfPiece) {
            Move newMove = new Move(this.coordinate, coord, this.pieceType, this.white);
            ChessBoard simulatedBoard = new ChessBoard(board);
            if (simulatedBoard.simulateMove(newMove)) {
                legalMoves.add(newMove);
            }
        }
        if (this.getIsShortCastleAllowed(board)) {
            Move newMove;
            if (this.isWhite()) {
                newMove = new Castle(
                        this.coordinate, new Coordinate(0, 6), this.pieceType, this.white,
                        new Coordinate(0, 7), new Coordinate(0, 5)
                );
            } else {
                newMove = new Castle(
                        this.coordinate, new Coordinate(0, 1), this.pieceType, this.white,
                        new Coordinate(0, 0), new Coordinate(0, 2)
                );
            }
            legalMoves.add(newMove);
        }
        if (this.getIsLongCastleAllowed(board)) {
            Move newMove;
            if (this.isWhite()) {
                newMove = new Castle(
                        this.coordinate, new Coordinate(0, 2), this.pieceType, this.white,
                        new Coordinate(0, 0), new Coordinate(0, 3)
                );

            } else {
                newMove = new Castle(
                        this.coordinate, new Coordinate(0, 5), this.pieceType, this.white,
                        new Coordinate(0, 7), new Coordinate(0, 4)
                );
            }
            legalMoves.add(newMove);
        }
        return legalMoves;
    }

}
