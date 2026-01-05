package test;

import main.chess.Coordinate;
import main.chess.Moves.Castle;
import main.chess.Moves.EnPassant;
import main.chess.Moves.Move;
import main.chess.Moves.Promotion;
import main.chess.Pieces.ChessBoard;
import main.chess.Pieces.King;
import main.chess.Pieces.Piece;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ChessBoardTest {
    ChessBoard b;

    @BeforeEach
    public void setUp() {
        b = new ChessBoard();
    }

    @Test
    public void getAllThreatenedCoordinatesInitialCondition() {
        boolean condition = true;
        for (Coordinate coord : b.getAllThreatendCoords(false)) {
            if (coord.getRow() != 5) {
                condition = false;
                break;
            }
        }
        assertTrue(condition);
    }

    @Test
    public void testUpdateStateOneMove() {
        Coordinate initCoord = new Coordinate(1, 2); // c2 pawm
        Coordinate finalCoord = new Coordinate(3, 2); // c4
        Move move1 = new Move(initCoord, finalCoord, Piece.PieceType.PAWN, true);

        b.updateState(move1);

        assertTrue(b.isTherePieceAtCoordinate(finalCoord));
        assertTrue(b.returnColorPieceAtCoordinate(finalCoord));
        assertEquals(b.returnPieceTypeAtCoordinate(finalCoord), Piece.PieceType.PAWN);
    }

    @Test
    public void testUpdateStatePawnTakesPawn() {
        Coordinate initCoord1 = new Coordinate(1, 2); // c2 pawm
        Coordinate finalCoord1 = new Coordinate(3, 2); // c4
        Move move1 = new Move(initCoord1, finalCoord1, Piece.PieceType.PAWN, true);

        b.updateState(move1);

        assertTrue(b.isTherePieceAtCoordinate(finalCoord1));
        assertTrue(b.returnColorPieceAtCoordinate(finalCoord1));
        assertEquals(Piece.PieceType.PAWN, b.returnPieceTypeAtCoordinate(finalCoord1));

        Coordinate initCoord2 = new Coordinate(6, 1); // b7 pawm
        Coordinate finalCoord2 = new Coordinate(4, 1); // b5
        Move move2 = new Move(initCoord2, finalCoord2, Piece.PieceType.PAWN, false);

        b.updateState(move2);

        assertTrue(b.isTherePieceAtCoordinate(finalCoord2));
        assertFalse(b.returnColorPieceAtCoordinate(finalCoord2));
        assertEquals(Piece.PieceType.PAWN, b.returnPieceTypeAtCoordinate(finalCoord2));

        Move move3 = new Move(finalCoord1, finalCoord2, Piece.PieceType.PAWN, true);
        b.updateState(move3);
        boolean cond = false;
        for (Piece piece : b.getPiecesTaken()) {
            if (!piece.isWhite() && piece.getPieceType() == Piece.PieceType.PAWN) {
                cond = true;
                break;
            }
        }
        assertTrue(cond);
        assertTrue(b.isTherePieceAtCoordinate(finalCoord2));
        assertTrue(b.returnColorPieceAtCoordinate(finalCoord2));

    }

    @Test
    public void testSimulateMoveAllowedLeadsOtherKingInCheck() {
        b.updateState(
                new Move(new Coordinate(1, 4), new Coordinate(3, 4), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 4), new Coordinate(4, 4), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 3), new Coordinate(2, 5), Piece.PieceType.QUEEN, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 5), new Coordinate(3, 1), Piece.PieceType.BISHOP, false)
        );

        Move moveToSimulate = new Move(
                new Coordinate(2, 5), new Coordinate(6, 5), Piece.PieceType.QUEEN, true
        );

        assertTrue(b.simulateMove(moveToSimulate));

        boolean condition = false;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPieceType() == Piece.PieceType.KING && !piece.isWhite()) {
                condition = ((King) piece).isInCheck(b);
                break;
            }
        }

        assertTrue(condition);
    }

    @Test
    public void testSimulateMoveNotAllowed() {
        b.updateState(
                new Move(new Coordinate(1, 4), new Coordinate(3, 4), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 4), new Coordinate(4, 4), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 3), new Coordinate(2, 5), Piece.PieceType.QUEEN, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 5), new Coordinate(3, 1), Piece.PieceType.BISHOP, false)
        );

        Move moveToSimulate = new Move(
                new Coordinate(1, 3), new Coordinate(4, 3), Piece.PieceType.PAWN, true
        );

        assertFalse(b.simulateMove(moveToSimulate));

    }

    @Test
    public void testCalculateAllLegalMoves() {
        Coordinate initCoord1 = new Coordinate(1, 4); // c2 pawm
        Coordinate finalCoord1 = new Coordinate(3, 4); // c4
        Move move1 = new Move(initCoord1, finalCoord1, Piece.PieceType.PAWN, true);
        b.updateState(move1);

        Coordinate initCoord2 = new Coordinate(6, 4); // b7 pawm
        Coordinate finalCoord2 = new Coordinate(4, 4); // b5
        Move move2 = new Move(initCoord2, finalCoord2, Piece.PieceType.PAWN, false);
        b.updateState(move2);

        Move move3 = new Move(
                new Coordinate(0, 3), new Coordinate(2, 5), Piece.PieceType.QUEEN, true
        );
        b.updateState(move3);

        Move move4 = new Move(
                new Coordinate(7, 1), new Coordinate(5, 2), Piece.PieceType.KNIGHT, false
        );
        b.updateState(move4);

        Move move7 = new Move(
                new Coordinate(2, 5), new Coordinate(6, 5), Piece.PieceType.QUEEN, true
        );
        b.updateState(move7);
        b.calculateAllLegalMoves();

        assertEquals(1, b.getAllLegalMoves().size());
    }

    @Test
    public void testRotateCoords() {
        b.rotateCoords();
        boolean answer = true;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPosition().getRow() >= 6) {
                answer = answer && piece.isWhite();
            }
        }

        assertTrue(answer);
    }

    @Test
    public void testCheckMateIsCheckMate() {
        Coordinate initCoord1 = new Coordinate(1, 4); // c2 pawm
        Coordinate finalCoord1 = new Coordinate(3, 4); // c4
        Move move1 = new Move(initCoord1, finalCoord1, Piece.PieceType.PAWN, true);
        b.updateState(move1);

        Coordinate initCoord2 = new Coordinate(6, 4); // b7 pawm
        Coordinate finalCoord2 = new Coordinate(4, 4); // b5
        Move move2 = new Move(initCoord2, finalCoord2, Piece.PieceType.PAWN, false);
        b.updateState(move2);

        Move move3 = new Move(
                new Coordinate(0, 3), new Coordinate(2, 5), Piece.PieceType.QUEEN, true
        );
        b.updateState(move3);

        Move move4 = new Move(
                new Coordinate(7, 1), new Coordinate(5, 2), Piece.PieceType.KNIGHT, false
        );
        b.updateState(move4);

        Move move5 = new Move(
                new Coordinate(0, 5), new Coordinate(3, 2), Piece.PieceType.BISHOP, true
        );
        b.updateState(move5);

        Move move6 = new Move(
                new Coordinate(5, 2), new Coordinate(3, 3), Piece.PieceType.KNIGHT, false
        );
        b.updateState(move6);

        Move move7 = new Move(
                new Coordinate(2, 5), new Coordinate(6, 5), Piece.PieceType.QUEEN, true
        );
        b.updateState(move7);
        b.calculateAllLegalMoves();

        assertEquals(2, b.checkMate());
    }

    @Test
    public void testCheckMateIsNotCheckMate() {
        Coordinate initCoord1 = new Coordinate(1, 4);
        Coordinate finalCoord1 = new Coordinate(3, 4);
        Move move1 = new Move(initCoord1, finalCoord1, Piece.PieceType.PAWN, true);
        b.updateState(move1);

        Coordinate initCoord2 = new Coordinate(6, 4);
        Coordinate finalCoord2 = new Coordinate(4, 4);
        Move move2 = new Move(initCoord2, finalCoord2, Piece.PieceType.PAWN, false);
        b.updateState(move2);

        Move move3 = new Move(
                new Coordinate(0, 3), new Coordinate(2, 5), Piece.PieceType.QUEEN, true
        );
        b.updateState(move3);

        Move move4 = new Move(
                new Coordinate(7, 1), new Coordinate(5, 2), Piece.PieceType.KNIGHT, false
        );
        b.updateState(move4);

        Move move7 = new Move(
                new Coordinate(2, 5), new Coordinate(6, 5), Piece.PieceType.QUEEN, true
        );
        b.updateState(move7);
        b.calculateAllLegalMoves();

        assertEquals(0, b.checkMate());
    }

    @Test
    public void testPinnedPieceCannotMove() {
        b.updateState(
                new Move(new Coordinate(1, 4), new Coordinate(3, 4), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 4), new Coordinate(4, 4), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 3), new Coordinate(2, 5), Piece.PieceType.QUEEN, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 5), new Coordinate(3, 1), Piece.PieceType.BISHOP, false)
        );
        b.calculateAllLegalMoves();

        boolean condition = false;
        for (Move move : b.getAllLegalMoves()) {
            if (move.getIntialCoordinate().equals(new Coordinate(1, 3))) { // checking the pawn at
                                                                           // that coordinate cannot
                                                                           // move
                condition = true;
            }
        }

        assertFalse(condition);
    }

    @Test
    public void testShortCastleIsAllowedWhite() {
        b.updateState(
                new Move(new Coordinate(1, 4), new Coordinate(3, 4), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 4), new Coordinate(4, 4), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 6), new Coordinate(2, 5), Piece.PieceType.KNIGHT, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 5), new Coordinate(3, 1), Piece.PieceType.BISHOP, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 5), new Coordinate(3, 2), Piece.PieceType.BISHOP, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 6), new Coordinate(5, 5), Piece.PieceType.KNIGHT, false)
        );
        b.calculateAllLegalMoves();

        boolean condition = false;
        for (Move move : b.getAllLegalMoves()) {
            if (move.getClass() == Castle.class) {
                condition = true;
                break;
            }
        }

        assertTrue(condition);

    }

    @Test
    public void testShortCastleMovesPiecesCorrectly() {
        b.updateState(
                new Move(new Coordinate(1, 4), new Coordinate(3, 4), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 4), new Coordinate(4, 4), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 6), new Coordinate(2, 5), Piece.PieceType.KNIGHT, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 5), new Coordinate(3, 1), Piece.PieceType.BISHOP, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 5), new Coordinate(3, 2), Piece.PieceType.BISHOP, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 6), new Coordinate(5, 5), Piece.PieceType.KNIGHT, false)
        );
        b.updateState(
                new Castle(
                        new Coordinate(0, 4), new Coordinate(0, 6), Piece.PieceType.KING, true,
                        new Coordinate(0, 7), new Coordinate(0, 5)
                )
        );
        b.calculateAllLegalMoves();

        boolean condition1 = false;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPieceType() == Piece.PieceType.KING && piece.isWhite()
                    && piece.getPosition().getColumn() == 6) {
                condition1 = true;
                break;
            }
        }

        boolean condition2 = false;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPieceType() == Piece.PieceType.ROOK && piece.isWhite()
                    && piece.getPosition().getColumn() == 5) {
                condition2 = true;
                break;
            }
        }

        assertTrue(condition2 && condition1);
    }

    @Test
    public void testCastleChecksForPreviousMovements() {
        b.updateState(
                new Move(new Coordinate(1, 4), new Coordinate(3, 4), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 4), new Coordinate(4, 4), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 6), new Coordinate(2, 5), Piece.PieceType.KNIGHT, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 5), new Coordinate(3, 1), Piece.PieceType.BISHOP, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 5), new Coordinate(3, 2), Piece.PieceType.BISHOP, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 6), new Coordinate(5, 5), Piece.PieceType.KNIGHT, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 4), new Coordinate(0, 5), Piece.PieceType.KING, true)
        );
        b.updateState(
                new Move(new Coordinate(5, 5), new Coordinate(7, 6), Piece.PieceType.KNIGHT, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 5), new Coordinate(0, 4), Piece.PieceType.KING, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 6), new Coordinate(5, 5), Piece.PieceType.KNIGHT, false)
        );
        b.calculateAllLegalMoves();

        boolean condition = true;
        for (Move move : b.getAllLegalMoves()) {
            if (move.getClass() == Castle.class) {
                condition = false;
                break;
            }
        }

        assertTrue(condition);
    }

    @Test
    public void testCastleIsNotAllowed() {
        b.updateState(
                new Move(new Coordinate(1, 4), new Coordinate(3, 4), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 4), new Coordinate(4, 4), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 6), new Coordinate(2, 5), Piece.PieceType.KNIGHT, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 5), new Coordinate(4, 2), Piece.PieceType.BISHOP, false)
        );
        b.updateState(
                new Move(new Coordinate(0, 5), new Coordinate(3, 2), Piece.PieceType.BISHOP, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 6), new Coordinate(5, 5), Piece.PieceType.KNIGHT, false)
        );
        b.updateState(
                new Castle(
                        new Coordinate(0, 4), new Coordinate(0, 6), Piece.PieceType.KING, true,
                        new Coordinate(0, 7), new Coordinate(0, 5)
                )
        );
        b.updateState(
                new Move(new Coordinate(6, 5), new Coordinate(5, 5), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(1, 0), new Coordinate(3, 0), Piece.PieceType.PAWN, true)
        );
        b.calculateAllLegalMoves();
        boolean condition = true;
        for (Move move : b.getAllLegalMoves()) {
            if (move.getClass() == Castle.class) {
                condition = false;
                break;
            }
        }

        assertTrue(condition);

        b.updateState(
                new Move(new Coordinate(6, 3), new Coordinate(4, 3), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(1, 7), new Coordinate(2, 7), Piece.PieceType.PAWN, true)
        );

        b.calculateAllLegalMoves();
        boolean condition2 = false;
        for (Move move : b.getAllLegalMoves()) {
            if (move.getClass() == Castle.class) {
                condition2 = true;
                break;
            }
        }

        assertTrue(condition2);
    }

    @Test
    public void testEnPassantWorksRight() {
        b.updateState(
                new Move(new Coordinate(1, 4), new Coordinate(3, 4), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 5), new Coordinate(4, 5), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(3, 4), new Coordinate(4, 5), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 6), new Coordinate(4, 6), Piece.PieceType.PAWN, false)
        );
        b.calculateAllLegalMoves();

        boolean condition = false;
        for (Move move : b.getAllLegalMoves()) {
            if (move.getClass() == EnPassant.class) {
                condition = true;
            }
        }

        assertTrue(condition);

        b.updateState(
                new EnPassant(
                        new Coordinate(4, 5), new Coordinate(3, 6), Piece.PieceType.PAWN, true,
                        new Coordinate(4, 6)
                )
        );
        boolean condition2 = true;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPosition().equals(new Coordinate(4, 6))) {
                condition2 = false;
                break;
            }
        }
        boolean condition3 = false;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPosition().equals(new Coordinate(3, 6))) {
                condition3 = true;
                break;
            }
        }

        assertTrue(condition3);
        assertTrue(condition2);
    }

    @Test
    public void testNoEnPassantLater() {
        b.updateState(
                new Move(new Coordinate(1, 4), new Coordinate(3, 4), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 5), new Coordinate(4, 5), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(3, 4), new Coordinate(4, 5), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 6), new Coordinate(4, 6), Piece.PieceType.PAWN, false)
        );
        b.calculateAllLegalMoves();

        boolean condition = false;
        for (Move move : b.getAllLegalMoves()) {
            if (move.getClass() == EnPassant.class) {
                condition = true;
            }
        }

        assertTrue(condition);

        b.updateState(
                new Move(new Coordinate(1, 0), new Coordinate(3, 0), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 0), new Coordinate(4, 0), Piece.PieceType.PAWN, false)
        );
        b.calculateAllLegalMoves();

        boolean condition2 = true;
        for (Move move : b.getAllLegalMoves()) {
            if (move.getClass() == EnPassant.class) {
                condition2 = false;
                break;
            }
        }

        assertTrue(condition2);
    }

    @Test
    public void testPromotion() {
        b.updateState(
                new Move(new Coordinate(1, 3), new Coordinate(3, 3), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 2), new Coordinate(4, 2), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(3, 3), new Coordinate(4, 2), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 3), new Coordinate(5, 3), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(4, 2), new Coordinate(5, 3), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 5), new Coordinate(4, 5), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(5, 3), new Coordinate(6, 3), Piece.PieceType.PAWN, true)
        );

        boolean condition = false;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPieceType() == Piece.PieceType.KING && !piece.isWhite()) {
                condition = ((King) piece).isInCheck(b);
                break;
            }
        }

        assertTrue(condition);

        b.updateState(
                new Move(new Coordinate(7, 4), new Coordinate(6, 5), Piece.PieceType.KING, false)
        );
        b.updateState(
                new Move(new Coordinate(1, 7), new Coordinate(3, 7), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(7, 3), new Coordinate(6, 2), Piece.PieceType.QUEEN, false)
        );
        Move promotionMove = new Promotion(
                new Coordinate(6, 3), new Coordinate(7, 3), Piece.PieceType.PAWN, true,
                Piece.PieceType.KNIGHT
        );
        b.updateState(promotionMove);

        boolean condition2 = false;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPieceType() == Piece.PieceType.KNIGHT && piece.isWhite()
                    && piece.getPosition().getRow() == 7) {
                condition2 = true;
                break;
            }
        }

        boolean condition3 = false;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPieceType() == Piece.PieceType.KING && !piece.isWhite()) {
                condition3 = ((King) piece).isInCheck(b);
                break;
            }
        }

        assertTrue(condition3);
        assertTrue(condition2);

    }

    @Test
    public void testCheckKingAndPromotion() {
        b.updateState(
                new Move(new Coordinate(1, 3), new Coordinate(3, 3), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 2), new Coordinate(4, 2), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(3, 3), new Coordinate(4, 2), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 3), new Coordinate(5, 3), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(4, 2), new Coordinate(5, 3), Piece.PieceType.PAWN, true)
        );
        b.updateState(
                new Move(new Coordinate(6, 5), new Coordinate(4, 5), Piece.PieceType.PAWN, false)
        );
        b.updateState(
                new Move(new Coordinate(5, 3), new Coordinate(6, 3), Piece.PieceType.PAWN, true)
        );

        boolean condition = false;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPieceType() == Piece.PieceType.KING && !piece.isWhite()) {
                condition = ((King) piece).isInCheck(b);
                break;
            }
        }

        assertTrue(condition);

        b.updateState(
                new Move(new Coordinate(7, 4), new Coordinate(6, 5), Piece.PieceType.KING, false)
        );
        Move promotionMove = new Promotion(
                new Coordinate(6, 3), new Coordinate(7, 2), Piece.PieceType.PAWN, true,
                Piece.PieceType.BISHOP
        );
        b.updateState(promotionMove);

        boolean condition2 = false;
        for (Piece piece : b.getPiecesOnBoard()) {
            if (piece.getPieceType() == Piece.PieceType.BISHOP && piece.isWhite()
                    && piece.getPosition().getRow() == 7) {
                condition2 = true;
                break;
            }
        }

        assertTrue(condition2);
    }


    @Test
    public void testPawn(){

    }
}
