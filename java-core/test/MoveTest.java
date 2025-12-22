package test;

import main.chess.Coordinate;
import main.chess.Moves.Move;
import main.chess.Pieces.Piece;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class MoveTest {
    Move m;

    @BeforeEach
    public void setUp() {
        m = new Move(new Coordinate(1, 0), new Coordinate(1, 3), Piece.PieceType.PAWN, true);
    }

    @Test
    public void testEncapsulation() {
        Move m1 = m.copy();
        Coordinate finalCoord = m.getFinalCoordinate();
        Coordinate intialCoord = m.getIntialCoordinate();
        Piece.PieceType pieceType = m.getPieceType();

        pieceType = Piece.PieceType.QUEEN;
        intialCoord.rotate();
        finalCoord.rotate();

        assertNotSame(pieceType, m.getPieceType());
        assertEquals(m1, m);
    }

    @Test
    public void testEquals() {
        Coordinate finalCoord = m.getFinalCoordinate();
        Coordinate intialCoord = m.getIntialCoordinate();

        Move m1 = new Move(intialCoord, finalCoord, Piece.PieceType.PAWN, true);

        assertEquals(m1, m);
    }

    @Test
    public void testRotateMove() {
        Coordinate finalCoord = m.getFinalCoordinate();
        Coordinate intialCoord = m.getIntialCoordinate();
        Piece.PieceType pieceType = m.getPieceType();

        intialCoord.rotate();
        finalCoord.rotate();

        Move m1 = new Move(intialCoord, finalCoord, pieceType, m.isWhite());
        m.rotateMove();
        assertEquals(m1, m);
    }
}
