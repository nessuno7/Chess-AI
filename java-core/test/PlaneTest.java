import main.chess.Pieces.ChessBoard;
import main.chess.Pieces.Pawn;
import main.chess.Pieces.Piece;
import main.response.Plane;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.*;

public class PlaneTest {
    ChessBoard b;

    @BeforeEach
    public void setUp(){
        b = new ChessBoard();
    }

    @Test
    public void testPlane1Pawn() throws Exception{
        Plane plane = new Plane(Piece.PieceType.PAWN);

        for(Piece piece: b.getPiecesOnBoard()){
            if(piece.getClass() == Pawn.class){
                plane.update(piece);
            }
        }

        int[] arr = {
                0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                0, 0, 0, 0, 0, 0, 0, 0,
                1, 1, 1, 1, 1, 1, 1, 1,
                0, 0, 0, 0, 0, 0, 0, 0
        };

        assertArrayEquals(arr, plane.flatten());
    }


}
