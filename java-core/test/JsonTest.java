import com.google.gson.Gson;
import com.google.gson.JsonObject;
import main.chess.Coordinate;
import main.chess.Moves.Castle;
import main.chess.Moves.EnPassant;
import main.chess.Moves.Move;
import main.chess.Moves.Promotion;
import main.chess.Pieces.Piece;
import org.junit.jupiter.api.Test;

import static org.junit.Assert.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

public class JsonTest {
    @Test
    public void testCoordinateToJson() {
        Coordinate coord = new Coordinate(5, 2);
        Gson gson = new Gson();
        assertEquals("{\"row\":5,\"column\":2}", gson.toJson(coord));
    }

    @Test
    public void testMoveToJson() {
        Coordinate coord1 = new Coordinate(5, 2);
        Coordinate coord2 = new Coordinate(7, 3);
        Move move = new Move(coord1, coord2, Piece.PieceType.KNIGHT, true);
        Gson gson = new Gson();
        assertEquals("{\"intialCoordinate\":{\"row\":5,\"column\":2},\"finalCoordinate\":{\"row\":7,\"column\":3},\"pieceType\":\"KNIGHT\",\"white\":true}", gson.toJson(move));
    }

    @Test
    public void testEnPassantToJson() {
        Coordinate coord1 = new Coordinate(5, 2);
        Coordinate coord2 = new Coordinate(7, 3);
        Coordinate coord3 = new Coordinate(5, 4);
        Move move = new EnPassant(coord1, coord2, Piece.PieceType.KNIGHT, true, coord3);
        Gson gson = new Gson();
        assertEquals("{\"pawnTakenCoordinate\":{\"row\":5,\"column\":4},\"intialCoordinate\":{\"row\":5,\"column\":2},\"finalCoordinate\":{\"row\":7,\"column\":3},\"pieceType\":\"KNIGHT\",\"white\":true}", gson.toJson(move));
    }

    @Test
    public void testCastleToJson() {
        Coordinate coord1 = new Coordinate(5, 2);
        Coordinate coord2 = new Coordinate(7, 3);
        Coordinate coord3 = new Coordinate(5, 4);
        Coordinate coord4 = new Coordinate(5, 5);
        Move move = new Castle(coord1, coord2, Piece.PieceType.KNIGHT, true, coord3, coord4);
        Gson gson = new Gson();
        assertEquals("{\"initCoordinateRook\":{\"row\":5,\"column\":4},\"finalCoordinateRook\":{\"row\":5,\"column\":5},\"intialCoordinate\":{\"row\":5,\"column\":2},\"finalCoordinate\":{\"row\":7,\"column\":3},\"pieceType\":\"KNIGHT\",\"white\":true}", gson.toJson(move));
    }

    @Test
    public void testPromotionToJson() {
        Coordinate coord1 = new Coordinate(5, 2);
        Coordinate coord2 = new Coordinate(7, 3);
        Move move = new Promotion(coord1, coord2, Piece.PieceType.PAWN, true, Piece.PieceType.QUEEN);
        Gson gson = new Gson();
        assertEquals("{\"newPieceType\":\"QUEEN\",\"intialCoordinate\":{\"row\":5,\"column\":2},\"finalCoordinate\":{\"row\":7,\"column\":3},\"pieceType\":\"PAWN\",\"white\":true}", gson.toJson(move));
    }

    @Test
    public void testNewFieldName(){
        Gson gson = new Gson();
        JsonObject json = new JsonObject();
        int[] newArr = {0,1,2,3,4,5};
        json.add("planes", gson.toJsonTree(newArr));
        assertEquals("{\"planes\":[0,1,2,3,4,5]}",gson.toJson(json));
    }

}
