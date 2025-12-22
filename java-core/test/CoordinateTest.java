package test;


import main.chess.Coordinate;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class CoordinateTest {

    @Test
    public void testSetterMethodsBounds() {
        Coordinate coord = new Coordinate(0, 0);
        assertThrows(IllegalArgumentException.class, () -> coord.setRow(9));
        assertThrows(IllegalArgumentException.class, () -> coord.setColumn(9));
    }

    @Test
    public void testCorrectEncapsulation() {
        Coordinate coord = new Coordinate(0, 0);
        int row = coord.getRow();
        row = 5;

        int column = coord.getColumn();
        column = 3;

        assertNotSame(row, coord.getRow());
        assertNotSame(column, coord.getColumn());
    }

    @Test
    public void testEquals() {
        Coordinate coord = new Coordinate(0, 5);
        Coordinate coord1 = new Coordinate(0, 5);
        assertTrue(coord.equals(coord1));
        assertEquals(coord, coord1);
    }

    @Test
    public void testRotate() {
        Coordinate coord = new Coordinate(0, 5);
        coord.rotate();
        assertEquals(7, coord.getRow());
        assertEquals(2, coord.getColumn());
    }

}
