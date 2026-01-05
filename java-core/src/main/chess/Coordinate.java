package main.chess;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;

public class Coordinate implements Comparable<Coordinate> {
    private int row;
    private int column;

    public Coordinate(int row, int column) {
        this.row = row;
        this.column = column;
    }

    public Coordinate(Coordinate initCoordinate) {
        this.row = initCoordinate.getRow();
        this.column = initCoordinate.getColumn();
    }

    public int getRow() {
        return row;
    }

    public int getColumn() {
        return column;
    }

    public void setCoordinate(Coordinate coordinate) {
        this.row = coordinate.getRow();
        this.column = coordinate.getColumn();
    }

    public void setRow(int row) {
        this.row = row;
    }

    public void setColumn(int column) {
        this.column = column;
    }

    public boolean isValid() {
        return row >= 0 && column >= 0 && row < 8 && column < 8;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) {
            return true;
        }
        if (o == null || getClass() != o.getClass()) {
            return false;
        }
        Coordinate that = (Coordinate) o;
        return row == that.getRow() && column == that.getColumn();
    }

    @Override
    public int compareTo(Coordinate other) {
        if (this.isValid() && other.isValid()) {
            if (this.row > other.getRow()) {
                return 1;
            } else if (this.row == other.getRow()) {
                if (this.column > other.getColumn()) {
                    return 1;
                } else if (this.column == other.getColumn()) {
                    return 0;
                } else {
                    return -1;
                }
            } else {
                return -1;
            }
        } else {
            throw new IllegalArgumentException("Illegal Coordinate");
        }
    }

    public void rotate() {
        this.row = 7 - this.row;
        this.column = 7 - this.column;
    }

    public int flatten(){
        return (row*8+column);
    }

    @Override
    public String toString(){
        return ""+ flatten();
    }
}
