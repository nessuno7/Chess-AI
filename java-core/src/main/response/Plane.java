package main.response;

import main.chess.Pieces.Piece;

public class Plane {
    private int[][] graphWhite;
    private int[][] graphBlack;
    private final Piece.PieceType pieceType;

    public Plane(Piece.PieceType type) {
        graphWhite = new int[8][8];
        graphBlack = new int[8][8];
        pieceType = type;
    }

    public Piece.PieceType getPieceType(){
        return pieceType;
    }

    public void update(Piece piece) throws Exception{
        if(piece.getPieceType() != this.pieceType) {
            throw new Exception("Wrong type of piece inserted in Plane");
        }

        if(piece.isWhite()){
            graphWhite[piece.getPosition().getRow()][piece.getPosition().getColumn()] = 1;
        }
        else{
            graphBlack[piece.getPosition().getRow()][piece.getPosition().getColumn()] = 1;
        }
    }

    public int[] flatten(){
        int[] rArray = new int[128];
        int count = 0;
        for(int[] white: graphWhite){
            for (int j : white) {
                rArray[count] = j;
                count++;
            }
        }
        for(int[] black: graphBlack){
            for (int j : black) {
                rArray[count] = j;
                count++;
            }
        }

        return rArray;
    }
}
