package main.Response;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import main.chess.Moves.Move;
import main.chess.Moves.Promotion;
import main.chess.Pieces.ChessBoard;
import main.chess.Pieces.Piece;

import java.util.LinkedList;
import java.util.List;


public class EnvEncoder {
    ChessBoard chessBoard;
    Plane[] piecePlaneArray;
    List<Move> currentMoves;
    double rewardBefore;
    double rewardCurrent;

    public EnvEncoder() {
        this.chessBoard = new ChessBoard();
        this.piecePlaneArray = new Plane[6];
        this.currentMoves = new LinkedList<>();

        for(int i=0; i<6; i++){
            piecePlaneArray[i] = new Plane(Piece.PieceType.values()[i]);
        }
    }

    public void reset() {
        chessBoard = new ChessBoard();
        chessBoard.calculateAllLegalMoves();
    }

    private void updatePlanes() throws Exception{
        for(Piece piece: this.chessBoard.getPiecesOnBoard()){
            piecePlaneArray[piece.getPieceType().getPieceId()].update(piece);
        }
    }

    private int[] flattenAll(){
        int[] rArray = new int[769];
        int count = 0;
        for(Plane plane: piecePlaneArray){
            for(Integer i: plane.flatten()){
                rArray[count] = i;
                count++;
            }
        }

        if(chessBoard.getCurrentPlayer()){
            rArray[count] = 1;
        }
        else{
            rArray[count] = 0;
        }

        return rArray;
    }

    public Gson respond(){
        Gson gson = new Gson();

        JsonObject json= new JsonObject();
        json.add("planes", gson.toJsonTree(flattenAll()));

        currentMoves = chessBoard.getAllLegalMoves();
        for(Move move: chessBoard.getAllLegalMoves()){
            if(move.getClass() == Promotion.class){
                Move epKnight = move.copy();
                ((Promotion)epKnight).setNewPieceType(Piece.PieceType.KNIGHT);
                Move epRook = move.copy();
                ((Promotion)epRook).setNewPieceType(Piece.PieceType.ROOK);
                Move epBishop = move.copy();
                ((Promotion)epBishop).setNewPieceType(Piece.PieceType.BISHOP);

                currentMoves.add(epKnight);
                currentMoves.add(epRook);
                currentMoves.add(epBishop);
                break;
            }
        }

        json.add("legal moves", gson.toJsonTree(currentMoves));
        json.addProperty("reward previous", rewardBefore);
        json.addProperty("reward current", rewardCurrent);

        gson.toJson(json);

        return gson;
    }


    public void step(int moveIndex) throws Exception{
        chessBoard.updateState(currentMoves.get(moveIndex));
        chessBoard.calculateAllLegalMoves();

        int winner = chessBoard.checkMate();

        if (winner == 2) {
            rewardBefore = 1;
            rewardCurrent = -1;

        } else if (winner >= 1) {
             rewardBefore = -0.5;
             rewardCurrent = -0.5;
        }
        else{
            rewardBefore = 0;
            rewardCurrent = 0;
        }

        updatePlanes();
    }

}
