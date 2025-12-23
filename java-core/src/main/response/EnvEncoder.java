package main.response;

import com.google.gson.Gson;
import com.google.gson.JsonObject;
import main.chess.Moves.Move;
import main.chess.Moves.Promotion;
import main.chess.Pieces.ChessBoard;
import main.chess.Pieces.Piece;

import java.util.LinkedList;
import java.util.List;
import rl.StepRequest;
import rl.StepResponse;
//import rl.Move;
import rl.LegalMoves;
import rl.Observation;


public class EnvEncoder {
    ChessBoard chessBoard;
    Plane[] piecePlaneArray;
    List<Move> currentMoves;
    float rewardBefore;
    float rewardCurrent;
    boolean done;

    public EnvEncoder() throws Exception{
        this.chessBoard = new ChessBoard();
        this.piecePlaneArray = new Plane[6];
        this.chessBoard.calculateAllLegalMoves();
        this.currentMoves =  this.chessBoard.getAllLegalMoves();

        for(int i=0; i<6; i++){
            piecePlaneArray[i] = new Plane(Piece.PieceType.values()[i]);
        }

        rewardBefore = 0;
        rewardCurrent = 0;
        updatePlanes();
        updateRewards();
    }

    public void reset() throws Exception{
        this.chessBoard = new ChessBoard();
        this.piecePlaneArray = new Plane[6];
        this.currentMoves = this.getExpandedLegalMoves();

        for(int i=0; i<6; i++){
            piecePlaneArray[i] = new Plane(Piece.PieceType.values()[i]);
        }

        rewardBefore = 0;
        rewardCurrent = 0;
        updatePlanes();
        updateRewards();
    }

    public void step(int moveIndex) throws Exception{
        if (currentMoves == null || currentMoves.isEmpty()) {
            currentMoves = getExpandedLegalMoves();
        }

        chessBoard.updateState(currentMoves.get(moveIndex));
        currentMoves = getExpandedLegalMoves();
        updateRewards();
        updateRewards();
    }

    private void updatePlanes() throws Exception{
        for(Piece piece: this.chessBoard.getPiecesOnBoard()){
            piecePlaneArray[piece.getPieceType().getPieceId()].update(piece);
        }
    }

    private int[] flattenAll() throws Exception{
        updatePlanes();
        int[] rArray = new int[768];
        int count = 0;
        for(Plane plane: piecePlaneArray){
            for(Integer i: plane.flatten()){
                rArray[count] = i;
                count++;
            }
        }

        return rArray;
    }

    public float[] getObservationFloat() throws Exception{
        int[] obsInt = flattenAll();
        float[] obs = new float[obsInt.length];
        for (int i = 0; i < obsInt.length; i++) obs[i] = obsInt[i];
        return obs;
    }

    public List<Move> getExpandedLegalMoves(){
        chessBoard.calculateAllLegalMoves();
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
        return currentMoves;
    }

    private void updateRewards(){
        int winner = chessBoard.checkMate();
        done = (winner != 0);

        if (winner == 2) {
            rewardBefore = 1;
            rewardCurrent = -1;

        } else if (winner >= 1) {
            rewardBefore = -0.5f;
            rewardCurrent = -0.5f;
        }
        else{
            rewardBefore = 0;
            rewardCurrent = 0;
        }
    }

    public float getRewardBefore() {
        updateRewards();
        return rewardBefore;
    }

    public float getRewardCurrent() {
        updateRewards();
        return rewardCurrent;
    }

    public boolean isdone(){
        return done;
    }

    public rl.RlProto.EnvState toEnvStateProto() {
        rl.RlProto.EnvState.Builder b = rl.RlProto.EnvState.newBuilder();

        float[] obs = getObservationFloat();
        b.setObsDim(obs.length);
        for (float v : obs) b.addObs(v);

        b.setReward(getRewardCurrent());
        b.setDone(isDone());

        // legal moves
        if (currentMoves == null || currentMoves.isEmpty()) {
            currentMoves = buildExpandedLegalMoveList();
        }
        for (Move m : currentMoves) {
            b.addLegalMoves(toProtoMove(m));
        }

        // optional metadata (if you add them to proto)
        // b.setWhiteToMove(chessBoard.getCurrentPlayer());
        // b.setPly(...);

        return b.build();
    }





    /*public Gson respond(){
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


    */
}
