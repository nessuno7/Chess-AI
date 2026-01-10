package main.response;

import main.chess.Chess;
import main.chess.Moves.Move;
import main.chess.Moves.Promotion;
import main.chess.Pieces.ChessBoard;
import main.chess.Pieces.King;
import main.chess.Pieces.Piece;

import java.awt.*;
import java.beans.PropertyChangeListener;
import java.beans.PropertyChangeSupport;
import java.util.LinkedList;
import java.util.List;

import rl.*;

import javax.swing.*;

import static main.chess.Pieces.Piece.PieceType.KING;


public class EnvEncoder {
    ChessBoard chessBoard;
    Plane[] piecePlaneArray;
    List<Move> currentMoves;
    float rewardBefore;
    float rewardCurrent;
    boolean done;
    boolean currentPlayer;
    int count =0;
    final PropertyChangeSupport pcs = new PropertyChangeSupport(this);
    final int index;
    int countReward;

    public EnvEncoder(int index) throws Exception{
        this.index = index;
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
        countReward = 0;

        currentPlayer = chessBoard.getCurrentPlayer();
    }

    public void addChangeListener(PropertyChangeListener l) {
        pcs.addPropertyChangeListener(l);
    }

    public void removeChangeListener(PropertyChangeListener l) {
        pcs.removePropertyChangeListener(l);
    }

    public void firePropertyChange(){
        pcs.firePropertyChange("position", null, null);
    }

    public ChessBoard getChessBoard() {
        return chessBoard;
    }

    public void reset() throws Exception{
        this.chessBoard = new ChessBoard();
        this.piecePlaneArray = new Plane[6];
        this.currentMoves = this.getExpandedLegalMoves();

        for(int i=0; i<6; i++){
            piecePlaneArray[i] = new Plane(Piece.PieceType.values()[i]);
        }

        updatePlanes();
        done = false;
        currentPlayer = chessBoard.getCurrentPlayer();
    }

    public void step(rl.ProtoMove action) throws Exception{
        count ++;
        if (currentMoves == null || currentMoves.isEmpty()) {
            currentMoves = getExpandedLegalMoves();
        }
        currentPlayer = chessBoard.getCurrentPlayer();

        if(action.getFromSq() != action.getToSq()){ //checks that the action is not an initial request action
            for(Move move: currentMoves){
                if(move.getFlattenInitCoord() == action.getFromSq() && move.getFlattenFinalCoord() == action.getToSq()){
                    if(move.getClass() == Promotion.class){
                        if (((Promotion)move).getNewPieceType() == switch(action.getPromotion()){
                            case 1-> Piece.PieceType.QUEEN;
                            case 2-> Piece.PieceType.ROOK;
                            case 3 -> Piece.PieceType.BISHOP;
                            case 4-> Piece.PieceType.KNIGHT;
                            default -> throw new IllegalArgumentException("Unexpected value: " + action.getPromotion());
                        }){
                            if(currentPlayer){
                                move.rotateMove();
                            }
                            chessBoard.updateState(move);
                            break;
                        }

                    }
                    else{
                        if(currentPlayer){
                            move.rotateMove();
                        }
                        chessBoard.updateState(move);
                        break;
                    }
                }
            }
        }

        currentMoves = getExpandedLegalMoves();
        updateRewards();
        updatePlanes();
        pcs.firePropertyChange("position", null, null);
    }

    private void updatePlanes() throws Exception{
        if(currentPlayer){
            this.chessBoard.rotateCoords(); //rotates if curr player is black
        }
        for(Piece piece: this.chessBoard.getPiecesOnBoard()){
            piecePlaneArray[piece.getPieceType().getPieceId()].update(piece);
        }

        if(currentPlayer){
            this.chessBoard.rotateCoords(); //rotates back
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
            }
        }
        return currentMoves;
    }

    private void updateRewards(){
        int winner = chessBoard.checkMate();
        done = (winner != 0);

        if (winner == 2) {
            this.countReward = 1;
            rewardBefore = 0;
            rewardCurrent = -1;
        } else if (winner == 1) {
            this.countReward = 2;
            rewardBefore = 0;
            rewardCurrent = -0.01f;
        }
        else{
            if(this.countReward == 1){
                rewardBefore = rewardCurrent;
                rewardCurrent = 1;
                this.countReward = 0;
            }
            else if(this.countReward == 2){
                rewardBefore = rewardCurrent;
                rewardCurrent = -0.02f;
                this.countReward = 0;
            }
            else{
                rewardBefore = rewardCurrent;
                rewardCurrent = 0;
            }
        }
    }

    public float getRewardBefore() {
        return rewardBefore;
    }

    public boolean isDone(){
        return done;
    }


    /*
    after step()
    firs call to the function, without a move, will return current state, and current possible moves
    Then futures calls with move to execute
     */
    public EnvState toEnvStateResponse() throws Exception{
        EnvState.Builder envStateBuild = EnvState.newBuilder();

        Observation.Builder obsBuild = Observation.newBuilder();
        float[] obs = getObservationFloat();
        obsBuild.setDimension(obs.length);
        for (float v : obs) obsBuild.addObservationPoints(v);
        envStateBuild.setObs(obsBuild);

        envStateBuild.setReward(getRewardBefore()); //always returns the

        for (Move m : currentMoves) { //rotates move if it is black so the RL always return in absolute terms
            if(!currentPlayer){
                m.rotateMove();
            }
            ProtoMove.Builder protoMove = ProtoMove.newBuilder();
            protoMove.setFromSq(m.getFlattenInitCoord());
            protoMove.setToSq(m.getFlattenFinalCoord());

            if(m.getClass() == Promotion.class){
                protoMove.setPromotion(switch(((Promotion) m).getNewPieceType()){
                    case QUEEN-> 1;
                    case ROOK-> 2;
                    case BISHOP -> 3;
                    case KNIGHT-> 4;
                    default -> throw new IllegalStateException("Unexpected value: " + ((Promotion) m).getNewPieceType());
                });
            }
            else{
                protoMove.setPromotion(0);
            }

            envStateBuild.addMoves(protoMove);
        }

        return envStateBuild.build();
    }
}
