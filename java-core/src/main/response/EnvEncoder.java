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
    private final PropertyChangeSupport pcs = new PropertyChangeSupport(this);
    final int index;

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

    /*public void displayChessBoardOnce(){
        SwingUtilities.invokeLater(() -> {
            final JFrame frame = new JFrame("Chess");
            frame.setLocation(300, 300);

            final JPanel status_panel = new JPanel();
            frame.add(status_panel, BorderLayout.SOUTH);
            final JLabel status = new JLabel("Setting up...");
            status_panel.add(status);
            // Game board
            final Chess board = new Chess(status);
            board.loadChessBoard(this.chessBoard);
            frame.add(board, BorderLayout.CENTER);

            frame.pack();
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setVisible(true);
        });
    }*/

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
                            System.out.printf("%-10s %-8s %-6s%n", index, currentPlayer, move);
                            chessBoard.updateState(move);

                            int kCount = 0;
                            for (Piece piece : this.chessBoard.getPiecesOnBoard()) { //TODO: lines to delete
                                if (piece.getPieceType() == KING) {
                                    kCount++;
                                }
                            }
                            if(kCount < 2){
                                System.out.println("Sone king missing: " + kCount);
                            }

                            break;

                        }

                    }
                    else{
                        if(currentPlayer){
                            move.rotateMove();
                        }
                        System.out.printf("%-10s %-8s %-6s%n", index, currentPlayer, move);

                        chessBoard.updateState(move);

                        int kCount = 0;
                        for (Piece piece : this.chessBoard.getPiecesOnBoard()) { //TODO: lines to delete
                            if (piece.getPieceType() == KING) {
                                kCount++;
                            }
                        }
                        if(kCount < 2){
                            //show the previous state
                            ChessBoard prevMove1 = new ChessBoard(this.chessBoard);
                            if (prevMove1.removeLastMoveTaken()) {
                                LinkedList<Move> moves = prevMove1.getAllMovesTaken();
                                prevMove1 = new ChessBoard();
                                for (Move move_ : moves) {
                                    prevMove1.updateState(move_);
                                }
                                prevMove1.calculateAllLegalMoves();//TODO: debugging lines to remove
                            }

                            ChessBoard prevMove2 = new ChessBoard(prevMove1);
                            if (prevMove2.removeLastMoveTaken()) {
                                LinkedList<Move> moves = prevMove2.getAllMovesTaken();
                                prevMove2 = new ChessBoard();
                                for (Move move_ : moves) {
                                    prevMove2.updateState(move_);
                                }
                                prevMove2.calculateAllLegalMoves();//TODO: debugging lines to remove
                            }

                            System.out.println("Sone king missing: " + kCount);
                        }

                        break;
                    }
                }
            }
        }

        try{
            currentMoves = getExpandedLegalMoves();
        }
        catch(Exception e){
            System.out.println("Exception King Piece not present"); //TODO: debugging lines to remove
            int sq_from = action.getFromSq();
            int sq_to = action.getToSq();
            int sq_k = -1;
            for(Piece piece: this.chessBoard.getPiecesTaken()){
                if(piece.getPieceType() == Piece.PieceType.KING){
                    sq_k = piece.getPosition().flatten();
                }
            }

            System.out.printf("%-10s %-8s %-6s%n", "from sq move: " + sq_from, "to sq move: " + sq_to, "position king: " + sq_k);

        }
        updateRewards();
        updatePlanes();
        pcs.firePropertyChange("position", null, null);
    }

    private void updatePlanes() throws Exception{
        if(currentPlayer){
            this.chessBoard.rotateCoords(); //rotates if curr player is black
        }
        for(Piece piece: this.chessBoard.getPiecesOnBoard()){
            piecePlaneArray[piece.getPieceType().getPieceId()].update(piece); //TODO: rotates ebfore move when it is black to move
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

        envStateBuild.setReward(getRewardBefore());
        envStateBuild.setDone(isDone());
        envStateBuild.setSide(currentPlayer);

        if (currentMoves == null || currentMoves.isEmpty()) {
            currentMoves = getExpandedLegalMoves();
        }


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
