package main.chess.Pieces;

import main.chess.Coordinate;
import main.chess.Moves.Castle;
import main.chess.Moves.EnPassant;
import main.chess.Moves.Move;
import main.chess.Moves.Promotion;

import java.util.HashMap;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.TreeSet;

import static main.chess.Pieces.Piece.PieceType.*;


public class ChessBoard {
    private LinkedList<Piece> piecesOnBoard;
    private LinkedList<Piece> piecesTaken;
    private LinkedList<Move> allLegalMoves; // all legal moves for the current player
    private LinkedList<Move> allMovesTaken; // all moves that have been taken
    private boolean currentPlayer; // if true is white if false is black
    private int numberMovesNoPiecesTaken;
    private HashMap<String, Integer> previousStates;

    // TODO: add al the others pieces to the treeSet
    public ChessBoard() {
        this.piecesOnBoard = new LinkedList<Piece>();
        this.piecesTaken = new LinkedList<Piece>();
        this.allLegalMoves = new LinkedList<Move>();
        this.allMovesTaken = new LinkedList<Move>();
        this.previousStates = new HashMap<>();
        currentPlayer = true;

        Piece whiteKing = new King(new Coordinate(0, 4), true);
        Piece blackKing = new King(new Coordinate(7, 4), false);
        this.piecesOnBoard.add(whiteKing);
        this.piecesOnBoard.add(blackKing);

        Piece whiteRightRook = new Rook(new Coordinate(0, 7), true, false);
        Piece whiteLeftRook = new Rook(new Coordinate(0, 0), true, true);
        Piece blackRightRook = new Rook(new Coordinate(7, 7), false, true);
        Piece blackLeftRook = new Rook(new Coordinate(7, 0), false, false);
        this.piecesOnBoard.add(whiteRightRook);
        this.piecesOnBoard.add(whiteLeftRook);
        this.piecesOnBoard.add(blackRightRook);
        this.piecesOnBoard.add(blackLeftRook);

        Piece whiteKnight1 = new Knight(new Coordinate(0, 1), true);
        Piece whiteKnight2 = new Knight(new Coordinate(0, 6), true);
        Piece blackKnight1 = new Knight(new Coordinate(7, 1), false);
        Piece blackKnight2 = new Knight(new Coordinate(7, 6), false);
        this.piecesOnBoard.add(whiteKnight1);
        this.piecesOnBoard.add(whiteKnight2);
        this.piecesOnBoard.add(blackKnight1);
        this.piecesOnBoard.add(blackKnight2);

        Piece whiteBishop1 = new Bishop(new Coordinate(0, 2), true);
        Piece whiteBishop2 = new Bishop(new Coordinate(0, 5), true);
        Piece blackBishop1 = new Bishop(new Coordinate(7, 2), false);
        Piece blackBishop2 = new Bishop(new Coordinate(7, 5), false);
        this.piecesOnBoard.add(whiteBishop1);
        this.piecesOnBoard.add(whiteBishop2);
        this.piecesOnBoard.add(blackBishop1);
        this.piecesOnBoard.add(blackBishop2);

        Piece whiteQueen = new Queen(new Coordinate(0, 3), true);
        Piece blackQueen = new Queen(new Coordinate(7, 3), false);
        this.piecesOnBoard.add(whiteQueen);
        this.piecesOnBoard.add(blackQueen);

        Piece whitePawn1 = new Pawn(new Coordinate(1, 0), true);
        Piece whitePawn2 = new Pawn(new Coordinate(1, 1), true);
        Piece whitePawn3 = new Pawn(new Coordinate(1, 2), true);
        Piece whitePawn4 = new Pawn(new Coordinate(1, 3), true);
        Piece whitePawn5 = new Pawn(new Coordinate(1, 4), true);
        Piece whitePawn6 = new Pawn(new Coordinate(1, 5), true);
        Piece whitePawn7 = new Pawn(new Coordinate(1, 6), true);
        Piece whitePawn8 = new Pawn(new Coordinate(1, 7), true);
        this.piecesOnBoard.add(whitePawn1);
        this.piecesOnBoard.add(whitePawn2);
        this.piecesOnBoard.add(whitePawn3);
        this.piecesOnBoard.add(whitePawn4);
        this.piecesOnBoard.add(whitePawn5);
        this.piecesOnBoard.add(whitePawn6);
        this.piecesOnBoard.add(whitePawn7);
        this.piecesOnBoard.add(whitePawn8);

        Piece blackPawn1 = new Pawn(new Coordinate(6, 0), false);
        Piece blackPawn2 = new Pawn(new Coordinate(6, 1), false);
        Piece blackPawn3 = new Pawn(new Coordinate(6, 2), false);
        Piece blackPawn4 = new Pawn(new Coordinate(6, 3), false);
        Piece blackPawn5 = new Pawn(new Coordinate(6, 4), false);
        Piece blackPawn6 = new Pawn(new Coordinate(6, 5), false);
        Piece blackPawn7 = new Pawn(new Coordinate(6, 6), false);
        Piece blackPawn8 = new Pawn(new Coordinate(6, 7), false);
        this.piecesOnBoard.add(blackPawn1);
        this.piecesOnBoard.add(blackPawn2);
        this.piecesOnBoard.add(blackPawn3);
        this.piecesOnBoard.add(blackPawn4);
        this.piecesOnBoard.add(blackPawn5);
        this.piecesOnBoard.add(blackPawn6);
        this.piecesOnBoard.add(blackPawn7);
        this.piecesOnBoard.add(blackPawn8);
    }

    public ChessBoard(ChessBoard copy) {
        this.piecesOnBoard = copy.getPiecesOnBoard();
        this.piecesTaken = copy.getPiecesTaken();
        this.allLegalMoves = copy.getAllLegalMoves();
        this.allMovesTaken = copy.getAllMovesTaken();
        this.currentPlayer = copy.getCurrentPlayer();
        this.numberMovesNoPiecesTaken = copy.getNumberMovesNoPiecesTaken();
        this.previousStates = copy.getPreviousState();
    }

    public boolean isTherePieceAtCoordinate(Coordinate coordinate) {
        for (Piece piece : this.piecesOnBoard) {
            if (piece.getPosition().equals(coordinate)) {
                return true;
            }
        }
        return false;
    }

    public boolean returnColorPieceAtCoordinate(Coordinate coordinate)
            throws IllegalArgumentException {
        for (Piece piece : this.piecesOnBoard) {
            if (piece.getPosition().equals(coordinate)) {
                return piece.isWhite();
            }
        }
        throw new IllegalArgumentException("There is no piece at given coordinate");
    }

    public Piece.PieceType returnPieceTypeAtCoordinate(Coordinate coordinate)
            throws IllegalArgumentException {
        for (Piece piece : this.piecesOnBoard) {
            if (piece.getPosition().equals(coordinate)) {
                return piece.getPieceType();
            }
        }
        throw new IllegalArgumentException("There is no piece at given coordinate");
    }

    public void updateState(Move newMove) {
        String newHash = this.computeKey();
        if(this.previousStates.containsKey(newHash)) {
            this.previousStates.put(newHash, this.previousStates.get(newHash)+1);
        }
        else{
            this.previousStates.put(newHash, 1);
        }

        allMovesTaken.add(newMove);
        int initialPiecesTaken = this.piecesTaken.size();

        if (newMove.getClass() == Castle.class) {
            Castle castleMove = (Castle) newMove;
            Iterator<Piece> iterator = this.piecesOnBoard.iterator();
            while (iterator.hasNext()) {
                Piece piece = iterator.next();
                if (piece.getPosition().equals(castleMove.getIntialCoordinate())) { // move king
                    piece.movePiece(castleMove.getFinalCoordinate());
                } else if (piece.getPosition().equals(castleMove.getInitCoordinateRook())) {
                    piece.movePiece(castleMove.getFinalCoordinateRook());
                }
            }
        } else if (newMove.getClass() == EnPassant.class) {
            EnPassant enPassantMove = (EnPassant) newMove;
            Iterator<Piece> iterator = this.piecesOnBoard.iterator();
            while (iterator.hasNext()) {
                Piece piece = iterator.next();
                if (piece.getPosition().equals(enPassantMove.getIntialCoordinate())) { // move king
                    piece.movePiece(enPassantMove.getFinalCoordinate());
                } else if (piece.getPosition().equals(enPassantMove.getPawnTakenCoordinate())) {
                    this.piecesTaken.add(piece);
                    iterator.remove();
                }
            }
        } else if (newMove.getClass() == Promotion.class) {
            Promotion promotionMove = (Promotion) newMove;
            Iterator<Piece> iterator = this.piecesOnBoard.iterator();
            Piece newPiece = null;
            while (iterator.hasNext()) {
                Piece piece = iterator.next();
                if (piece.getPosition().equals(promotionMove.getIntialCoordinate())) {
                    newPiece = switch (promotionMove.getNewPieceType()) {
                        case KNIGHT -> new Knight(
                                new Coordinate(promotionMove.getFinalCoordinate()), piece.isWhite()
                        );
                        case QUEEN -> new Queen(
                                new Coordinate(promotionMove.getFinalCoordinate()), piece.isWhite()
                        );
                        case BISHOP -> new Bishop(
                                new Coordinate(promotionMove.getFinalCoordinate()), piece.isWhite()
                        );
                        case ROOK -> new Rook(
                                new Coordinate(promotionMove.getFinalCoordinate()), piece.isWhite(),
                                true
                        );
                        default -> throw new IllegalArgumentException("Illegal promotion piece");
                    };
                    iterator.remove();
                }
                if (piece.getPosition().equals(promotionMove.getFinalCoordinate())) {
                    iterator.remove();
                }

            }

            if (newPiece != null) {
                this.piecesOnBoard.add(newPiece);
            } else {
                throw new IllegalArgumentException("No pawn to promote");
            }
        } else {
            Iterator<Piece> iterator = this.piecesOnBoard.iterator();
            while (iterator.hasNext()) {
                Piece piece = iterator.next();
                if (piece.getPosition().equals(newMove.getFinalCoordinate())) {
                    iterator.remove();
                    piecesTaken.add(piece);
                }
            }
            iterator = this.piecesOnBoard.iterator();
            while (iterator.hasNext()) {
                Piece piece = iterator.next();
                if (piece.getPosition().equals(newMove.getIntialCoordinate())) {
                    piece.movePiece(newMove);
                }
            }
        }

        if(this.piecesTaken.size()!=initialPiecesTaken){
            numberMovesNoPiecesTaken = 0;
        }
        else{
            numberMovesNoPiecesTaken++;
        }

        this.currentPlayer = !this.currentPlayer; // at the end of every turn
    }

    public Move lastMove() {
        return this.allMovesTaken.getLast();
    }

    public TreeSet<Coordinate> getAllThreatendCoords(boolean white) {
        // color of the pieces you want
        // to see the scope (usually it is going to be !this.white)
        TreeSet<Coordinate> threatenedCoords = new TreeSet<Coordinate>();

        for (Piece piece : this.piecesOnBoard) {
            if (piece.isWhite() == white) {
                threatenedCoords.addAll(piece.returnScope(this));
            }
        }

        return threatenedCoords;
    }

    public HashMap<String, Integer> getPreviousState(){
        HashMap<String, Integer> returnState = new HashMap<>();
        for(String key: this.previousStates.keySet()){
            returnState.put(key, this.previousStates.get(key));
        }

        return returnState;
    }

    public LinkedList<Piece> getPiecesOnBoard() {
        LinkedList<Piece> returnList = new LinkedList<>();
        for (Piece piece : this.piecesOnBoard) {
            returnList.add(piece.copy());
        }
        return returnList;
    }

    public int getNumberMovesNoPiecesTaken(){
        return this.numberMovesNoPiecesTaken;
    }

    public LinkedList<Piece> getPiecesTaken() {
        LinkedList<Piece> returnList = new LinkedList<>();
        for (Piece piece : this.piecesTaken) {
            returnList.add(piece.copy());
        }
        return returnList;
    }

    public void calculateAllLegalMoves() {
        this.allLegalMoves.clear();
        LinkedList<Move> appList = new LinkedList<>();
        if (!this.currentPlayer) { // if is black rotates the coordinates
            ChessBoard rotChessBoard = new ChessBoard(this);
            rotChessBoard.rotateCoords();
            for (Piece piece : rotChessBoard.getPiecesOnBoard()) {
                if (piece.isWhite() == rotChessBoard.getCurrentPlayer()) {
                    appList.addAll(piece.getAllLegalMoves(rotChessBoard));
                }
            }
            for (Move move : appList) { // rotates back
                move.rotateMove();
            }
            allLegalMoves.addAll(appList);
        } else {
            for (Piece piece : this.piecesOnBoard) {
                if (piece.isWhite() == this.currentPlayer) {
                    allLegalMoves.addAll(piece.getAllLegalMoves(this));
                }
            }
        }
    }

    public LinkedList<Move> getAllLegalMoves() {
        LinkedList<Move> returnList = new LinkedList<>();
        for (Move move : this.allLegalMoves) {
            returnList.add(move.copy());
        }
        return returnList;
    }

    public LinkedList<Move> getAllMovesTaken() {
        LinkedList<Move> returnList = new LinkedList<>();
        for (Move move : this.allMovesTaken) {
            returnList.add(move.copy());
        }
        return returnList;
    }

    public boolean getCurrentPlayer() {
        return this.currentPlayer;
    }

    public boolean simulateMove(Move move) {
        this.updateState(move);
        for (Piece piece : this.piecesOnBoard) {
            if (piece.getPieceType() == KING
                    && piece.isWhite() == !this.getCurrentPlayer()) {
                King kingPiece = (King) piece;
                return !kingPiece.isInCheck(this);
            }
        }

        System.out.print("Wait");
        throw new IllegalArgumentException("There is no King Piece");
    }

    public void rotateCoords() {
        // when it is black's turn to move, a copy of the pieces List and a
        // copy of the MoveTaken list get their coordinate swapped
        // a new chessBoard object is create and a list of all legal moves is found
        // then all the coordinate of the legal moves lost have they coordinates swapped
        // again, so they can be used o the standard chessboad.

        for (Piece piece : this.piecesOnBoard) {
            Coordinate rotatedCoord = new Coordinate(piece.getPosition());
            rotatedCoord.rotate();
            piece.movePiece(rotatedCoord);
        }
        for (Move move : this.allMovesTaken) {
            move.rotateMove();
        }
    }

    public int checkMate() {
        if(this.isDrawByRepetition()){
            this.allLegalMoves.clear();
            return 5;//draw by repetition
        }

        if(numberMovesNoPiecesTaken >= 50){
            this.allLegalMoves.clear();
            return 4; //draw for the 50 move rule
        }

        if (this.allLegalMoves.isEmpty()) {
            for (Piece piece : this.piecesOnBoard) {
                if (piece.getPieceType() == KING
                        && piece.isWhite() == this.currentPlayer) {
                    King kingPiece = (King) piece;
                    if (kingPiece.isInCheck(this)) {
                        return 2; // checkmate
                    } else {
                        return 1; // stalemate
                    }
                }
            }
        }

        if (this.piecesOnBoard.size() == 2) {
            this.allLegalMoves.clear();
            return 3; //draw for insufficient material
        }
        if (this.piecesOnBoard.size() == 3) {
            for (Piece piece : this.piecesOnBoard) {
                if (piece.getPieceType() == KNIGHT || piece.getPieceType() == BISHOP) {
                    this.allLegalMoves.clear();
                    return 3; //draw for insufficient material
                }
            }
        }
        if (piecesOnBoard.size() == 4) {
            int bishopCount = 0;
            int knightCount = 0;
            LinkedList<Piece> bishops = new LinkedList<>();

            for (Piece p : piecesOnBoard) {
                switch (p.getPieceType()) {
                    case BISHOP -> {
                        bishopCount++;
                        bishops.add(p);
                    }
                    case KNIGHT -> knightCount++;
                    case KING -> {
                    }
                    default -> {
                        return 0; //continue
                    }
                }
            }
            if (bishopCount == 2 && knightCount == 0) {
                boolean firstColorSquare = (bishops.get(0).getPosition().getRow()
                        + bishops.get(0).getPosition().getColumn()) % 2 == 1;
                boolean secondColorSquare = (bishops.get(1).getPosition().getRow()
                        + bishops.get(1).getPosition().getColumn()) % 2 == 0;
                if (firstColorSquare == secondColorSquare) {
                    this.allLegalMoves.clear();
                    return 3; //draw for unsufficient material
                } else {
                    return 0; //continue
                }
            }
        }

        return 0; //continue
    }

    public boolean removeLastMoveTaken() {
        if (!this.allMovesTaken.isEmpty()) {
            this.allMovesTaken.removeLast();
            return true;
        } else {
            return false;
        }
    }

    public boolean isDrawByRepetition(){
        if(this.previousStates.containsKey(this.computeKey())){
            return this.previousStates.get(this.computeKey())>=3;
        }
        else{
            return false;
        }
    }

    private String computeKey(){
        StringBuilder sb = new StringBuilder();
        for(Piece piece : this.piecesOnBoard){
            String c;
            if(piece.isWhite()){
                c="w";
            }
            else{
                c="b";
            }
            switch (piece.getPieceType()){
                case KING -> {
                    sb.append("K"+piece.getPosition().getRow()+piece.getPosition().getColumn()+c);
                    if (((King)piece).getIsLongCastleAllowedVar()){
                        sb.append("LCA");
                    }
                    else{
                        sb.append("LCN");
                    }

                    if (((King)piece).getIsShortCastleAllowedVar()){
                        sb.append("SCA");
                    }
                    else{
                        sb.append("SCN");
                    }
                }
                case QUEEN -> sb.append("Q"+piece.getPosition().getRow()+piece.getPosition().getColumn()+c);
                case BISHOP -> sb.append("B"+piece.getPosition().getRow()+piece.getPosition().getColumn()+c);
                case PAWN -> sb.append("P"+piece.getPosition().getRow()+piece.getPosition().getColumn()+c);
                case ROOK -> {
                    sb.append("R"+piece.getPosition().getRow()+piece.getPosition().getColumn()+c);
                    if(((Rook)piece).getHasMovedAlready()){
                        sb.append("MA");
                    }
                    else{
                        sb.append("MN");
                    }
                }
                case KNIGHT -> sb.append("N"+piece.getPosition().getRow()+piece.getPosition().getColumn()+c);
            }
        }

        if(this.currentPlayer){
            sb.append("w");
        }
        else{
            sb.append("b");
        }

        for(Move move : this.allLegalMoves){
            if(move instanceof EnPassant){
                sb.append("EP"+move.getIntialCoordinate().getRow()+move.getIntialCoordinate().getColumn()+move.getFinalCoordinate().getRow()+move.getFinalCoordinate().getColumn());
            }
        }

        return sb.toString();
    }
}
