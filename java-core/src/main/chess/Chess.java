package main.chess;

import main.chess.Moves.Move;
import main.chess.Moves.Promotion;
import main.chess.Pieces.ChessBoard;
import main.chess.Pieces.King;
import main.chess.Pieces.Piece;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;

import static main.chess.Pieces.Piece.PieceType;

public class Chess extends JPanel {
    ChessBoard currentGameState;
    private LinkedList<Move> allCurrentLegalMoves;
    private JLabel status; // current status text
    private static BufferedImage chessBoardImage;
    private static BufferedImage whitePawn;
    private static BufferedImage blackPawn;
    private static BufferedImage whiteRook;
    private static BufferedImage blackRook;
    private static BufferedImage whiteBishop;
    private static BufferedImage blackBishop;
    private static BufferedImage whiteKnight;
    private static BufferedImage blackKnight;
    private static BufferedImage whiteKing;
    private static BufferedImage blackKing;
    private static BufferedImage whiteQueen;
    private static BufferedImage blackQueen;
    private boolean pieceSelected;
    private Coordinate initialCoordinate;

    public void calculateCurrentLegalMoves() {
        allCurrentLegalMoves = currentGameState.getAllLegalMoves();
    }

    // Game constants
    public static final int BOARD_WIDTH = 500;
    public static final int BOARD_HEIGHT = 500;

    /**
     * Initializes the game board.
     */
    private static BufferedImage loadImage(String path) throws  IOException {
        InputStream is = Chess.class.getClassLoader().getResourceAsStream(path);

        if (is == null) {
            throw new IllegalArgumentException("Image not found: " + path);
        }
        return ImageIO.read(is);
    }

    public Chess(JLabel statusInit) {
        this.initialCoordinate = new Coordinate(0, 0);
        try {
            if (chessBoardImage == null) {
                chessBoardImage = loadImage("files/chessboard.png");
            }
            if (whitePawn == null) {
                whitePawn = loadImage("files/whitepawn.png");
            }
            if (blackPawn == null) {
                blackPawn = loadImage("files/blackpawn.png");
            }
            if (whiteKing == null) {
                whiteKing = loadImage("files/whiteking.png");
            }
            if (blackKing == null) {
                blackKing = loadImage("files/blackking.png");
            }
            if (whiteBishop == null) {
                whiteBishop = loadImage("files/whitebishop.png");
            }
            if (blackBishop == null) {
                blackBishop = loadImage("files/blackbishop.png");
            }
            if (whiteKnight == null) {
                whiteKnight = loadImage("files/whiteknight.png");
            }
            if (blackKnight == null) {
                blackKnight = loadImage("files/blackknight.png");
            }
            if (whiteRook == null) {
                whiteRook = loadImage("files/whiterook.png");
            }
            if (blackRook == null) {
                blackRook = loadImage("files/blackrook.png");
            }
            if (whiteQueen == null) {
                whiteQueen = loadImage("files/whitequeen.png");
            }
            if (blackQueen == null) {
                blackQueen = loadImage("files/blackqueen.png");
            }
        } catch (IOException e) {
            System.out.println("Internal Error:" + e.getMessage());
        }

        this.pieceSelected = false;
        // creates border around the court area, JComponent method
        setBorder(BorderFactory.createLineBorder(Color.BLACK));

        // Enable keyboard focus on the court area. When this component has the
        // keyboard focus, key events are handled by its key listener.
        setFocusable(true);

        currentGameState = new ChessBoard(); // initializes model for the game
        status = statusInit; // initializes the status JLabel
        currentGameState.calculateAllLegalMoves();
        allCurrentLegalMoves = currentGameState.getAllLegalMoves();
        /*
         * Listens for mouseclicks. Updates the model, then updates the game
         * board based off of the updated model.
         */
        addMouseListener(new MouseAdapter() {
            @Override
            public void mouseReleased(MouseEvent e) {
                Point p = e.getPoint();
                int col = (8 * p.x) / BOARD_HEIGHT;
                int row = 7 - ((p.y * 8) / BOARD_HEIGHT);
                Coordinate finCoordinate = new Coordinate(row, col);
                if (!finCoordinate.isValid()) {
                    pieceSelected = false;
                } else if (pieceSelected) {
                    if (finCoordinate.equals(initialCoordinate)) {
                        pieceSelected = false;
                    } else {
                        Move newMove = null;
                        for (Move move : allCurrentLegalMoves) {
                            if (move.getIntialCoordinate().equals(initialCoordinate)
                                    && move.getFinalCoordinate().equals(finCoordinate)) {
                                newMove = move;
                                break;
                            }
                        }
                        // updates the model given the coordinates of the mouseclick
                        if (newMove != null) {
                            if (newMove.getClass() == Promotion.class) {
                                PieceType type = promotionMenu(newMove.isWhite());
                                ((Promotion) newMove).setNewPieceType(type);
                            }
                            currentGameState.updateState(newMove);
                            currentGameState.calculateAllLegalMoves();
                            allCurrentLegalMoves = currentGameState.getAllLegalMoves();
                            pieceSelected = false;
                            updateStatus(); // updates the status JLabel
                        } else {
                            pieceSelected = false;
                        }

                    }
                    repaint();
                } else {
                    initialCoordinate.setRow(row);
                    initialCoordinate.setColumn(col);
                    if (currentGameState.isTherePieceAtCoordinate(initialCoordinate)) {
                        pieceSelected = true;
                        repaint();
                    }
                }
            }
        });
    }

    /**
     * (Re-)sets the game to its initial state.
     */
    public void reset() {
        currentGameState = new ChessBoard();
        currentGameState.calculateAllLegalMoves();
        allCurrentLegalMoves = currentGameState.getAllLegalMoves();
        pieceSelected = false;
        status.setText("White Turn");
        repaint();

        // Makes sure this component has keyboard/mouse focus
        requestFocusInWindow();
    }

    public void undoMove() {
        if (currentGameState.removeLastMoveTaken()) {
            LinkedList<Move> moves = currentGameState.getAllMovesTaken();
            currentGameState = new ChessBoard();
            for (Move move : moves) {
                currentGameState.updateState(move);
            }
            currentGameState.calculateAllLegalMoves();
            allCurrentLegalMoves = currentGameState.getAllLegalMoves();
            this.updateStatus();
            this.repaint();
        }
    }

    /**
     * Updates the JLabel to reflect the current state of the game.
     */
    private void updateStatus() {
        if (currentGameState.getCurrentPlayer()) {
            status.setText("White Turn");
        } else {
            status.setText("Black Turn");
        }

        int winner = currentGameState.checkMate();
        if (winner == 2) {
            if (currentGameState.getCurrentPlayer()) {
                status.setText("Black wins by checkmate");
            } else {
                status.setText("White wins by checkmate");
            }
        } else if (winner == 1) {
            status.setText("Draw by stalemate");
        } else if (winner == 3) {
            status.setText("Draw for insufficient pieces");
        } else if (winner == 4) {
            status.setText("Draw for the 50-move rule");
        }
        else if(winner == 5){
            status.setText("Draw by repetition");
        }

    }

    private PieceType promotionMenu(boolean white) {
        Object[] options = new Object[4];

        if (white) {
            options[0] = new ImageIcon("files/whitequeen.png");
            options[1] = new ImageIcon("files/whiterook.png");
            options[2] = new ImageIcon("files/whitebishop.png");
            options[3] = new ImageIcon("files/whiteknight.png");
        } else {
            options[0] = new ImageIcon("files/blackqueen.png");
            options[1] = new ImageIcon("files/blackrook.png");
            options[2] = new ImageIcon("files/blackbishop.png");
            options[3] = new ImageIcon("files/blackknight.png");
        }

        int choice = JOptionPane.showOptionDialog(
                null,
                "Choose promotion piece:",
                "Promote Pawn",
                JOptionPane.DEFAULT_OPTION,
                JOptionPane.PLAIN_MESSAGE,
                null,
                options,
                options[0]
        );

        return switch (choice) {
            case 1 -> PieceType.ROOK;
            case 2 -> PieceType.BISHOP;
            case 3 -> PieceType.KNIGHT;
            default -> PieceType.QUEEN;
        };
    }

    /**
     * Returns the size of the game board.
     */
    @Override
    public Dimension getPreferredSize() {
        return new Dimension(BOARD_WIDTH, BOARD_HEIGHT);
    }

    public void drawPiece(Piece piece, Graphics g) {
        int x = (piece.getPosition().getColumn()) * BOARD_HEIGHT / 8;
        int y = (7 - piece.getPosition().getRow()) * BOARD_HEIGHT / 8;

        switch (piece.getPieceType()) {
            case KING:
                if (piece.isWhite()) {
                    g.drawImage(whiteKing, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                } else {
                    g.drawImage(blackKing, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                }
                break;
            case ROOK:
                if (piece.isWhite()) {
                    g.drawImage(whiteRook, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                } else {
                    g.drawImage(blackRook, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                }
                break;
            case BISHOP:
                if (piece.isWhite()) {
                    g.drawImage(whiteBishop, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                } else {
                    g.drawImage(blackBishop, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                }
                break;
            case PAWN:
                if (piece.isWhite()) {
                    g.drawImage(whitePawn, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                } else {
                    g.drawImage(blackPawn, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                }
                break;
            case QUEEN:
                if (piece.isWhite()) {
                    g.drawImage(whiteQueen, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                } else {
                    g.drawImage(blackQueen, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                }
                break;
            case KNIGHT:
                if (piece.isWhite()) {
                    g.drawImage(whiteKnight, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                } else {
                    g.drawImage(blackKnight, x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8, null);
                }
                break;
            default:
                throw new IllegalArgumentException("Illegal piece type");
        }
    }

    @Override
    public void paintComponent(Graphics g) {
        g.drawImage(chessBoardImage, 0, 0, BOARD_WIDTH, BOARD_HEIGHT, null);
        if (pieceSelected) {
            Graphics2D g2 = (Graphics2D) g;
            Color transparentGreen = new Color(0, 255, 0, 150);
            g2.setColor(transparentGreen);

            int x = (initialCoordinate.getColumn()) * BOARD_WIDTH / 8;
            int y = (7 - initialCoordinate.getRow()) * BOARD_HEIGHT / 8;
            g2.fillRect(x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8);
        }
        for (Piece piece : this.currentGameState.getPiecesOnBoard()) {
            drawPiece(piece, g);
            if (piece.getPieceType() == PieceType.KING) {
                King kingPiece = (King) piece;
                ChessBoard chessBoardCopy = new ChessBoard(this.currentGameState);
                if (kingPiece.isInCheck(chessBoardCopy)) {
                    Graphics2D g2 = (Graphics2D) g;
                    Color transparentRed = new Color(255, 0, 0, 150);
                    g2.setColor(transparentRed);

                    int x = (kingPiece.getPosition().getColumn()) * BOARD_WIDTH / 8;
                    int y = (7 - kingPiece.getPosition().getRow()) * BOARD_HEIGHT / 8;
                    g2.fillRect(x, y, BOARD_WIDTH / 8, BOARD_HEIGHT / 8);
                }
            }
        }
        if (pieceSelected) {
            Graphics2D g3 = (Graphics2D) g;
            Color gray = new Color(150, 150, 150, 150);
            g3.setColor(gray);

            int radius = 20;

            for (Move move : allCurrentLegalMoves) {
                if (move.getIntialCoordinate().equals(initialCoordinate)) {
                    int xc = (move.getFinalCoordinate().getColumn()) * BOARD_WIDTH / 8;
                    int yc = (7 - move.getFinalCoordinate().getRow()) * BOARD_HEIGHT / 8;
                    int cx = xc + (BOARD_WIDTH / 8 - radius) / 2; // center inside the square
                    int cy = yc + (BOARD_HEIGHT / 8 - radius) / 2;
                    g3.fillOval(cx, cy, radius, radius);
                }
            }
        }
    }

}
