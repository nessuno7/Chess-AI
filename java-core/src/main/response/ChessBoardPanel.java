package main.response;

import main.chess.Chess;
import main.chess.Coordinate;
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
import java.beans.PropertyChangeListener;
import java.io.IOException;
import java.io.InputStream;
import java.util.LinkedList;

public class ChessBoardPanel extends JPanel {
    EnvEncoder currentGameState;
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

    private final PropertyChangeListener listener = evt ->
            SwingUtilities.invokeLater(this::repaint);
    // Game constants
    public final int BOARD_WIDTH = 200;
    public final int BOARD_HEIGHT = 200;

    /**
     * Initializes the game board.
     */
    private static BufferedImage loadImage(String path) throws IOException {
        InputStream is = Chess.class.getClassLoader().getResourceAsStream(path);

        if (is == null) {
            throw new IllegalArgumentException("Image not found: " + path);
        }
        return ImageIO.read(is);
    }

    public ChessBoardPanel(EnvEncoder chessBoard) {
        this.currentGameState = chessBoard;

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

        // creates border around the court area, JComponent method
        setBorder(BorderFactory.createLineBorder(Color.BLACK));

        // Enable keyboard focus on the court area. When this component has the
        // keyboard focus, key events are handled by its key listener.

        /*
         * Listens for mouseclicks. Updates the model, then updates the game
         * board based off of the updated model.
         */

        currentGameState.addChangeListener(listener);
    }

    /*public void loadEnvEncoder(ChessBoard chessBoard){
        this.currentGameState = new ChessBoard(chessBoard);
        repaint();
    }*/

    /**
     * (Re-)sets the game to its initial state.
     */
    public void reset(EnvEncoder chessBoard) {
        if (this.currentGameState != null) {
            this.currentGameState.removeChangeListener(listener);
        }
        currentGameState = chessBoard;
        currentGameState.addChangeListener(evt -> {
            // ensure repaint runs on Swing's Event Dispatch Thread
            SwingUtilities.invokeLater(this::repaint);
        });
        currentGameState.firePropertyChange();
    }

    /**
     * Updates the JLabel to reflect the current state of the game.
     */



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
        super.paintComponent(g);
        g.drawImage(chessBoardImage, 0, 0, BOARD_WIDTH, BOARD_HEIGHT, null);

        for (Piece piece : this.currentGameState.getChessBoard().getPiecesOnBoard()) {
            drawPiece(piece, g);
        }
    }

}
