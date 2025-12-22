package main.chess;

import javax.swing.*;
import java.awt.*;

public class RunChess implements Runnable {
    public void run() {
        // NOTE: the 'final' keyword denotes immutability even for local variables.

        // Top-level frame in which game components live
        final JFrame frame = new JFrame("Chess");
        frame.setLocation(300, 300);

        // Status panel
        final JPanel status_panel = new JPanel();
        frame.add(status_panel, BorderLayout.SOUTH);
        final JLabel status = new JLabel("Setting up...");
        status_panel.add(status);

        // Game board
        final Chess board = new Chess(status);
        frame.add(board, BorderLayout.CENTER);

        // Reset button
        final JPanel control_panel = new JPanel();
        frame.add(control_panel, BorderLayout.NORTH);

        // Note here that when we add an action listener to the reset button, we
        // define it as an anonymous inner class that is an instance of
        // ActionListener with its actionPerformed() method overridden. When the
        // button is pressed, actionPerformed() will be called.
        final JButton reset = new JButton("Reset");
        reset.addActionListener(e -> board.reset());
        control_panel.add(reset);

        final JButton undo = new JButton("Undo move");
        undo.addActionListener(e -> board.undoMove());
        control_panel.add(undo);

        final JButton instructionsButton = new JButton("Game Instructions");
        instructionsButton.addActionListener(e -> {
            String message = "This game is chess\n" +
                    "Click the piece you want to move, a green square" +
                    "will appear on the Piece that has been selected\n" +
                    "gray circles will appear on the squares where the " +
                    "piece can move to, showing the possible moves\n" +
                    "the user can take\n" +
                    "The turn of the player is shown at the bottom," +
                    "if the king is in check it will be marked red and \n" +
                    "if the game ends n checkmate or stalemate it will be shown at the bottom";
            JOptionPane.showMessageDialog(
                    frame, message, "Game Instructions", JOptionPane.INFORMATION_MESSAGE
            );
        });
        control_panel.add(instructionsButton);

        // Put the frame on the screen
        frame.pack();
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        frame.setVisible(true);

        // Start the game
        board.reset();
    }
}
