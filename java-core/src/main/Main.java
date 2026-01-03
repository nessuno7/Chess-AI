package main;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import main.response.ChessBoardPanel;
import main.response.EnvEncoder;
import main.response.RlEnvService;

import javax.swing.*;
import java.awt.*;

class Main
{
    public static void main(String[] args) throws Exception {
        final int numEnv = 10;
        RlEnvService envService = new RlEnvService(numEnv);

        Server server = ServerBuilder.forPort(50051)
                .addService(envService)
                .build()
                .start();
        System.out.println("Server started, listening on " + server.getPort());

        SwingUtilities.invokeLater(() -> {
            JFrame f = new JFrame(numEnv + "Env Chess Games");
            f.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

            JPanel grid = new JPanel(new GridLayout(2, 5, 8, 8)); // 2 rows, 5 cols
            grid.setBorder(BorderFactory.createEmptyBorder(8, 8, 8, 8));
            EnvEncoder[] games = envService.getEnvs();
            for (int i = 0; i < numEnv; i++) {
                grid.add(new ChessBoardPanel(games[i]));
            }

            f.setContentPane(new JScrollPane(grid));
            f.pack();
            f.setLocationRelativeTo(null);
            f.setVisible(true);
        });
        server.awaitTermination();

        System.out.println("Server shutting down...");

    }
}