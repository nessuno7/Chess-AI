package main;

import io.grpc.Server;
import io.grpc.ServerBuilder;
import main.response.RlEnvService;

class Main
{
    public static void main(String[] args) throws Exception {
        Server server = ServerBuilder.forPort(50051)
                .addService(new RlEnvService(10))
                .build()
                .start();
        System.out.println("Server started, listening on " + server.getPort());
        server.awaitTermination();

        System.out.println("Server shutting down...");

    }
}