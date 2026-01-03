package main.response;

import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import main.chess.Chess;
import rl.RLEnvGrpc;
import rl.StepRequest;
import rl.StepResponse;

import javax.swing.*;
import java.awt.*;

public class RlEnvService extends RLEnvGrpc.RLEnvImplBase {
    private EnvEncoder[] envs;
    int count = 0;

    public RlEnvService(int numEnv) throws Exception{
        envs = new EnvEncoder[numEnv];

        for (int i = 0; i < numEnv; i++) {
            envs[i] = new EnvEncoder();
        }
    }

    public EnvEncoder[] getEnvs(){
        return envs;
    }

    @Override
    public StreamObserver<StepRequest> rollout(StreamObserver<StepResponse> responseObserver) {
        return new StreamObserver<>() {
            @Override
            public void onNext(StepRequest req) {
                count++;
                try {
                    System.out.println("Batch number " + count);
                    System.out.printf("%-10s %-8s%n", "Player", "Move");
                    System.out.println("----------------------------");
                    int n = envs.length;

                    StepResponse.Builder resp = StepResponse.newBuilder()
                            .setNumEnvs(n);

                    if (req.getActionCount() != n) {
                        throw new IllegalArgumentException(
                                "Expected " + envs.length + " actions, got " + req.getActionCount()
                        );
                    }

                    for (int i = 0; i < n; i++) {
                        envs[i].step(req.getAction(i));
                        //System.out.println("Server call number " + count + " on env number: " + i);

                        // add encoded state
                        resp.addStates(envs[i].toEnvStateResponse());

                        // optional: auto-reset if done
                        if (envs[i].isDone()) { //TODO: remove resets from the AI
                            envs[i].reset();
                        }
                    }

                    responseObserver.onNext(resp.build());
                } catch (Exception e) {
                    e.printStackTrace();
                    responseObserver.onError(
                            Status.INTERNAL
                                    .withDescription(e.toString())
                                    .withCause(e)
                                    .asRuntimeException()
                    );
                }
            }

            @Override public void onError(Throwable t) {}
            @Override public void onCompleted() { responseObserver.onCompleted(); }
        };
    }
}

