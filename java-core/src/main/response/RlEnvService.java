package main.response;

import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import rl.RLEnvGrpc;
import rl.StepRequest;
import rl.StepResponse;

public class RlEnvService extends RLEnvGrpc.RLEnvImplBase {
    private EnvEncoder[] envs;

    public RlEnvService(int numEnv) throws Exception{
        envs = new EnvEncoder[numEnv];

        for (int i = 0; i < numEnv; i++) {
            envs[i] = new EnvEncoder();
        }
    }

    @Override
    public StreamObserver<StepRequest> rollout(StreamObserver<StepResponse> responseObserver) {
        return new StreamObserver<>() {
            @Override
            public void onNext(StepRequest req) {
                System.out.println("Received StepRequest with "
                        + req.getActionCount() + " actions");
                try {
                    int n = envs.length;

                    StepResponse.Builder resp = StepResponse.newBuilder()
                            .setNumEnvs(n);

                    if (req.getActionCount() != n) {
                        throw new IllegalArgumentException(
                                "Expected " + envs.length + " actions, got " + req.getActionCount()
                        );
                    }

                    for (int i = 0; i < n; i++) {
                        // apply action for env i
                        envs[i].step(req.getAction(i));

                        // add encoded state
                        resp.addStates(envs[i].toEnvStateResponse());

                        // optional: auto-reset if done
                        if (envs[i].isDone()) {
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

