package main.response;
import io.grpc.Status;
import io.grpc.stub.StreamObserver;
import rl.*;

public class ProblemsEnvService extends ProblemsEnvGrpc.ProblemsEnvImplBase {
    private EnvEncoder[] envs;
    int count = 0;

    public ProblemsEnvService(int numEnv) throws Exception{
        envs = new EnvEncoder[numEnv];

        for (int i = 0; i < numEnv; i++) {
            envs[i] = new EnvEncoder(i);
        }
    }

    public EnvEncoder[] getEnvs(){
        return envs;
    }

    @Override
    public void init(InitRequest request,
                     StreamObserver<StepResponse> responseObserver){

        count++;
        int numEnvs = request.getNumEnvs();
        String fen = request.getFen();
        try{
            for(EnvEncoder env : envs){
                env.reset(fen);
            }
        }
        catch(Exception e){
            System.out.println("Env reset failed at one of the envs");
        }

        System.out.println("Init called: "+count);
        System.out.println("fen = " + fen);
        StepResponse.Builder resp = StepResponse.newBuilder()
                .setNumEnvs(numEnvs);

        try{
            for (int i = 0; i < numEnvs; i++) {
                resp.addStates(envs[i].toEnvStateResponse());
            }
        }
        catch(Exception e){
            System.out.println("Env to response failed at one of the envs");
            throw new IllegalArgumentException("STOP");
        }


        responseObserver.onNext(resp.build());
        responseObserver.onCompleted();
    }

    @Override
    public StreamObserver<StepRequest> step(
            StreamObserver<StepResponse> responseObserver) {

        return new StreamObserver<StepRequest>() {

            @Override
            public void onNext(StepRequest req) {
                count++;
                try {
                    System.out.println("BATCH NUMBER: " + count);
                    System.out.printf("%-10s %-8s %-6s%n", "Index", "Player", "Move");
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

                        if (envs[i].isDone()) {
                            envs[i].reset();
                        }
                        // add encoded state
                        resp.addStates(envs[i].toEnvStateResponse());
                    }
                    responseObserver.onNext(resp.build());
                    System.out.println("----------------------------");
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

            @Override
            public void onError(Throwable t) {
                t.printStackTrace();
            }

            @Override
            public void onCompleted() {
                responseObserver.onCompleted();
            }
        };
    }

    @Override
    public void close(CloseRequest request,
                      StreamObserver<CloseResponse> responseObserver) {

        boolean close = request.getClose();

        System.out.println("Close called: " + close);

        CloseResponse response = CloseResponse.newBuilder()
                .setClose(close)
                .build();

        responseObserver.onNext(response);
        responseObserver.onCompleted();
    }
}

