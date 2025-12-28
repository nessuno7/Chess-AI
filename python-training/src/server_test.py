import queue
import sys
import time
import random
import grpc
import os

# so Python can import generated/rl_env_pb2*.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "..", "generated")

sys.path.append(GENERATED_DIR)

import rl_env_pb2 as pb
import rl_env_pb2_grpc as pb_grpc


def request_generator(q: "queue.Queue[pb.StepRequest]"):
    """Yields StepRequest objects to the gRPC stream until None is received."""
    while True:
        item = q.get()
        if item is None:
            return
        yield item


def pick_random_legal(env_state: pb.EnvState) -> pb.ProtoMove:
    if len(env_state.moves) == 0:
        return pb.ProtoMove(from_sq=0, to_sq=0, promotion=0)
    return random.choice(env_state.moves)


def main():
    NUM_ENVS = 10

    channel = grpc.insecure_channel("localhost:50051")
    stub = pb_grpc.RLEnvStub(channel)

    q: queue.Queue[pb.StepRequest | None] = queue.Queue()

    # Start the bidirectional stream: iterator in, iterator out
    responses = stub.Rollout(request_generator(q))

    # Send an initial request (dummy actions)
    dummy_actions = [pb.ProtoMove(from_sq=0, to_sq=0, promotion=0) for _ in range(NUM_ENVS)]
    q.put(pb.StepRequest(num_envs=NUM_ENVS, action=dummy_actions))

    # Read the first response
    resp = next(responses)
    print("Connected. num_envs =", resp.num_envs)

    # Now do a few steps
    for t in range(20):
        actions = [pick_random_legal(env) for env in resp.states]
        q.put(pb.StepRequest(num_envs=NUM_ENVS, action=actions))

        resp = next(responses)

        r0 = resp.states[0].reward
        d0 = resp.states[0].done
        lm0 = len(resp.states[0].moves)
        print(f"t={t:02d} env0 reward={r0:.3f} done={d0} legal_moves={lm0}")

    # Close stream + channel
    q.put(None)  # signals generator to end
    channel.close()
    print("Done.")


if __name__ == "__main__":
    main()