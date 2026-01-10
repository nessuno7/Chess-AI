import queue
import sys
import os
import grpc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "..", "generated")
sys.path.append(GENERATED_DIR)

import rl_env_pb2 as pb
import rl_env_pb2_grpc as pb_grpc

def request_generator(q: "queue.Queue[pb.StepRequest | None]"):
    while True: #waits for actions_queue.put(), if none closes the requests
        item = q.get()
        if item is None:
            return
        yield item

class RemoteChessEnv:
    def __init__(self, num_envs, host):
        self.num_envs = num_envs
        self.channel = grpc.insecure_channel(host)
        self.stub = pb_grpc.RLEnvStub(self.channel)
        self.actions_queue: "queue.Queue[pb.StepRequest | None]" = queue.Queue()

        self.stream = self.stub.Rollout(request_generator(self.actions_queue))

        dummy_actions = [pb.ProtoMove(from_sq=0, to_sq=0, promotion=0) for _ in range(self.num_envs)]
        self.actions_queue.put(pb.StepRequest(num_envs=self.num_envs, action=dummy_actions))

        continue_ = True
        while continue_:
            try:
                self.current_games = next(self.stream) #always has the current games, code waits until request
                continue_ = False
            except Exception as e:
                print("Failed connection")

    def close(self):
        self.actions_queue.put(None)
        self.channel.close()

    def step(self, actions) -> pb.StepResponse:
        self.actions_queue.put(pb.StepRequest(num_envs=self.num_envs, action=actions))
        self.current_games = next(self.stream)
        return self.current_games

    def reset(self) -> pb.StepResponse:
        dummy = [pb.ProtoMove(from_sq=0, to_sq=0, promotion=0) for _ in range(self.num_envs)]
        return self.step(dummy)
