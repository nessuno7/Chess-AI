import sys
import grpc

from generated import rl_env_pb2 as pb
from generated import rl_env_pb2_grpc as pb_grpc

sys.path.append("generated")

class RemoteChessEnv:
    def __init__(self, host="localhost:50051", num_envs=10):
        self.num_envs = num_envs
        self.channel = grpc.insecure_channel(host)
        self.stub = pb_grpc.RLEnvStub(self.channel)
        self.stream = self.stub.Rollout()
        self.last_resp = None

    def close(self):
        try:
            self.stream.done_writing()
        except Exception:
            pass
        self.channel.close()

    def step(self, actions):
        """
        actions: list[pb.ProtoMove] length = num_envs
        returns: pb.StepResponse
        """
        req = pb.StepRequest(num_envs=self.num_envs, action=actions)
        self.stream.write(req)
        self.last_resp = next(self.stream)
        return self.last_resp
