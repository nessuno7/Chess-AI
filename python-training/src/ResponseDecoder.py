import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "..", "generated")

sys.path.append(GENERATED_DIR)

import rl_env_pb2 as pb
import rl_env_pb2_grpc as pb_grpc


class ResponseDecoder:
    def __init__(self, response: pb.StepResponse):
        self.response = response
        self.counter = 0

    def hasNext(self) ->bool:
        return self.counter < self.response.num_envs

    def next(self) -> pb.EnvState:
        if self.hasNext():
            self.counter = self.counter + 1
            return self.response.steps[self.counter-1]
        else:
            raise StopIteration("ResponseDecoderIterator called next when next does not exist")

    def get(self, index):
        if 0 < index < self.response.num_envs:
            return self.response.steps[index]
        else:
            raise IndexError("ResponseDecoder index out of bounds")




