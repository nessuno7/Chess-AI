import sys
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "..", "generated")

sys.path.append(GENERATED_DIR)

import rl_env_pb2 as pb
import rl_env_pb2_grpc as pb_grpc

OBS_DIM = 768
PROMO_V = 5
N_SQ = 64
ACTION_DIM = N_SQ * N_SQ #64*64, promotions are handled separately through the promo handler



def moveToIndex(m: pb.ProtoMove):
    if m.promotion != 0:
        index = 4095 + ((m.from_sq%8)*2 + m.to_sq)*4 + m.promotion
    else:
        index = (m.from_sq * N_SQ) + m.to_sq

    return index

def indexToMove(index):
    if index > 4183:
        raise Exception("Index out of range")

    if index > 4095: #promotion move
        rest = index - 4095
        promotion = rest%4
        if promotion == 0:
            promotion = 4
        rest2 = (rest-promotion)/4
        from_h = int((rest2+1)/3)
        from_sq = 8+from_h
        to_sq = rest2-from_h*2
    else:
        to_sq = index%N_SQ
        from_sq = ((index-to_sq)/N_SQ)%N_SQ
        promotion = 0
    return pb.ProtoMove(from_sq=from_sq, to_sq=to_sq, promotion=promotion)

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

    def getLegalMovesMatrixMask(self):
        mask = np.zeros((self.response.num_envs, ACTION_DIM), dtype=np.bool_)
        for i, state  in enumerate(self.response.states):
            for m in state.moves:
                j = moveToIndex(m)
                if 0 <= j < ACTION_DIM:
                    mask[i][j] = True

        return mask

    def getObservationMatrix(self):
        obs = np.zeros((self.response.num_envs,OBS_DIM), dtype=np.float32)
        for i, state in enumerate(self.response.states):
            obs[i] = state.obs.observationPoints

        return obs

    def getRewardArray(self):
        rewards = np.zeros(self.response.num_envs, dtype=np.float32)
        for i, state in enumerate(self.response.states):
            rewards[i] = state.reward

    def getWhiteArray(self ):
        whites = np.zeros(self.response.num_envs, dtype=np.bool)
        for i, state in enumerate(self.response.states):
            whites[i] = state.size
        return whites

    def getDoneArray(self):
        dones = np.zeros(self.response.num_envs, dtype=np.bool)
        for i, state in enumerate(self.response.states):
            dones[i] = state.done
        return dones



