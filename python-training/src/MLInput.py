import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "..", "generated")

sys.path.append(GENERATED_DIR)

import rl_env_pb2 as pb
import rl_env_pb2_grpc as pb_grpc

PROMO_V = 5
N_SQ = 64
ACTION_DIM = N_SQ * N_SQ * PROMO_V #64*64 + all possible promotions that ca only be from 2nd or 7th row to 1st or 8th * 4(possible pormotions)

def moveToIndex(m: pb.ProtoMove):
    return ((int(m.from_sq) * N_SQ) + int(m.to_sq)) * PROMO_V + int(m.promotion)

def indextoMove(index):
    promo = index%PROMO_V
    to_sq = ((index-promo)/PROMO_V)%N_SQ
    from_sq = ((((index-promo)/PROMO_V)-to_sq)/N_SQ)%N_SQ
    return pb.ProtoMove(from_sq=from_sq, to_sq=to_sq, promotion=promo)


class MLInput:
    def __init__(self, state: pb.EnvState):
        self.state = state

    def getReward(self) -> float:
        return self.state.reward

    def isDone(self) -> bool:
        return self.state.done

    def isWhite(self) ->bool:
        return self.state.white

    def getObservation(self) -> list:
        return self.state.obs.observationPoints

    def getAllLegalMoves(self) -> list:
        return self.state.moves

    def legalMovesMask(self) -> np.ndarray: #maybe make it into matrices
        mask = np.zeros(ACTION_DIM, dtype=np.bool_)
        for m in self.state.moves:
            j = moveToIndex(m)
            if 0 <= j < ACTION_DIM:
                mask[j] = True
        return mask



