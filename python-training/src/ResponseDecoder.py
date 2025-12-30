import sys
import numpy as np
import os
import torch
from typing import Tuple

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "..", "generated")

sys.path.append(GENERATED_DIR)

import rl_env_pb2 as pb
import rl_env_pb2_grpc as pb_grpc

OBS_DIM = 768
PROMO_V = 5
N_SQ = 64


class ResponseDecoder:
    def __init__(self, response: pb.StepResponse, device):
        self.num_envs = response.num_envs
        self.envs = response.states
        self.counter = 0
        self.device = device

    def hasNext(self) ->bool:
        return self.counter < self.num_envs

    def next(self) -> pb.EnvState:
        if self.hasNext():
            self.counter = self.counter + 1
            return self.envs[self.counter-1]
        else:
            raise StopIteration("ResponseDecoderIterator called next when next does not exist")

    def getEnv(self, index):
        if 0 < index < self.num_envs:
            return self.envs[index]
        else:
            raise IndexError("ResponseDecoder index out of bounds")

    def getEnvObs(self, index):
        if 0 < index < self.num_envs:
            return np.asarray(self.envs[index].obs, dtype=np.float32)
        else:
            raise IndexError("ResponseDecoder index out of bounds")

    def getLegalMovesTensors(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        max_dim = max((len(state.moves) for state in self.envs), default=0)

        moves_from = torch.zeros((self.num_envs, max_dim), dtype=torch.long, device=self.device)
        moves_to = torch.zeros((self.num_envs, max_dim), dtype=torch.long, device=self.device)
        moves_promo = torch.zeros((self.num_envs, max_dim), dtype=torch.long, device=self.device)
        mask = torch.zeros((self.num_envs, max_dim), dtype=torch.bool, device=self.device)

        for i, state  in enumerate(self.envs):
            for k, m in enumerate(state.moves):
                moves_from[i, k] = m.from_sq
                moves_to[i, k] = m.to_sq
                moves_promo[i, k] = m.promotion
                mask[i, k] = True

        return moves_from, moves_to, moves_promo, mask

    def getObservationMatrixTensor(self)-> torch.Tensor:
        obs = torch.empty(
            (self.num_envs, OBS_DIM),
            dtype=torch.float32,
            device=self.device
        )

        for i, state in enumerate(self.envs):
            obs[i] = torch.as_tensor(
                list(state.obs.observationPoints),   # <-- materialize gRPC repeated
                dtype=torch.float32,
                device=self.device
            )

        return obs

    def getRewardTensor(self) -> torch.Tensor:
        rewards = torch.empty(
            (self.num_envs,),
            dtype=torch.float32,
            device=self.device
        )

        for i, state in enumerate(self.envs):
            rewards[i] = state.reward

        return rewards

    def getWhiteTensor(self ) -> torch.Tensor:
        whites = torch.empty(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device
        )

        for i, state in enumerate(self.envs):
            whites[i] = bool(state.size)

        return whites

    def getDoneTensor(self) -> torch.Tensor:
        dones = torch.empty(
            (self.num_envs,),
            dtype=torch.bool,
            device=self.device
        )

        for i, state in enumerate(self.envs):
            dones[i] = state.done

        return dones

    def getNextResets(self) -> list:
        list_resets = []
        for env in self.envs:
            list_resets.append(env.reward == -0.5 or env.reward == 0.0)
        return list_resets



