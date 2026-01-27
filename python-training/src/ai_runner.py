import sys
from dataclasses import dataclass
from typing import List, Tuple
import os

import numpy as np
import torch
import torch.nn as nn

# --- import generated protobuf stubs ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GENERATED_DIR = os.path.join(BASE_DIR, "..", "generated")
sys.path.append(GENERATED_DIR)

import rl_env_pb2 as pb
import rl_env_pb2_grpc as pb_grpc

# =========================
# Remote env wrapper and message decoder (bidirectional streaming using request iterator)
# ========================
import ResponseDecoder as rs
from RemoteChessEnv import RemoteChessEnv


# =========================
# Actor-Critic: score legal moves + value(obs)
# =========================
class ActorCritic(nn.Module):
    """
    Actor: scores each legal move (variable-length list) using (obs + move features)
    Critic: value(obs)
    """
    def __init__(self, obs_dim=rs.OBS_DIM, hidden=256, sq_embed=16, promo_embed=4):
        super().__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )

        self.sq_emb = nn.Embedding(64, sq_embed) # one embedding for each square
        self.promo_emb = nn.Embedding(5, promo_embed)  # 0..4

        # actor head outputs scalar logit per move
        self.actor_head = nn.Linear(hidden + 2*sq_embed + promo_embed, 1)

        # critic head outputs scalar value per state
        self.critic_head = nn.Linear(hidden, 1)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        h = self.obs_net(obs)
        return self.critic_head(h).squeeze(-1)  # [B]

    def move_logits(
            self,
            obs: torch.Tensor,            # [B, obs_dim] # planes observation
            moves_from: torch.Tensor,     # [B, K] include all the initial coordinate of the legal moves
            moves_to: torch.Tensor,       # [B, K] includes all the final coordinates of the legal moves
            moves_promo: torch.Tensor,    # [B, K] include all the final promo coordinate (0 if no promotion)
    ) -> torch.Tensor:
        """
        Returns logits over the K moves for each batch item: [B, K]
        """
        B, K = moves_from.shape
        h = self.obs_net(obs)  # [B, hidden]
        h = h.unsqueeze(1).expand(B, K, h.shape[-1])  # [B, K, hidden]

        ef = self.sq_emb(moves_from)      # [B, K, sq_embed]
        et = self.sq_emb(moves_to)        # [B, K, sq_embed]
        ep = self.promo_emb(moves_promo)  # [B, K, promo_embed]

        x = torch.cat([h, ef, et, ep], dim=-1)        # [B, K, ...]
        logits = self.actor_head(x).squeeze(-1)       # [B, K]
        return logits


# =========================
# Utilities: pad legal moves, sample only legal moves
# =========================

@torch.no_grad()
def select_actions(
        model: ActorCritic,
        decoder: rs.ResponseDecoder,
) -> Tuple[List[pb.ProtoMove], torch.Tensor, torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Returns:
      chosen_moves: list[ProtoMove] length B
      chosen_idx: tensor [B] index into padded legal move list
      logprob: tensor [B]
      value: tensor [B]
      padded_move_tensors: (moves_from, moves_to, moves_promo, mask) for later PPO update
    """

    obs = decoder.getObservationMatrixTensor()
    moves_from, moves_to, moves_promo, mask = decoder.getLegalMovesTensors()

    logits = model.move_logits(obs, moves_from, moves_to, moves_promo)  # [B, K]
    logits = logits.masked_fill(~mask, -1e9)

    dist = torch.distributions.Categorical(logits=logits)
    idx = dist.sample()  # [B]
    logprob = dist.log_prob(idx)  # [B]
    value = model.value(obs)      # [B]

    # Convert idx -> actual ProtoMove
    chosen_moves = []
    for b in range(decoder.num_envs):
        k = idx[b].item()
        fr = moves_from[b, k].item()
        to = moves_to[b, k].item()
        pr = moves_promo[b, k].item()   # 0 = no promo (depending on your encoding)

        chosen_moves.append((fr, to, pr))

    return chosen_moves, idx, logprob, value, (moves_from, moves_to, moves_promo, mask)


# =========================
# PPO storage + advantage calculation
# =========================

@dataclass
class RolloutBatch:
    obs: torch.Tensor                 # [T*B, obs_dim]
    moves_from: torch.Tensor          # [T*B, Kmax_global] padded
    moves_to: torch.Tensor
    moves_promo: torch.Tensor
    mask: torch.Tensor                # [T*B, Kmax_global] bool
    action_idx: torch.Tensor          # [T*B]
    old_logprob: torch.Tensor         # [T*B]
    returns: torch.Tensor             # [T*B]
    advantage: torch.Tensor           # [T*B]
    old_value: torch.Tensor           # [T*B]
    valid: torch.Tensor


def compute_gae_unsupervised(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
    """
    rewards: [T, B]
    dones:   [T, B] bool
    values:  [T, B]
    next_value: [B]
    returns advantages: [T, B], returns: [T, B]
    """
    T, B = rewards.shape
    adv = torch.zeros((T, B), dtype=torch.float32, device=values.device)
    last_gae = torch.zeros((B,), dtype=torch.float32, device=values.device)

    for t in reversed(range(T)):
        nonterminal = (~dones[t]).float()
        next_v = next_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last_gae = delta + gamma * lam * nonterminal * last_gae
        adv[t] = last_gae

    returns = adv + values
    return adv, returns
"""
def compute_gae_supervised(rewards, valids, dones, values, next_value, gamma=0.99, lam=0.95):
    #dones_shifted = torch.roll(dones, shifts=1, dims=0) #shift all envs by one step
    dones_shifted = torch.zeros_like(valids)
    dones_shifted[1:] = valids[:-1]
    valid = ~(dones_shifted & valids) #~dones_shifted implies dones logically equivalent to valid

    T, B = rewards.shape

    adv = torch.zeros((T, B), dtype=torch.float32, device=values.device)
    last_gae = torch.zeros((B,), dtype=torch.float32, device=values.device)

    for t in reversed(range(T)):
        nonterminal = (~dones[t]).float()
        next_v = next_value if t == T - 1 else values[t + 1] #if it is the last index it gets the last value from prev batch
        delta = rewards[t] + gamma * next_v * nonterminal - values[t]
        last_gae = delta + gamma * lam * last_gae * nonterminal
        vmask = valid[t].float()
        adv[t] = last_gae * vmask

    returns = (adv + values) * valid.float()
    return adv, returns, valid
"""

def compute_gae_supervised(rewards, valids, dones, values, next_value, gamma=0.99, lam=0.95):
    T, B = rewards.shape
    device = values.device

    adv = torch.zeros((T, B), dtype=torch.float32, device=device)
    last_gae = torch.zeros((B,), dtype=torch.float32, device=device)

    for t in reversed(range(T)):
        vmask = valids[t].float()  # whether this step exists

        # If this step is invalid/padding, reset recursion for those env lanes
        last_gae = last_gae * vmask

        # Determine whether we are allowed to bootstrap/propagate beyond step t
        if t == T - 1:
            next_v = next_value
            next_valid = torch.ones_like(vmask)  # no t+1 inside buffer
        else:
            next_v = values[t + 1]
            next_valid = valids[t + 1].float()

        nonterminal = (~dones[t]).float()

        # Only bootstrap into next state if:
        # - current step is nonterminal, and
        # - the next step is valid (same puzzle segment)
        cont = nonterminal * next_valid

        delta = rewards[t] + gamma * next_v * cont - values[t]
        last_gae = delta + gamma * lam * cont * last_gae

        adv[t] = last_gae * vmask  # zero for invalid steps

    returns = (adv + values) * valids.float()
    return adv, returns


# ===============================
# UNSUPERVISED LEARNING PPO LOOP
# ===============================

def ppo_train_unsupervised(
        host="localhost:50051",
        num_envs=10,
        total_updates=200,
        rollout_steps=128,      # T
        minibatch_size=256,
        ppo_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        lr=3e-4,
        device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = RemoteChessEnv(num_envs=num_envs, host=host)
    model = ActorCritic(obs_dim=rs.OBS_DIM).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = "unsupervised_checkpoint.pth"
    start_update = 0

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optim.load_state_dict(checkpoint["optimizer_state"])

        start_update = int(checkpoint.get("update", 0))
        print(f"Loaded checkpoint. Already trained updates={start_update}")
    else:
        print("Training from scratch for first startup")
        #raise FileNotFoundError("Could not find checkpoint file")

    # Prime env (get first response)
    resp = rs.ResponseDecoder(env.reset(), device)

    end_update = start_update + total_updates
    for update in range(start_update, end_update):
        # --- Collect rollout of length T across B envs ---
        obs_buf = []
        rewards_buf = []
        dones_buf = []
        values_buf = []
        logprob_buf = []
        action_idx_buf = []

        # store per-step padded legal sets (variable K each step), we'll repad later
        moves_from_list = []
        moves_to_list = []
        moves_promo_list = []
        mask_list = []

        for t in range(rollout_steps):
            #appends observation from previous state
            obs_t = resp.getObservationMatrixTensor()

            #chooses actions
            chosen_moves, action_id, logprob, value, (mf, mt, mp, msk) = select_actions(
                model, resp
            )

            # data for step k
            obs_buf.append(obs_t)  # [B, obs_dim]
            values_buf.append(value)
            logprob_buf.append(logprob)
            action_idx_buf.append(action_id)
            moves_from_list.append(mf)
            moves_to_list.append(mt)
            moves_promo_list.append(mp)
            mask_list.append(msk)

            #now step k+1
            resp = rs.ResponseDecoder(env.step(chosen_moves), device=device)

            #rewards for step (k+1)-1
            rewards_t = resp.getRewardTensor()
            rewards_buf.append(rewards_t)

            dones_t = (rewards_t == 1) | (rewards_t == -0.02)
            dones_buf.append(dones_t)

        # bootstrap value for GAE
        with torch.no_grad():
            next_obs = resp.getObservationMatrixTensor()
            next_value = model.value(next_obs)  # [B]

        rewards_t = torch.stack(rewards_buf, dim=0)  # [T, B]
        dones_t = torch.stack(dones_buf, dim=0)      # [T, B]
        values_t = torch.stack(values_buf, dim=0)    # [T, B]

        adv_t, ret_t = compute_gae_unsupervised(rewards_t, dones_t, values_t, next_value, gamma=gamma, lam=gae_lambda)

        # Flatten T,B -> N
        T, B = rollout_steps, num_envs
        obs_flat = torch.cat(obs_buf, dim=0)                 # [T*B, obs_dim]
        old_logprob_flat = torch.cat(logprob_buf, dim=0)     # [T*B]
        old_value_flat = torch.cat(values_buf, dim=0)        # [T*B]
        action_idx_flat = torch.cat(action_idx_buf, dim=0)   # [T*B]
        adv_flat = adv_t.reshape(T*B)
        ret_flat = ret_t.reshape(T*B)

        # Advantage normalization (common PPO trick)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        # --- Re-pad move tensors to a single Kmax across the whole rollout ---
        Kmax_global = max(m.shape[1] for m in moves_from_list)

        def repad(m_list, fill=0, dtype=torch.long):
            out = []
            for m in m_list:
                if m.shape[1] == Kmax_global:
                    out.append(m)
                else:
                    pad = torch.full((m.shape[0], Kmax_global - m.shape[1]), fill_value=fill, dtype=dtype, device=device)
                    out.append(torch.cat([m, pad], dim=1))
            return torch.cat(out, dim=0)  # [T*B, Kmax_global]

        moves_from_flat = repad(moves_from_list, fill=0, dtype=torch.long)
        moves_to_flat = repad(moves_to_list, fill=0, dtype=torch.long)
        moves_promo_flat = repad(moves_promo_list, fill=0, dtype=torch.long)

        def repad_mask(msk_list):
            out = []
            for msk in msk_list:
                if msk.shape[1] == Kmax_global:
                    out.append(msk)
                else:
                    pad = torch.zeros((msk.shape[0], Kmax_global - msk.shape[1]), dtype=torch.bool, device=device)
                    out.append(torch.cat([msk, pad], dim=1))
            return torch.cat(out, dim=0)

        mask_flat = repad_mask(mask_list)  # [T*B, Kmax_global]

        batch = RolloutBatch(
            obs=obs_flat,
            moves_from=moves_from_flat,
            moves_to=moves_to_flat,
            moves_promo=moves_promo_flat,
            mask=mask_flat,
            action_idx=action_idx_flat,
            old_logprob=old_logprob_flat,
            returns=ret_flat,
            advantage=adv_flat,
            old_value=old_value_flat,
            valid=None
        )

        # --- PPO update ---
        N = T * B
        idxs = np.arange(N)

        for epoch in range(ppo_epochs):
            np.random.shuffle(idxs)

            for start in range(0, N, minibatch_size):
                mb = idxs[start:start + minibatch_size]
                mb = torch.tensor(mb, dtype=torch.long, device=device)

                obs_mb = batch.obs[mb]                # [M, obs_dim]
                mf_mb = batch.moves_from[mb]          # [M, K]
                mt_mb = batch.moves_to[mb]
                mp_mb = batch.moves_promo[mb]
                msk_mb = batch.mask[mb]               # [M, K]
                act_mb = batch.action_idx[mb]         # [M]
                old_lp_mb = batch.old_logprob[mb]     # [M]
                adv_mb = batch.advantage[mb]          # [M]
                ret_mb = batch.returns[mb]            # [M]
                old_v_mb = batch.old_value[mb]        # [M]

                # New policy logits over the SAME legal move set
                logits = model.move_logits(obs_mb, mf_mb, mt_mb, mp_mb)  # [M, K]
                logits = logits.masked_fill(~msk_mb, float("-inf"))
                dist = torch.distributions.Categorical(logits=logits)

                new_lp = dist.log_prob(act_mb)  # [M]
                entropy = dist.entropy().mean()

                ratio = (new_lp - old_lp_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (optionally clip like PPO tutorial)
                new_v = model.value(obs_mb)
                v_clipped = old_v_mb + (new_v - old_v_mb).clamp(-clip_eps, clip_eps)
                v_loss1 = (new_v - ret_mb).pow(2)
                v_loss2 = (v_clipped - ret_mb).pow(2)
                value_loss = 0.5 * torch.max(v_loss1, v_loss2).mean()

                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optim.step()

        # Logging
        mean_reward = rewards_t.mean().item()
        mean_done = dones_t.float().mean().item()
        print(f"update={update:04d} mean_step_reward={mean_reward:+.4f} done_rate={mean_done:.3f} Kmax={Kmax_global}")

    #save and close the model
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "update": end_update,   # total number of updates trained so far
            "lr": lr,
        },
        ckpt_path
    )
    print(f"Saved checkpoint to {ckpt_path} at update={end_update}.")
    env.close()



# ===================================
#           SUPERVISED LEARNING
# ===================================

from supervised_learning.lichessProblemsFetcher import ProblemIterator, intToMove, moveToInt
from supervised_learning.ProblemEnv import RemoteProblemsEnv


def ppo_train_supervised(
        host="localhost:50051",
        num_envs=10,
        total_updates=200,
        numberProblemAttempts=40,      # attempts 40 different problemsper batch
        minibatch_size=256,
        ppo_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_eps=0.2,
        vf_coef=0.5,
        ent_coef=0.01,
        lr=3e-4,
        device=None,
        skip_n_problems = 0
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = RemoteProblemsEnv(num_envs=num_envs, host=host)
    model = ActorCritic(obs_dim=rs.OBS_DIM).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    iterator = ProblemIterator(skip=skip_n_problems)

    ckpt_path = "supervised_checkpoint.pth"
    start_update = 0

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optim.load_state_dict(checkpoint["optimizer_state"])

        start_update = int(checkpoint.get("update", 0))
        print(f"Loaded checkpoint. Already trained updates={start_update}")
    else:
        print("Training from scratch for first startup")
        #raise FileNotFoundError("Could not find checkpoint file")



    end_update = start_update + total_updates
    for update in range(start_update, end_update):
        # --- Collect rollout of length T across B envs ---
        obs_buf = []
        rewards_buf = []
        dones_buf = []
        valid_buf = []
        values_buf = []
        logprob_buf = []
        action_idx_buf = []

        # store per-step padded legal sets (variable K each step), we'll repad later
        moves_from_list = []
        moves_to_list = []
        moves_promo_list = []
        mask_list = []


        for nProb in range(numberProblemAttempts): #loop pver the number of porblme you want to solve per batch

            puzzleIndex, fen, puzzleMoves, puzzleRating = next(iterator)
            #env.startMessage(fen) TODO: initMessage, will get Step Response as answer
            # Prime env (get first response)
            resp = rs.ResponseDecoder(env.reset_with_new_fen(fen), device)
            continue_rewards =[True for _ in range(num_envs)]

            for t in range(len(puzzleMoves)): #each step is one problem attempted to solve

                #appends observation from previous state
                obs_t = resp.getObservationMatrixTensor()

                #chooses actions
                chosen_moves, action_id, logprob, value, (mf, mt, mp, msk) = select_actions(
                    model, resp
                )

                print(chosen_moves)
                # data for step k
                obs_buf.append(obs_t)  # [B, obs_dim]
                values_buf.append(value)
                logprob_buf.append(logprob)
                action_idx_buf.append(action_id)
                moves_from_list.append(mf)
                moves_to_list.append(mt)
                moves_promo_list.append(mp)
                mask_list.append(msk)

                rewards_t = []
                valid_t = []

                for i in range(num_envs):
                    if continue_rewards[i]:
                        if intToMove(square_from = chosen_moves[i][0], square_to = chosen_moves[i][1], promotion=chosen_moves[i][2]) == puzzleMoves[t]:
                            difficulty = min(1, puzzleRating/2500) #rewards increase linearly with difficulty () capped at 2000
                            rewards_t.insert(i, (0.8*difficulty) +0.2) #0.1 min reward for easy puzzles, 1 max reward for hard puzzles
                            valid_t.insert(i, False)
                        elif moveToInt(puzzleMoves[t])[0] == chosen_moves[i][0]: #TODO: introduce partial credit
                            valid_t.insert(i, True)
                            rewards_t.insert(i, 0.0)
                            continue_rewards[i] = False
                        else: #TODO: give negative reward for not correct move and inversely proportion to distance from correct state at that move
                            valid_t.insert(i, True)
                            rewards_t.insert(i, -0.5)
                            continue_rewards[i] = False
                    else:
                        rewards_t.insert(i, 0)
                        valid_t.insert(i, True)

                rewards_buf.append(rewards_t)
                valid_buf.append(valid_t)

                if t == len(puzzleMoves) - 1:
                    dones_t = [False for _ in range(num_envs)]
                else:
                    dones_t = [True for _ in range(num_envs)]

                dones_buf.append(dones_t)

                #now step k+1
                resp = rs.ResponseDecoder(env.step(chosen_moves), device=device)

        # bootstrap value for GAE
        with torch.no_grad():
            next_obs = resp.getObservationMatrixTensor()
            next_value = model.value(next_obs)  # [B]

        rewards_t = torch.stack(rewards_buf, dim=0)  # [T, B]
        valid_t = torch.stack(valid_buf, dim=0)      # [T, B]
        values_t = torch.stack(values_buf, dim=0)    # [T, B]
        dones_t = torch.stack(dones_buf, dim=0)

        adv_t, ret_t = compute_gae_supervised(rewards=rewards_t, valids=valid_t, values=values_t, dones=dones_t, next_value=next_value)

        # Flatten T,B -> N
        T, B = rewards_t.shape() #TODO: check len(rewards_t) is correct
        valid_flat = valid_t.reshape(T * B).float()
        obs_flat = torch.cat(obs_buf, dim=0)                 # [T*B, obs_dim]
        old_logprob_flat = torch.cat(logprob_buf, dim=0)     # [T*B]
        old_value_flat = torch.cat(values_buf, dim=0)        # [T*B]
        action_idx_flat = torch.cat(action_idx_buf, dim=0)   # [T*B]
        adv_flat = adv_t.reshape(T*B)
        ret_flat = ret_t.reshape(T*B)

        # Advantage normalization (common PPO trick)
        #adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)
        adv_flat = adv_t.reshape(T * B)
        valid_idx = valid_flat > 0
        valid_adv = adv_flat[valid_idx]
        adv_mean = valid_adv.mean()
        adv_std = valid_adv.std().clamp_min(1e-8)
        adv_flat = (adv_flat - adv_mean) / adv_std
        adv_flat = adv_flat * valid_flat

        # --- Re-pad move tensors to a single Kmax across the whole rollout ---
        Kmax_global = max(m.shape[1] for m in moves_from_list)

        def repad(m_list, fill=0, dtype=torch.long):
            out = []
            for m in m_list:
                if m.shape[1] == Kmax_global:
                    out.append(m)
                else:
                    pad = torch.full((m.shape[0], Kmax_global - m.shape[1]), fill_value=fill, dtype=dtype, device=device)
                    out.append(torch.cat([m, pad], dim=1))
            return torch.cat(out, dim=0)  # [T*B, Kmax_global]

        moves_from_flat = repad(moves_from_list, fill=0, dtype=torch.long)
        moves_to_flat = repad(moves_to_list, fill=0, dtype=torch.long)
        moves_promo_flat = repad(moves_promo_list, fill=0, dtype=torch.long)

        def repad_mask(msk_list):
            out = []
            for msk in msk_list:
                if msk.shape[1] == Kmax_global:
                    out.append(msk)
                else:
                    pad = torch.zeros((msk.shape[0], Kmax_global - msk.shape[1]), dtype=torch.bool, device=device)
                    out.append(torch.cat([msk, pad], dim=1))
            return torch.cat(out, dim=0)

        mask_flat = repad_mask(mask_list)  # [T*B, Kmax_global]

        batch = RolloutBatch(
            obs=obs_flat,
            moves_from=moves_from_flat,
            moves_to=moves_to_flat,
            moves_promo=moves_promo_flat,
            mask=mask_flat,
            action_idx=action_idx_flat,
            old_logprob=old_logprob_flat,
            returns=ret_flat,
            advantage=adv_flat,
            old_value=old_value_flat,
            valid=valid_flat
        )

        # --- PPO update ---
        valid_indices = torch.nonzero(valid_flat > 0, as_tuple=False).squeeze(1)  # [N_valid]
        idxs = valid_indices.cpu().numpy()
        N_valid = len(idxs)

        for epoch in range(ppo_epochs):
            np.random.shuffle(idxs)

            for start in range(0, N_valid, minibatch_size):
                mb = idxs[start:start + minibatch_size]
                mb = torch.tensor(mb, dtype=torch.long, device=device)

                obs_mb = batch.obs[mb]                # [M, obs_dim]
                mf_mb = batch.moves_from[mb]          # [M, K]
                mt_mb = batch.moves_to[mb]
                mp_mb = batch.moves_promo[mb]
                msk_mb = batch.mask[mb]               # [M, K]
                act_mb = batch.action_idx[mb]         # [M]
                old_lp_mb = batch.old_logprob[mb]     # [M]
                adv_mb = batch.advantage[mb]          # [M]
                ret_mb = batch.returns[mb]            # [M]
                old_v_mb = batch.old_value[mb]        # [M]

                valid_mb = batch.valid[mb].float()    # [M]
                den = valid_mb.sum().clamp_min(1.0)


                # New policy logits over the SAME legal move set
                logits = model.move_logits(obs_mb, mf_mb, mt_mb, mp_mb)  # [M, K]
                logits = logits.masked_fill(~msk_mb, float("-inf"))
                dist = torch.distributions.Categorical(logits=logits)

                new_lp = dist.log_prob(act_mb)  # [M]
                entropy = (dist.entropy() * valid_mb).sum() / den

                ratio = (new_lp - old_lp_mb).exp()
                surr1 = ratio * adv_mb
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_mb
                policy_loss = -(torch.min(surr1, surr2) * valid_mb).sum() / den

                # Value loss (optionally clip like PPO tutorial)
                new_v = model.value(obs_mb)
                v_clipped = old_v_mb + (new_v - old_v_mb).clamp(-clip_eps, clip_eps)
                v_loss1 = (new_v - ret_mb).pow(2)
                v_loss2 = (v_clipped - ret_mb).pow(2)
                value_loss = 0.5 * (torch.max(v_loss1, v_loss2) * valid_mb).sum() / den


                loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 0.5)
                optim.step()

        # Logging
        mean_reward = rewards_t.mean().item()
        mean_done = dones_t.float().mean().item()
        print(f"update={update:04d} mean_step_reward={mean_reward:+.4f} done_rate={mean_done:.3f} Kmax={Kmax_global}")

    #save and close the model
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optim.state_dict(),
            "update": end_update,   # total number of updates trained so far
            "lr": lr,
        },
        ckpt_path
    )
    print(f"Saved checkpoint to {ckpt_path} at update={end_update}.")
    env.close()

if __name__ == "__main__":
    ppo_train_supervised(host="localhost:50051", num_envs=10)















