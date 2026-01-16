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
        chosen_moves.append(decoder.envs[b].moves[idx[b].item()])

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


def compute_gae(rewards, dones, values, next_value, gamma=0.99, lam=0.95):
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

    ckpt_path = "ac_checkpoint.pth"
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

        adv_t, ret_t = compute_gae(rewards_t, dones_t, values_t, next_value, gamma=gamma, lam=gae_lambda)

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
        skip_n_problems = 0
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    env = RemoteProblemsEnv(num_envs=num_envs, host=host)
    model = ActorCritic(obs_dim=rs.OBS_DIM).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    iterator = ProblemIterator(skip=skip_n_problems)

    ckpt_path = "ac_checkpoint.pth"
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
        values_buf = []
        logprob_buf = []
        action_idx_buf = []

        # store per-step padded legal sets (variable K each step), we'll repad later
        moves_from_list = []
        moves_to_list = []
        moves_promo_list = []
        mask_list = []


        puzzleIndex, fen, puzzleMoves, puzzleRating = next(iterator)
        #env.startMessage(fen) TODO: initMessage, will get Step Response as answer
        # Prime env (get first response)
        resp = rs.ResponseDecoder(env.reset(), device)

        for t in range(puzzleMoves): #each step is one problem attempted to solve

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

            moves_string = [] #rewards increase exponentially with difficult, also give more emphasis to first move important
            for i in range(num_envs):
                if intToMove(mf[i], mt[i], mp[i]) == puzzleMoves[t]:
                    rewards[i] ==
                elif moveToInt(puzzleMoves[t])[0] == mf[i]: #if correct piece moves still give partial credit
                else: #give negative reward for not correct move and give negative reward inversely proportion to distance from correct state at that move



            #rewards for step (k+1)-1 TODO: switch reward to see if it equals to problem
            rewards_buf.append(rewards_t)

            dones_t =
            dones_buf.append(dones_t)

            #now step k+1
            resp = rs.ResponseDecoder(env.step(chosen_moves), device=device)

        # bootstrap value for GAE
        with torch.no_grad():
            next_obs = resp.getObservationMatrixTensor()
            next_value = model.value(next_obs)  # [B]

        rewards_t = torch.stack(rewards_buf, dim=0)  # [T, B]
        dones_t = torch.stack(dones_buf, dim=0)      # [T, B]
        values_t = torch.stack(values_buf, dim=0)    # [T, B]

        adv_t, ret_t = compute_gae(rewards_t, dones_t, values_t, next_value, gamma=gamma, lam=gae_lambda)

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


if __name__ == "__main__":
    ppo_train_unsupervised(host="localhost:50051", num_envs=10)















