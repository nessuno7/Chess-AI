"""
Everything self-play training and puzzle training have in common: the sample format, batching,
and the loss.

A sample is (state, pi_idx, pi_p, z):
    state   np.float16[C, 8, 8]   ChessGame.encode(), halved in size (every plane is 0/1 except the
                                  halfmove clock, which loses nothing that matters in float16)
    pi_idx  np.int64[k]           hash_move indices of the target moves
    pi_p    np.float32[k]         their probabilities (sums to 1)
    z       float                 game result from the side to move's perspective
The policy is kept sparse because a full 4672 wide vector per sample would be ~150x larger
than the ~30 legal moves it actually describes.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from monte_carlo_tree_search import POLICY_SIZE


def make_sample(game, pi_idx, pi_p, z):
    return (game.encode().astype(np.float16),
            np.asarray(pi_idx, dtype=np.int64),
            np.asarray(pi_p, dtype=np.float32),
            float(z))


def collate(samples, device="cpu"):
    """list of samples -> (states (B,C,8,8), dense pi (B,4672), z (B,)) tensors."""
    states = torch.from_numpy(np.stack([s[0] for s in samples]).astype(np.float32))
    pis = torch.zeros(len(samples), POLICY_SIZE)
    for i, (_, idx, p, _) in enumerate(samples):
        pis[i, torch.from_numpy(idx)] = torch.from_numpy(p)
    zs = torch.tensor([s[3] for s in samples], dtype=torch.float32)
    return states.to(device), pis.to(device), zs.to(device)


def loss_fn(net, states, pis, zs):
    """
    AlphaZero loss: cross entropy of the policy against pi, plus MSE of the value against z.
    (The L2 term of the paper is the optimizer's weight_decay.) Returns (total, policy, value).
    """
    logits, values = net(states)
    policy_loss = -(pis * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(values, zs)
    return policy_loss + value_loss, policy_loss, value_loss


def train_steps(net, optimizer, samples, steps, batch_size, device="cpu"):
    """
    `steps` gradient steps on batches drawn uniformly (with replacement across batches) from
    `samples`. Returns the mean (total, policy, value) loss over the steps.
    """
    net.train()
    totals = np.zeros(3)
    for _ in range(steps):
        batch = random.sample(samples, min(batch_size, len(samples)))
        loss, pl, vl = loss_fn(net, *collate(batch, device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals += (loss.item(), pl.item(), vl.item())
    return totals / max(steps, 1)


def train_epoch(net, optimizer, samples, batch_size, device="cpu"):
    """One shuffled pass over `samples`. Returns the mean (total, policy, value) loss."""
    net.train()
    order = list(range(len(samples)))
    random.shuffle(order)
    totals, n = np.zeros(3), 0
    for start in range(0, len(order), batch_size):
        batch = [samples[i] for i in order[start:start + batch_size]]
        loss, pl, vl = loss_fn(net, *collate(batch, device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        totals += (loss.item(), pl.item(), vl.item())
        n += 1
    return totals / max(n, 1)


def make_optimizer(net, lr=1e-3, weight_decay=1e-4):
    return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)


# ---- replay buffer on disk -----------------------------------------------------------

def save_samples(path, samples):
    """
    Store a list of samples as one .npz: the states stacked into a single float16 array, and the
    sparse policies concatenated with an offsets array marking where each sample's moves start.
    A few large arrays save and load far faster than pickling 100k small tuples. Atomic like
    save_checkpoint (temporary file, then rename).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lengths = np.array([len(s[1]) for s in samples], dtype=np.int64)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp,
             states=np.stack([s[0] for s in samples]) if samples else np.zeros((0,), np.float16),
             pi_idx=np.concatenate([s[1] for s in samples]) if samples else np.zeros(0, np.int64),
             pi_p=np.concatenate([s[2] for s in samples]) if samples else np.zeros(0, np.float32),
             offsets=np.concatenate([[0], np.cumsum(lengths)]),
             z=np.array([s[3] for s in samples], dtype=np.float32))
    os.replace(tmp, path)


def load_samples(path):
    """Inverse of save_samples: the list of (state, pi_idx, pi_p, z) tuples."""
    d = np.load(path)
    states, idx, p, off, z = d["states"], d["pi_idx"], d["pi_p"], d["offsets"], d["z"]
    return [(states[i], idx[off[i]:off[i + 1]], p[off[i]:off[i + 1]], float(z[i]))
            for i in range(len(z))]
