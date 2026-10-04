"""
The policy/value network and the evaluator that plugs it into the search.

Input   ChessGame.encode(): (14 * stored_timesteps + 7, 8, 8), planes indexed [plane, rank, file]
Policy  4672 logits, the index space of ChessGame.hash_move: 73 * from_square + move_type.
        The policy head outputs (73, 8, 8) = [move_type, rank, file]; moving the 73 to the back
        and flattening gives (rank * 8 + file) * 73 + move_type, and rank * 8 + file is exactly
        python-chess's square number, so no lookup table is needed.
Value   one tanh scalar in [-1, 1], from the side to move's perspective.
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from board import N_FRAME, N_GLOBALS
from monte_carlo_tree_search import POLICY_SIZE, BatchedEvaluator

N_MOVE_TYPES = 73

# every script saves its weights here by default, wherever it is run from (git ignored)
REPO_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


def input_channels(stored_timesteps):
    return N_FRAME * stored_timesteps + N_GLOBALS


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class ChessNet(nn.Module):
    """
    AlphaZero shaped residual tower, sized down so it trains on a CPU. The defaults (6 blocks of
    64 channels, ~0.5M parameters) are a starting point, not a tuned choice.
    """

    def __init__(self, stored_timesteps=1, channels=64, blocks=6):
        super().__init__()
        self.stored_timesteps = stored_timesteps
        self.channels = channels
        self.blocks = blocks

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels(stored_timesteps), channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])

        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, N_MOVE_TYPES, 1),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 4, 1, bias=False),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(4 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        """x: (B, C, 8, 8) -> (policy_logits (B, 4672), value (B,))"""
        h = self.tower(self.stem(x))
        p = self.policy_head(h)                                      # (B, 73, rank, file)
        p = p.permute(0, 2, 3, 1).reshape(x.shape[0], POLICY_SIZE)   # (B, square * 73 + type)
        v = self.value_head(h).squeeze(-1)
        return p, v

    def hparams(self):
        return {"stored_timesteps": self.stored_timesteps, "channels": self.channels, "blocks": self.blocks}


class NetEvaluator(BatchedEvaluator):
    """
    Runs ChessNet for the search. evaluate_batch stacks every game's encode() into one tensor and
    does a single forward pass; the net is put in eval mode (BatchNorm uses running stats).
    """

    def __init__(self, net: ChessNet, device="cpu"):
        self.net = net
        self.device = torch.device(device)

    @torch.inference_mode()
    def evaluate_batch(self, games):
        self.net.eval()
        x = torch.from_numpy(np.stack([g.encode() for g in games])).to(self.device)
        logits, values = self.net(x)
        logits = logits.float().cpu().numpy()
        values = values.float().cpu().numpy()
        return [(logits[i], float(values[i])) for i in range(len(games))]


# ---- checkpoints ---------------------------------------------------------------------

def save_checkpoint(path, net, optimizer=None, **extra):
    """
    Write the weights (plus the hparams needed to rebuild the net, the optimizer state and any
    extra fields) to `path`. Written to a temporary file first and then renamed over the old one,
    so killing the process mid-save never leaves a corrupt checkpoint behind.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({
        "hparams": net.hparams(),
        "model": net.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        **extra,
    }, tmp)
    os.replace(tmp, path)


def load_checkpoint(path, device="cpu"):
    """(net, checkpoint_dict). The net is rebuilt from the stored hparams, so the size travels with it."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    net = ChessNet(**ckpt["hparams"]).to(device)
    net.load_state_dict(ckpt["model"])
    return net, ckpt
