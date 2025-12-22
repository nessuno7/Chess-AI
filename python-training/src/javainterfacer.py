from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import requests

@dataclass
class JavaChessHTTPClient:
    """
    A thin client responsible ONLY for talking to the Java server.

    This class should not contain RL logic. It just sends requests and returns parsed JSON.
    """

    def __init__(self):
        base_url: str
        timeout_s: float = 5.0
        pass



    def health(self) -> Dict[str, Any]:
        r = requests.get(f"{self.base_url}/health", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if seed is not None:
            payload["seed"] = int(seed)
        if options is not None:
            payload["options"] = options
        r = requests.post(f"{self.base_url}/reset", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def step(self, action: int) -> Dict[str, Any]:
        payload = {"action": int(action)}
        r = requests.post(f"{self.base_url}/step", json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def render(self) -> Dict[str, Any]:
        """
        Optional: if you want Java to return a printable board / FEN / ASCII.
        """
        r = requests.get(f"{self.base_url}/render", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()
