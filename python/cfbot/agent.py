"""Deployment-side agent: public observations in, one action index out.

    agent = Agent.load("runs/main/checkpoint.pt", spec)
    action = agent.act(obs)          # flat simulator action index per row

The agent reads only the public planes (``cfbot.env.POLICY_KEYS``), never the
``priv_*`` planes, and keeps no hidden memory: everything it knows about the
match is in the observation. It is reproducible: the same checkpoint, seed and
observation sequence always give the same actions on the same machine.

Modes
-----
``sample`` (default)
    Samples from the policy with a seeded generator. Use this for real play.
    The policy usually assigns most probability to waiting even when it intends
    to play soon, so always taking the single most likely action can make it
    wait forever.
``greedy``
    Takes the most likely action. Needs no seed, but see above.
``temperature`` below 1 sharpens ``sample`` toward ``greedy``.
"""

from __future__ import annotations

import torch

from .env import POLICY_KEYS
from .league import load_policy
from .model import Policy


class Agent:
    def __init__(self, model: Policy, mode: str = "sample", seed: int = 0, temperature: float = 1.0):
        if mode not in ("sample", "greedy"):
            raise ValueError("mode must be 'sample' or 'greedy'")
        self.model = model.eval()
        self.mode = mode
        self.temperature = temperature
        self.generator = torch.Generator().manual_seed(seed)

    @classmethod
    def load(cls, path: str, spec: dict, device="cpu", **kwargs) -> "Agent":
        model, _ = load_policy(path, spec, torch.device(device))
        return cls(model, **kwargs)

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the simulator's flat action index for each row."""
        public = {key: obs[key] for key in POLICY_KEYS}
        out = self.model.act(
            public,
            deterministic=self.mode == "greedy",
            generator=self.generator,
            temperature=self.temperature,
            with_value=False,
        )
        return out["flat"]

    @torch.no_grad()
    def action_log_probs(self, obs: dict[str, torch.Tensor]):
        """Log-probability of all 4,611 actions and the win-value estimate."""
        return self.model.full_log_probs({key: obs[key] for key in POLICY_KEYS})
