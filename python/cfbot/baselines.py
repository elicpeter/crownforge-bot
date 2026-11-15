"""Scripted opponents. They anchor the rating scale and give early training a
target that does not move."""

from __future__ import annotations

import torch


def _random_legal_play(legal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """A uniformly random legal non-wait action per row, and whether one exists."""
    weights = legal.float()
    weights[:, 0] = 0
    has_play = weights.sum(-1) > 0
    weights[~has_play, 0] = 1
    choice = torch.multinomial(weights.cpu(), 1).squeeze(-1).to(legal.device)
    return choice, has_play


class RandomBot:
    """Plays a random legal action with a fixed probability per decision."""

    name = "random"

    def __init__(self, play_prob: float = 0.2):
        self.play_prob = play_prob

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        choice, has_play = _random_legal_play(obs["legal"])
        play = has_play & (torch.rand(choice.shape[0], device=choice.device) < self.play_prob)
        return torch.where(play, choice, torch.zeros_like(choice))


class SpendBot:
    """Saves elixir to a threshold, then plays a random legal action. Never
    leaks elixir, so it punishes agents that waste theirs."""

    name = "spend"

    def __init__(self, threshold: float = 8.0):
        self.threshold = threshold

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        choice, has_play = _random_legal_play(obs["legal"])
        elixir = obs["global"][:, 1]
        play = has_play & (elixir >= self.threshold)
        return torch.where(play, choice, torch.zeros_like(choice))


def make_baselines() -> list:
    return [RandomBot(0.2), SpendBot(8.0), SpendBot(5.0)]
