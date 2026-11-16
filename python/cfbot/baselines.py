"""Scripted opponents: fixed reference points for the gauntlet's rating scale,
and a target that does not move for early training. All of them place random
legal cards on random legal tiles, so they are weak and about equally so."""

from __future__ import annotations

import torch


def _random_legal_play(legal: torch.Tensor, generator) -> tuple[torch.Tensor, torch.Tensor]:
    """A uniformly random legal non-wait action per row, and whether one exists."""
    weights = legal.float()
    weights[:, 0] = 0
    has_play = weights.sum(-1) > 0
    weights[~has_play, 0] = 1
    choice = torch.multinomial(weights.cpu(), 1, generator=generator).squeeze(-1).to(legal.device)
    return choice, has_play


def _generator(seed: int | None):
    return None if seed is None else torch.Generator().manual_seed(seed)


class RandomBot:
    """Plays a random legal action with a fixed probability per decision."""

    name = "random"

    def __init__(self, play_prob: float = 0.2, seed: int | None = None):
        self.play_prob = play_prob
        self.generator = _generator(seed)

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        choice, has_play = _random_legal_play(obs["legal"], self.generator)
        draw = torch.rand(choice.shape[0], generator=self.generator).to(choice.device)
        play = has_play & (draw < self.play_prob)
        return torch.where(play, choice, torch.zeros_like(choice))


class SpendBot:
    """Saves elixir to a threshold, then plays a random legal action. Never
    leaks elixir, so it punishes agents that waste theirs."""

    name = "spend"

    def __init__(self, threshold: float = 8.0, seed: int | None = None):
        self.threshold = threshold
        self.generator = _generator(seed)

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        choice, has_play = _random_legal_play(obs["legal"], self.generator)
        elixir = obs["global"][:, 1]
        play = has_play & (elixir >= self.threshold)
        return torch.where(play, choice, torch.zeros_like(choice))


class RushBot:
    """The arena's example client, at this bot's decision rate: whenever it can
    afford a card it plays one at once, a random card on a random tile of the
    six rows before the river. The likeliest opponent in a tournament of
    newcomers, and one that never holds elixir back."""

    name = "rush"

    def __init__(self, play_prob: float = 0.95, seed: int | None = None):
        self.play_prob = play_prob
        self.generator = _generator(seed)

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        from . import actions as A

        legal = obs["legal"]
        front = torch.zeros(A.HEIGHT, A.WIDTH, dtype=torch.bool)
        front[9:15, 2:16] = True
        mask = torch.zeros(legal.shape[-1], dtype=torch.bool)
        mask[1 : A.PLAY_ACTION_COUNT] = front.flatten().repeat(A.HAND_SLOTS)
        choice, has_play = _random_legal_play(legal & mask.to(legal.device), self.generator)
        draw = torch.rand(choice.shape[0], generator=self.generator).to(choice.device)
        play = has_play & (draw < self.play_prob)
        return torch.where(play, choice, torch.zeros_like(choice))


class LaneBot:
    """Arena only. Pressure down one lane and nothing else: the pattern two
    people beat an early network with. It picks the lane whose enemy princess
    tower is weaker (a fallen one first; its own deck decides a tie) and puts
    every troop at that bridge, or, once the tower is down, in the pocket
    beside the king tower. With elixir to spare it throws spells at the tower
    in that lane, or at a king tower down to a quarter of its health, and sets
    buildings behind the bridge. It never defends.

    ``rush`` plays the moment a troop is affordable. ``push`` saves to nine
    elixir and then empties its hand, one card a decision: a push of several
    units arriving together."""

    def __init__(self, spec: dict, mode: str = "rush", seed: int | None = None):
        if mode not in ("rush", "push"):
            raise ValueError("mode is 'rush' or 'push'")
        table = torch.as_tensor(spec["card_table"])
        self.spell = table[:, 4] > 0
        self.building = table[:, 3] > 0
        self.mode = mode
        self.name = f"lane{mode}"
        self.generator = _generator(seed)
        self.dumping = torch.zeros(0, dtype=torch.bool)

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        from . import actions as A
        from .env import observed_towers

        device = obs["legal"].device
        legal = obs["legal"].bool().cpu()
        n = legal.shape[0]
        plays = legal[:, 1 : A.PLAY_ACTION_COUNT].view(n, A.HAND_SLOTS, A.CELLS)
        afford = plays.any(-1)
        token = obs["hand"][:, :, 1].long().cpu()
        spell, building = self.spell[token], self.building[token]
        elixir = obs["global"][:, 1].cpu()
        towers = observed_towers({"entities": obs["entities"].cpu()})
        left, right = towers[:, 4], towers[:, 5]
        tie = obs["own_deck"][:, :, 0].sum(-1).long().cpu() % 2
        lane = torch.where(left == right, tie, (right < left).long())
        standing = torch.where(lane == 0, left, right) > 0

        troop = afford & ~spell & ~building
        if self.mode == "push":
            if self.dumping.shape[0] != n:
                self.dumping = torch.zeros(n, dtype=torch.bool)
            self.dumping = (self.dumping | (elixir >= 9.0)) & afford.any(-1)
            troop = troop & self.dumping[:, None]
        spare = (elixir >= 8.0)[:, None]
        draw = torch.rand((n, A.HAND_SLOTS + 1), generator=self.generator)
        score = draw[:, : A.HAND_SLOTS] + 10.0 * troop + 5.0 * (afford & spell & spare) + 2.0 * (afford & building & spare)
        best, slot = score.max(-1)

        jitter = (draw[:, A.HAND_SLOTS] * 3).long() - 1
        bridge = 14 * A.WIDTH + (3 + 11 * lane + jitter)
        pocket = 19 * A.WIDTH + (7 + 3 * lane)
        # Spells go at the lane's tower, and at the king tower once that is
        # gone or the king is nearly down: a king can fall behind a standing
        # princess tower.
        king = ~standing | (towers[:, 3] <= 0.25)
        tower = torch.where(king, torch.full_like(lane, 29 * A.WIDTH + 8), 25 * A.WIDTH + (3 + 11 * lane))
        behind = 10 * A.WIDTH + (4 + 9 * lane)
        rows = torch.arange(n)
        mine = plays[rows, slot]
        in_pocket = mine[rows, pocket]
        cell = torch.where(in_pocket, pocket, bridge)
        cell = torch.where(spell[rows, slot], tower, cell)
        cell = torch.where(building[rows, slot], behind, cell)
        play = (best >= 2.0) & mine[rows, cell]
        action = 1 + slot * A.CELLS + cell
        return torch.where(play, action, torch.zeros_like(action)).to(device)


def make_baselines(simulator: str | None = None, spec: dict | None = None) -> list:
    """The league's scripted opponents. On the arena the two spend bots give
    way to lane pressure: the network beats all the random-card bots alike."""
    if simulator == "arena" and spec is not None:
        return [RandomBot(0.2), RushBot(), LaneBot(spec, "rush"), LaneBot(spec, "push")]
    return [RandomBot(0.2), SpendBot(8.0), SpendBot(5.0)]


def baseline_label(bot) -> str:
    return f"{bot.name}{getattr(bot, 'threshold', '')}"
