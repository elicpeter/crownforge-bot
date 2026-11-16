"""The cr-rl-arena simulator behind the same interface as ``cfbot.env.Env``.

cr-rl-arena (https://github.com/ilumn/cr-rl-arena) is the simulator of an Elo
tournament for agents. ``arena/`` in this repository is a Rust extension,
``cfbot_arena``, that compiles that simulator's engine source unchanged and
writes observations in this bot's planes. Build it with

    scripts/setup_arena.sh          (or scripts\\setup_arena.ps1 on Windows)

The planes hold only what the tournament server sends a player: its own hand
and elixir, and every unit and tower. Spell areas, projectiles and status
effects are not sent, so the hazard plane is empty and the status columns of
the entity plane are zero. Opponent cards and elixir are inferred from the
units that appear (``arena/src/track.rs``).

Card tokens are the simulator's library ids plus one: ``spec["card_names"]``.
"""

from __future__ import annotations

import torch

from .env import Env, card_ids


def bridge():
    try:
        import cfbot_arena
    except ImportError as error:
        raise RuntimeError(
            "the arena bridge is not built: run scripts/setup_arena.sh (see the README)"
        ) from error
    return cfbot_arena


class ArenaEnv(Env):
    """Batch of cr-rl-arena battles. Rows are ``env * 2 + seat``, as in ``Env``.
    Both seats see themselves at the bottom of the board, as on the server."""

    def __init__(
        self,
        num_envs: int,
        workers: int = 0,
        decision_ticks: int = 6,
        seed: int = 0,
        max_entities: int = 128,
        deck_pool: list[list[str]] | None = None,
        pool_prob: float = 1.0,
        mirror_decks: bool = False,
        max_skip: int = 8,
        exclude_cards=(),
        rules: str | None = None,
        level_jitter: tuple[int, int] | None = None,
        device: torch.device | str = "cpu",
    ):
        if rules is not None or level_jitter is not None:
            raise ValueError("rules and level_jitter belong to the Crownforge simulator; the arena plays its own cards")
        vec = bridge().VecEnv
        spec = vec(num_envs=1, workers=1).spec()
        known = {name for name in spec["card_names"] if name}
        self.env = vec(
            num_envs=num_envs,
            workers=workers,
            decision_ticks=decision_ticks,
            seed=seed,
            max_entities=max_entities,
            deck_pool=[card_ids(spec, deck) for deck in deck_pool] if deck_pool else None,
            pool_prob=pool_prob,
            mirror_decks=mirror_decks,
            max_skip=max_skip,
            # The default exclusions name another simulator's cards; those are skipped.
            exclude_cards=card_ids(spec, [name for name in exclude_cards if name in known]),
        )
        self.spec = self.env.spec()
        self.num_envs = num_envs
        self.rows = num_envs * 2
        self.device = torch.device(device)

        self.focus_seat = None

    def set_focus(self, decks: list[list[str]], prob: float) -> None:
        """Decks one seat (chosen at random) of a new battle plays with chance
        ``prob``, whatever the other seat draws: the decks being prepared for
        the tournament. ``focus_seat`` then says, per battle, which seat holds
        one in the running episode (-1 for neither); the collector puts the
        learner on that seat in league battles, so the practice with these
        decks goes to the network that is learning and not to a frozen one."""
        self.env.set_focus([card_ids(self.spec, deck) for deck in decks], prob)

    def _obs(self, raw: dict) -> dict[str, torch.Tensor]:
        self.focus_seat = raw["focus_seat"]
        return super()._obs(raw)
