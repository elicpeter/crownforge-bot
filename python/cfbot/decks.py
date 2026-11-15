"""Decks for training: a seed pool of archetypes, and per-card and per-deck
results from self-play.

The archetypes are familiar deck shapes from the real game. Nothing shows they
are strong in this simulator, so they get no special weight: they seed the
pool that training samples and mutates, next to random playable decks.
Names are the simulator's card ids.
"""

from __future__ import annotations

import numpy as np

ARCHETYPES = {
    "starter": ["Knight", "Archers", "Giant", "Musketeer", "Skeletons", "Fireball", "Arrows", "Zap"],
    "hog_cycle": ["HogRider", "Musketeer", "IceGolem", "IceSpirit", "Skeletons", "Cannon", "Fireball", "TheLog"],
    "hog_quake": ["HogRider", "Earthquake", "Firecracker", "Valkyrie", "Skeletons", "IceSpirit", "Cannon", "TheLog"],
    "giant_princes": ["Giant", "Prince", "DarkPrince", "MegaMinion", "ElectroWizard", "Bats", "Fireball", "Zap"],
    "golem": ["Golem", "NightWitch", "BabyDragon", "Lumberjack", "MegaMinion", "Tornado", "Lightning", "BarbarianBarrel"],
    "lava_loon": ["LavaHound", "Balloon", "MegaMinion", "Minions", "SkeletonDragons", "Tombstone", "Fireball", "Arrows"],
    "log_bait": ["GoblinBarrel", "Princess", "GoblinGang", "Knight", "IceSpirit", "InfernoTower", "Rocket", "TheLog"],
    "xbow": ["Xbow", "Tesla", "Archers", "Knight", "Skeletons", "IceSpirit", "Fireball", "TheLog"],
    "miner_poison": ["Miner", "Poison", "Valkyrie", "ElectroWizard", "InfernoDragon", "Bats", "GoblinGang", "TheLog"],
    "pekka_bridge": ["Pekka", "BattleRam", "Bandit", "RoyalGhost", "ElectroWizard", "MagicArcher", "Poison", "Zap"],
    "royal_giant": ["RoyalGiant", "Fisherman", "Hunter", "RoyalGhost", "ElectroSpirit", "Skeletons", "Lightning", "TheLog"],
    "graveyard": ["Graveyard", "Poison", "Knight", "BabyDragon", "IceWizard", "Tombstone", "Tornado", "BarbarianBarrel"],
    "mortar": ["Mortar", "Knight", "Archers", "GoblinGang", "SpearGoblins", "IceSpirit", "Fireball", "TheLog"],
    "mega_knight": ["MegaKnight", "Miner", "Bats", "GoblinGang", "InfernoDragon", "SpearGoblins", "Fireball", "Zap"],
    "three_musketeers": ["ThreeMusketeers", "BattleRam", "ElixirCollector", "IceGolem", "HealSpirit", "RoyalGhost", "Bandit", "BarbarianBarrel"],
    "sparky": ["GoblinGiant", "Sparky", "DarkPrince", "MegaMinion", "ElectroWizard", "HealSpirit", "Rage", "Zap"],
    "royal_hogs": ["RoyalHogs", "RoyalRecruits", "FlyingMachine", "GoblinCage", "ElectroSpirit", "Fireball", "Arrows", "BarbarianBarrel"],
    "skeleton_king": ["SkeletonKing", "Graveyard", "Bowler", "BabyDragon", "Tombstone", "Tornado", "Poison", "BarbarianBarrel"],
    "archer_queen": ["ArcherQueen", "HogRider", "Valkyrie", "IceSpirit", "Skeletons", "Cannon", "Fireball", "TheLog"],
    "golden_knight": ["GoldenKnight", "BattleRam", "Bandit", "ElectroWizard", "MagicArcher", "RoyalGhost", "Poison", "Zap"],
}


def resolve_pool(deck_pool) -> list[list[str]] | None:
    """Turn a config value into explicit decks."""
    if deck_pool is None:
        return None
    if deck_pool == "archetypes":
        return list(ARCHETYPES.values())
    if isinstance(deck_pool, str):
        raise ValueError(f"unknown deck pool: {deck_pool}")
    return deck_pool


class DeckStats:
    """Self-play results by card and by deck. Both seats are the same policy
    there, so a lopsided score points at the decks. Each side's score also
    reflects how much practice the policy has had with that deck, so read these
    as leads to confirm with paired games, not as a ranking."""

    def __init__(self, n_cards: int):
        self.card_games = np.zeros(n_cards)
        self.card_score = np.zeros(n_cards)
        self.decks: dict[tuple[int, ...], list[float]] = {}

    def record(self, decks: np.ndarray, blue_result: float) -> None:
        """``decks`` is ``[2, 8]`` card tokens; ``blue_result`` is +1, 0 or -1."""
        for side, sign in ((0, 1.0), (1, -1.0)):
            score = (sign * blue_result + 1.0) / 2.0
            cards = np.asarray(decks[side], dtype=np.int64)
            self.card_games[cards] += 1
            self.card_score[cards] += score
            key = tuple(sorted(int(card) for card in cards))
            entry = self.decks.setdefault(key, [0.0, 0.0])
            entry[0] += 1
            entry[1] += score
        if len(self.decks) > 20_000:
            # Random decks rarely repeat; keep the ones seen most.
            keep = sorted(self.decks.items(), key=lambda item: -item[1][0])[:5_000]
            self.decks = dict(keep)

    def summary(self, card_names: list[str], min_games: int = 30) -> dict:
        cards = {
            card_names[token]: {
                "games": int(self.card_games[token]),
                "score": round(float(self.card_score[token] / self.card_games[token]), 4),
            }
            for token in np.nonzero(self.card_games)[0]
        }
        decks = [
            {"deck": [card_names[token] for token in key], "games": int(games),
             "score": round(score / games, 4)}
            for key, (games, score) in self.decks.items()
            if games >= min_games
        ]
        decks.sort(key=lambda entry: -entry["games"])
        return {"cards": cards, "decks": decks}

    def state_dict(self) -> dict:
        return {"card_games": self.card_games, "card_score": self.card_score, "decks": self.decks}

    def load_state_dict(self, state: dict) -> None:
        self.card_games = state["card_games"]
        self.card_score = state["card_score"]
        self.decks = state["decks"]
