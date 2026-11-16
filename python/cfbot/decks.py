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


# The same shapes under the cr-rl-arena simulator's card names, where it has the
# cards. The last three are decks its own material hands to a newcomer: the
# connection guide's example, the example client's, and the built-in default.
ARENA_ARCHETYPES = {
    "starter": ["knight", "archers", "giant", "musketeer", "skeletons", "fireball", "arrows", "zap"],
    "hog_cycle": ["hogrider", "musketeer", "ice_golem", "ice_spirit", "skeletons", "cannon", "fireball", "the_log"],
    "hog_quake": ["hogrider", "earthquake", "firecracker", "valkyrie", "skeletons", "ice_spirit", "cannon", "the_log"],
    "giant_princes": ["giant", "prince", "dark_prince", "mega_minion", "electro_wizard", "bats", "fireball", "zap"],
    "golem": ["golem", "night_witch", "babydragon", "lumberjack", "mega_minion", "tornado", "lightning", "barbarian_barrel"],
    "lava_loon": ["lava_hound", "balloon", "mega_minion", "minions", "skeleton_dragons", "tombstone", "fireball", "arrows"],
    "log_bait": ["goblin_barrel", "princess", "goblin_gang", "knight", "ice_spirit", "inferno_tower", "rocket", "the_log"],
    "xbow": ["xbow", "tesla", "archers", "knight", "skeletons", "ice_spirit", "fireball", "the_log"],
    "miner_poison": ["miner", "poison", "valkyrie", "electro_wizard", "inferno_dragon", "bats", "goblin_gang", "the_log"],
    "pekka_bridge": ["pekka", "battle_ram", "bandit", "royal_ghost", "electro_wizard", "magic_archer", "poison", "zap"],
    "royal_giant": ["royal_giant", "fisherman", "hunter", "royal_ghost", "electro_spirit", "skeletons", "lightning", "the_log"],
    "graveyard": ["graveyard", "poison", "knight", "babydragon", "ice_wizard", "tombstone", "tornado", "barbarian_barrel"],
    "mortar": ["mortar", "knight", "archers", "goblin_gang", "spear_goblins", "ice_spirit", "fireball", "the_log"],
    "mega_knight": ["mega_knight", "miner", "bats", "goblin_gang", "inferno_dragon", "spear_goblins", "fireball", "zap"],
    "three_musketeers": ["three_musketeers", "battle_ram", "elixir_collector", "ice_golem", "ice_spirit", "royal_ghost", "bandit", "barbarian_barrel"],
    "sparky": ["goblin_giant", "sparky", "dark_prince", "mega_minion", "electro_wizard", "bats", "rage", "zap"],
    "royal_hogs": ["royal_hogs", "royal_recruits", "musketeer", "goblin_cage", "electro_spirit", "fireball", "arrows", "barbarian_barrel"],
    "skeleton_king": ["skeleton_king", "graveyard", "bowler", "babydragon", "tombstone", "tornado", "poison", "barbarian_barrel"],
    "archer_queen": ["archer_queen", "hogrider", "valkyrie", "ice_spirit", "skeletons", "cannon", "fireball", "the_log"],
    "golden_knight": ["golden_knight", "battle_ram", "bandit", "electro_wizard", "magic_archer", "royal_ghost", "poison", "zap"],
    "guide_example": ["knight", "archers", "giant", "musketeer", "valkyrie", "minions", "wizard", "minipekka"],
    "builtin_default": ["archer_queen", "archers", "babydragon", "balloon", "bandit", "barbarian_hut", "barbarians", "bats"],
}


# What training samples as pool decks on the arena: the archetypes with Barbarian
# Barrel replaced (it spawns no barbarian there), and twelve decks built on how
# that simulator's cards really behave (docs/arena-deck-research.md): a Miner
# that hits towers at full damage, an Elixir Golem that pays the defender
# nothing, an X-Bow that troops do not distract, a Graveyard of twenty
# skeletons, Royal Delivery as a cheap area hit. ARENA_ARCHETYPES itself stays
# as it is: the gauntlet's deck pairs are drawn from it, and a rating scale is
# only one scale while those stay the same.
ARENA_CANDIDATES = {
    "healer_elixir_golem": ["elixir_golem", "battle_healer", "night_witch", "phoenix", "firecracker", "earthquake", "tornado", "royal_delivery"],
    "miner_wall_breakers": ["miner", "poison", "knight", "skeletons", "wall_breakers", "tesla", "bats", "the_log"],
    "tower_first_xbow": ["xbow", "tesla", "knight", "archers", "skeletons", "ice_spirit", "royal_delivery", "the_log"],
    "graveyard_hut": ["graveyard", "poison", "knight", "babydragon", "goblin_hut", "ice_wizard", "skeletons", "the_log"],
    "minion_giant_cycle": ["minion_giant", "knight", "skeletons", "ice_wizard", "bats", "cannon", "lightning", "the_log"],
    "giant_anti_rush": ["giant", "giant_skeleton", "firecracker", "musketeer", "minipekka", "royal_delivery", "zap", "skeletons"],
    "lavaloon_anti_air": ["lava_hound", "balloon", "babydragon", "minions", "tombstone", "barbarians", "fireball", "zap"],
    "hog_delivery_cycle": ["hogrider", "firecracker", "knight", "cannon", "skeletons", "ice_spirit", "the_log", "royal_delivery"],
    "royal_giant_control": ["royal_giant", "hunter", "knight", "skeletons", "tesla", "lightning", "the_log", "firecracker"],
    "barrel_air_bait": ["goblin_barrel", "princess", "goblin_gang", "minions", "knight", "inferno_tower", "royal_delivery", "the_log"],
    "drill_wall_breakers": ["goblin_drill", "wall_breakers", "knight", "firecracker", "bats", "poison", "tesla", "the_log"],
    "battle_ram_counterpush": ["battle_ram", "knight", "giant_skeleton", "electro_wizard", "dark_prince", "minions", "royal_delivery", "zap"],
}
# Bandit is swapped too: its dash can leave it frozen and untouchable for the
# rest of the match. The built-in default deck stays as newcomers will run it.
_SWAPS = {"barbarian_barrel": "the_log", "bandit": "knight"}
ARENA_POOL = {
    **{name: deck if name == "builtin_default" else [_SWAPS.get(card, card) for card in deck]
       for name, deck in ARENA_ARCHETYPES.items()},
    **ARENA_CANDIDATES,
}


def archetypes(simulator: str | None) -> dict[str, list[str]]:
    """The archetype decks under the named simulator's card names: the decks
    the gauntlet and paired evaluations draw their matchups from."""
    return ARENA_ARCHETYPES if simulator == "arena" else ARCHETYPES


def pool_decks(simulator: str | None) -> dict[str, list[str]]:
    """The decks training samples as pool decks on the named simulator."""
    return ARENA_POOL if simulator == "arena" else ARCHETYPES


def resolve_pool(deck_pool, simulator: str | None = None) -> list[list[str]] | None:
    """Turn a config value into explicit decks."""
    if deck_pool is None:
        return None
    if deck_pool == "archetypes":
        return list(pool_decks(simulator).values())
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
