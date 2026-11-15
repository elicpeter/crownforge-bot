"""Curated decks for the training curriculum.

Uniformly random decks are mostly incoherent, and an agent that only ever sees
them learns little about real matchups. Training therefore starts mostly on
these archetypes and gradually mixes in random decks for coverage.
Names are the simulator's card ids.
"""

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
