"""The CRForge simulator behind the same interface as ``cfbot.env.Env``.

CRForge (https://github.com/voonhous/crforge) is a Java battle core held tick
for tick to battles recorded in the game. ``java/cfbot/bridge/CfVector.java``
drives its Ladder battles on the JVM's threads and writes observations in this
bot's layout, so the network, the collector and the league see the planes they
always have. Three things are needed, none of them shipped here:

* a CRForge checkout with its env module installed
  (``./gradlew :env:installDist`` makes ``env/build/install/env``);
* the game tables of one data version (``./gradlew :tables:dataVersionTables``);
* the bridge class, compiled against that distribution
  (``scripts/build_crforge.sh <install>``), and the ``jpype1`` package.

The folders come from ``TrainConfig.crforge_install`` and ``crforge_tables``,
or the environment variables ``CRFORGE_INSTALL`` and ``CRFORGE_TABLES``. The
JVM is the one ``JAVA_HOME`` names; CRForge wants Java 21.

Card tokens are the same as on the other simulator: an index into
``CARD_NAMES``. ``ROW_NAMES`` gives the tables' row name where it differs.
What differs in meaning from the other simulator's planes is listed in the
README section on simulators.
"""

from __future__ import annotations

import glob
import os

import numpy as np
import torch

from .env import ENTITY_KEYS, Env, entity_width

# The bot's card catalogue, by token. Token 0 is "none".
CARD_NAMES = (
    "", "Knight", "Archers", "Giant", "Musketeer", "Skeletons", "Fireball", "Arrows", "Zap",
    "MiniPekka", "Goblins", "Barbarians", "EliteBarbarians", "Pekka", "SpearGoblins", "DartGoblin",
    "RoyalGiant", "Cannon", "Minions", "MegaMinion", "Bats", "GoblinGang", "Rascals", "Xbow",
    "MinionHorde", "Wizard", "BabyDragon", "Rocket", "Valkyrie", "GiantSkeleton", "Balloon",
    "Bomber", "Guards", "RoyalRecruits", "SkeletonArmy", "SkeletonDragons", "FlyingMachine",
    "GoblinCage", "Tombstone", "BombTower", "BarbarianHut", "Princess", "Freeze", "Mortar", "Witch",
    "Rage", "NightWitch", "GoblinBarrel", "Lumberjack", "IceWizard", "IceGolem", "Berserker",
    "Prince", "DarkPrince", "Bowler", "Golem", "Executioner", "Firecracker", "FireSpirit",
    "IceSpirit", "WallBreakers", "GiantSnowball", "HealSpirit", "InfernoTower", "InfernoDragon",
    "Lightning", "Poison", "Earthquake", "Zappies", "ElectroWizard", "ElectroSpirit",
    "ElectroDragon", "Furnace", "BattleRam", "ThreeMusketeers", "HogRider", "RoyalHogs",
    "RoyalDelivery", "Hunter", "ElixirCollector", "LavaHound", "MinionGiant", "ArcherQueen",
    "Sparky", "TheLog", "BarbarianBarrel", "Tesla", "ElixirGolem", "MagicArcher", "GoblinHut",
    "SkeletonBarrel", "BattleHealer", "Bandit", "RoyalGhost", "ElectroGiant", "GoblinGiant",
    "MegaKnight", "Miner", "GoblinDemolisher", "Tornado", "CannonCart", "Graveyard", "Phoenix",
    "Void", "RamRider", "SuspiciousBush", "MotherWitch", "Fisherman", "GoblinDrill", "GoblinCurse",
    "Ronin", "Vines", "GoldenKnight", "Mirror", "SkeletonKing", "MergeMaiden", "LittlePrince",
    "BossBandit", "MightyMiner", "Clone", "GoblinMachine", "GiantBuffer", "Monk", "Goblinstein",
)
N_CARDS = 160
# Where the game tables name a card differently.
ROW_NAMES = {
    "Archers": "Archer", "EliteBarbarians": "AngryBarbarians", "DartGoblin": "BlowdartGoblin",
    "Guards": "SkeletonWarriors", "FlyingMachine": "DartBarrell", "NightWitch": "DarkWitch",
    "Lumberjack": "RageBarbarian", "IceGolem": "IceGolemite", "Executioner": "AxeMan",
    "FireSpirit": "FireSpirits", "IceSpirit": "IceSpirits", "WallBreakers": "Wallbreakers",
    "GiantSnowball": "Snowball", "HealSpirit": "Heal", "Zappies": "MiniSparkys",
    "Furnace": "FirespiritHut", "ElixirCollector": "Elixir Collector", "Sparky": "ZapMachine",
    "TheLog": "Log", "BarbarianBarrel": "BarbLog", "MagicArcher": "EliteArcher",
    "SkeletonBarrel": "SkeletonBalloon", "Bandit": "Assassin", "RoyalGhost": "Ghost",
    "CannonCart": "MovingCannon", "Void": "DarkMagic", "MotherWitch": "WitchMother",
}
DEFAULT_DECK = ("Knight", "Archers", "Giant", "Musketeer", "Skeletons", "Fireball", "Arrows", "Zap")
TICK_MS = 50
ACTIONS = 4611
# How long each probed card's battle runs, in decisions.
PROBE_DECISIONS = 700
JVM_ARGS = "-XX:+UseParallelGC -XX:+DisableExplicitGC -Xms6g -Xmx6g -Xmn4g"

_vector_class = None
_catalogs: dict = {}


def locate(install: str | None, tables: str | None) -> tuple[str, str]:
    install = install or os.environ.get("CRFORGE_INSTALL")
    tables = tables or os.environ.get("CRFORGE_TABLES")
    if not install or not tables:
        raise RuntimeError(
            "the CRForge simulator needs crforge_install (the env module's installed distribution) "
            "and crforge_tables (a data version's game tables), or CRFORGE_INSTALL and CRFORGE_TABLES"
        )
    return os.path.abspath(os.path.expanduser(install)), os.path.abspath(os.path.expanduser(tables))


def bridge_class(install: str):
    """Starts the JVM once, with CRForge's jars and the bridge class on its class path."""
    global _vector_class
    if _vector_class is not None:
        return _vector_class
    import jpype

    classes = os.environ.get("CFBOT_CRFORGE_CLASSES") or os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "build", "crforge"
    )
    if not os.path.exists(os.path.join(classes, "cfbot", "bridge", "CfVector.class")):
        raise RuntimeError(f"no bridge class under {classes}: run scripts/build_crforge.sh {install}")
    jars = sorted(glob.glob(os.path.join(install, "lib", "*.jar")))
    if not jars:
        raise RuntimeError(f"no jars under {install}/lib: run ./gradlew :env:installDist in CRForge")
    if not jpype.isJVMStarted():
        # The battle core allocates steadily on every thread (about 2 GB a
        # second for 256 battles). The throughput collector steps a batch about
        # a fifth faster than the JVM's default (6.9 against 8.9 ms on an M2
        # Max). JPype asks for a full collection whenever Python's own
        # collector runs, a pause of about 40 ms each time that frees nothing
        # here, so explicit collections are off; a fixed heap with a large
        # young space makes the remaining collections rarer. CFBOT_JVM_ARGS
        # replaces these options (separated by spaces).
        extra = os.environ.get("CFBOT_JVM_ARGS", JVM_ARGS).split()
        jpype.startJVM(
            "--enable-native-access=ALL-UNNAMED", *extra, classpath=[classes, *jars], convertStrings=False
        )
    _vector_class = jpype.JClass("cfbot.bridge.CfVector")
    return _vector_class


def _view(buffer, dtype) -> np.ndarray:
    array = np.frombuffer(memoryview(buffer), dtype=dtype)
    if array.flags["OWNDATA"]:
        raise RuntimeError("a JVM buffer was copied, not viewed")
    return array


def _rows() -> list:
    rows = [None] * N_CARDS
    for token, name in enumerate(CARD_NAMES):
        if name:
            rows[token] = ROW_NAMES.get(name, name)
    return rows


def _tokens(names) -> list[int]:
    out = []
    for name in names:
        if name not in CARD_NAMES or not name:
            raise ValueError(f"unknown card: {name}")
        out.append(CARD_NAMES.index(name))
    return out


def catalog(install: str, tables: str, level: int, decision_ticks: int) -> dict:
    """What is the same for every vector on these tables: the static card and
    unit features, which cards the battle core plays, and the match clock.
    Worked out once per process; the card probe plays every card in a battle."""
    key = (install, tables, level, decision_ticks)
    if key in _catalogs:
        return _catalogs[key]
    import jpype

    vector_class = bridge_class(install)
    probe = vector_class(
        tables, 1, os.cpu_count() or 1, decision_ticks, 0, 16, level, 0,
        jpype.JArray(jpype.JString)(_rows()), jpype.JArray(jpype.JInt)(_tokens(DEFAULT_DECK)),
    )
    try:
        status = np.array(probe.probeCards(PROBE_DECISIONS), dtype=np.int64)
        clock = [int(value) for value in probe.timeline()]
        n_units = -(-int(probe.nUnits()) // 32) * 32
        n_hazards = -(-int(probe.nHazards()) // 32) * 32
        unit_table = np.zeros((n_units, int(vector_class.UNIT_F)), dtype=np.float32)
        raw = np.array(probe.unitTable(), dtype=np.float32).reshape(-1, unit_table.shape[1])
        unit_table[: len(raw)] = raw
        unit_names = [str(name) for name in probe.unitNames()] + [""] * (n_units - len(raw))
        card_table = np.array(probe.cardTable(), dtype=np.float32).reshape(N_CARDS, -1).copy()
        supported = status == 1
        card_table[~supported] = 0.0
        # The probe sees each change one tick after the tick it takes effect on.
        double, triple, overtime, end, _bar = clock
        double, triple, overtime = double - 1, triple - 1, overtime - 1
        overtime = overtime if overtime > 0 else end
        out = {
            "status": status,
            "supported": supported,
            "card_table": card_table,
            "unit_table": unit_table,
            "unit_names": unit_names,
            "n_units": n_units,
            "n_hazards": n_hazards,
            "cost": np.array(probe.cardCosts(), dtype=np.int64),
            "kind": np.array(probe.cardKinds(), dtype=np.int64),
            "champion": np.array(probe.cardChampions(), dtype=bool),
            "max_elixir": int(probe.maxElixir()),
            "regular_ticks": overtime,
            "overtime_ticks": max(end - overtime, 0),
            "double_elixir_tick": double if double > 0 else end,
            "triple_elixir_tick": triple if triple > 0 else end,
            "data_version": str(probe.dataVersion()),
            "digest": str(probe.vocabularyHash()),
        }
    finally:
        probe.close()
    _catalogs[key] = out
    return out


class Sampler:
    """Decks for new battles: a pool deck, a pool deck with one to three cards
    swapped, or a random playable deck. The same rules as the other
    simulator's sampler (``src/decks.rs``)."""

    def __init__(self, facts: dict, exclude: set[int], pool, pool_prob: float, mirror: bool):
        self.facts = facts
        self.exclude = exclude
        self.cards = [t for t in range(1, N_CARDS) if facts["supported"][t] and t not in exclude]
        self.pool: list[list[int]] = []
        self.cumulative = np.zeros(0)
        self.pool_prob = pool_prob
        self.mutant_prob = 0.0
        self.mirror = mirror
        self.default = _tokens(DEFAULT_DECK)
        if pool:
            self.set_pool(pool, None)
        if not self.pool and len(self.cards) < 8:
            raise ValueError("fewer than eight cards are allowed and there is no deck pool")

    def check(self, deck, pool: bool = False) -> list[int]:
        deck = [int(token) for token in deck]
        if len(deck) != 8 or len(set(deck)) != 8:
            raise ValueError("a deck is eight different cards")
        for token in deck:
            if not 0 < token < N_CARDS or not self.facts["supported"][token]:
                name = CARD_NAMES[token] if 0 < token < len(CARD_NAMES) else token
                raise ValueError(f"the simulator does not play this card: {name}")
            if pool and token in self.exclude:
                raise ValueError(f"a pool deck holds an excluded card: {CARD_NAMES[token]}")
        if sum(bool(self.facts["champion"][token]) for token in deck) > 2:
            raise ValueError("a deck holds at most two champions")
        return deck

    def set_pool(self, decks, weights) -> None:
        pool = [self.check(deck, pool=True) for deck in decks]
        if weights is not None and len(weights) != len(pool):
            raise ValueError("one weight per pool deck")
        w = np.ones(len(pool)) if weights is None else np.maximum(np.asarray(weights, dtype=float), 0)
        if w.sum() <= 0:
            w = np.ones(len(pool))
        self.pool = pool
        self.cumulative = np.cumsum(w / w.sum())

    def playable(self, deck) -> bool:
        kind, champion, cost = self.facts["kind"], self.facts["champion"], self.facts["cost"]
        champions = sum(bool(champion[t]) for t in deck)
        spells = sum(kind[t] == 3 for t in deck)
        buildings = sum(kind[t] == 2 for t in deck)
        troops = sum(kind[t] == 1 and not champion[t] for t in deck)
        total = sum(int(cost[t]) for t in deck)
        return champions <= 1 and 1 <= spells <= 3 and buildings <= 2 and troops >= 4 and 21 <= total <= 37

    def random(self, rng) -> list[int]:
        if len(self.cards) >= 8:
            for _ in range(64):
                deck = [int(t) for t in rng.choice(self.cards, size=8, replace=False)]
                if self.playable(deck):
                    return deck
        return list(self.default)

    def pool_deck(self, rng) -> list[int]:
        index = int(np.searchsorted(self.cumulative, rng.random(), side="right"))
        return list(self.pool[min(index, len(self.pool) - 1)])

    def mutant(self, rng) -> list[int]:
        base = self.pool_deck(rng)
        if not self.cards:
            return base
        for _ in range(32):
            deck = list(base)
            draw = rng.random()
            for _ in range(1 if draw < 0.5 else 2 if draw < 0.75 else 3):
                card = int(rng.choice(self.cards))
                if card not in deck:
                    deck[int(rng.integers(8))] = card
            if deck != base and self.playable(deck):
                return deck
        return base

    def one(self, rng) -> list[int]:
        draw = rng.random()
        if self.pool and (len(self.cards) < 8 or draw < self.pool_prob):
            return self.pool_deck(rng)
        if self.pool and draw < self.pool_prob + self.mutant_prob:
            return self.mutant(rng)
        return self.random(rng)

    def pair(self, rng) -> list[int]:
        first = self.one(rng)
        return first + (list(first) if self.mirror else self.one(rng))


class CrforgeEnv(Env):
    """Batch of CRForge battles. Rows are ``env * 2 + side``, as in ``Env``."""

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
        install: str | None = None,
        tables: str | None = None,
        level: int = 11,
    ):
        import jpype

        if rules is not None or level_jitter is not None:
            raise ValueError("rules and level_jitter belong to the other simulator; CRForge plays its tables")
        install, tables = locate(install, tables)
        vector_class = bridge_class(install)
        self.facts = catalog(install, tables, level, decision_ticks)
        self.layout = vector_class
        self.num_envs = num_envs
        self.rows = num_envs * 2
        self.device = torch.device(device)
        self.master_seed = int(seed)
        self.episode = np.zeros(num_envs, dtype=np.int64)
        self.sampler = Sampler(
            self.facts, set(_tokens(exclude_cards)),
            [_tokens(deck) for deck in deck_pool] if deck_pool else None, pool_prob, mirror_decks,
        )
        self.env = vector_class(
            tables, num_envs, workers or (os.cpu_count() or 1), decision_ticks, max_skip, max_entities,
            level, 0, jpype.JArray(jpype.JString)(_rows()), jpype.JArray(jpype.JInt)(_tokens(DEFAULT_DECK)),
        )
        v = self.env
        names = jpype.JClass("org.crforge.env.Schema").refusalNames()
        self._refusal_names = [str(name) if name is not None else "" for name in names]
        self.float_row, self.byte_row = int(v.floatRow()), int(v.byteRow())
        self._floats = _view(v.floats(), np.float32).reshape(self.rows, self.float_row)
        self._bytes = _view(v.bytes(), np.uint8).reshape(self.rows, self.byte_row)
        self._actions = _view(v.actions(), np.int32)
        self._row_ints = _view(v.rowInts(), np.int32).reshape(self.rows, int(vector_class.ROW_INTS))
        self._row_floats = _view(v.rowFloats(), np.float32).reshape(self.rows, int(vector_class.ROW_FLOATS))
        self._row_flags = _view(v.rowFlags(), np.uint8).reshape(self.rows, int(vector_class.ROW_FLAGS))
        self._env_ints = _view(v.envInts(), np.int32).reshape(num_envs, int(vector_class.ENV_INTS))
        self._env_flags = _view(v.envFlags(), np.uint8).reshape(num_envs, int(vector_class.ENV_FLAGS))
        self._decks = _view(v.decks(), np.int16).reshape(num_envs, 2, 8)
        self.max_entities = max_entities
        self._planes = self._plane_layout(vector_class)
        self.spec = self._spec(vector_class, decision_ticks, level, max_entities)
        self._jint = jpype.JArray(jpype.JInt)
        self._refill(np.arange(num_envs))
        self.env.reset()
        self._refill()

    # ---- layout ----

    @staticmethod
    def _plane_layout(c) -> dict:
        """Name, offset and shape of each fixed plane in a row of the two blocks."""
        floats = {
            "global": (int(c.O_GLOBAL), (int(c.GLOBAL_F),)),
            "abilities": (int(c.O_ABILITY), (int(c.ABILITY_ROWS), int(c.ABILITY_F))),
            "hand": (int(c.O_HAND), (4, int(c.HAND_F))),
            "own_deck": (int(c.O_DECK), (int(c.DECK_ROWS), int(c.DECK_F))),
            "deck_extras": (int(c.O_DECK_X), (int(c.DECK_ROWS), int(c.DECK_X_F))),
            "recent_plays": (int(c.O_PLAYS), (int(c.PLAY_ROWS), int(c.PLAY_F))),
            "revealed_deck": (int(c.O_REVEAL), (int(c.REVEAL_ROWS), int(c.REVEAL_F))),
            "reveal_extras": (int(c.O_REVEAL_X), (int(c.REVEAL_ROWS), int(c.REVEAL_X_F))),
            "hazards": (int(c.O_HAZARD), (int(c.HAZARD_ROWS), int(c.HAZARD_F))),
            "extras": (int(c.O_EXTRAS), (int(c.EXTRA_F),)),
            "priv_global": (int(c.O_PRIV_GLOBAL), (int(c.PRIV_GLOBAL_F),)),
            "priv_hand": (int(c.O_PRIV_HAND), (4,)),
            "priv_deck": (int(c.O_PRIV_DECK), (int(c.DECK_ROWS),)),
        }
        masks = {
            "play_mask": (int(c.B_PLAY_MASK), (int(c.PLAY_ROWS),)),
            "reveal_mask": (int(c.B_REVEAL_MASK), (int(c.REVEAL_ROWS),)),
            "hazard_mask": (int(c.B_HAZARD_MASK), (int(c.HAZARD_ROWS),)),
            "legal": (int(c.B_LEGAL), (int(c.ACTIONS),)),
        }
        return {
            "floats": floats,
            "bytes": masks,
            "entity": int(c.O_ENTITY),
            "entity_row": int(c.ENTITY_ROW),
            "entity_f": int(c.ENTITY_F),
            "entity_mask": int(c.B_ENTITY_MASK),
        }

    def _spec(self, c, decision_ticks: int, level: int, max_entities: int) -> dict:
        f = self.facts
        names = list(CARD_NAMES) + [""] * (N_CARDS - len(CARD_NAMES))
        return {
            "simulator": "crforge",
            "schema_version": 2,
            "bot_schema": 3,
            "rules_id": f"crforge-{f['data_version']}-level{level}",
            "rules_digest": f["digest"],
            "base_level": level,
            "tick_ms": TICK_MS,
            "decision_ticks": decision_ticks,
            "regular_ticks": f["regular_ticks"],
            "overtime_ticks": f["overtime_ticks"],
            "double_elixir_tick": f["double_elixir_tick"],
            "triple_elixir_tick": f["triple_elixir_tick"],
            "max_elixir": f["max_elixir"],
            "num_actions": ACTIONS,
            "grid_width": 18,
            "grid_height": 32,
            "play_action_count": 2305,
            "secondary_play_start": 2307,
            "max_entities": max_entities,
            "global_f": int(c.GLOBAL_F),
            "entity_f": int(c.ENTITY_F),
            "entity_x_f": int(c.ENTITY_X_F),
            "hand_f": int(c.HAND_F),
            "deck_f": int(c.DECK_F),
            "deck_x_f": int(c.DECK_X_F),
            "play_rows": int(c.PLAY_ROWS),
            "play_f": int(c.PLAY_F),
            "reveal_rows": int(c.REVEAL_ROWS),
            "reveal_f": int(c.REVEAL_F),
            "reveal_x_f": int(c.REVEAL_X_F),
            "hazard_rows": int(c.HAZARD_ROWS),
            "hazard_f": int(c.HAZARD_F),
            "extra_f": int(c.EXTRA_F),
            "ability_f": int(c.ABILITY_F),
            "priv_global_f": int(c.PRIV_GLOBAL_F),
            "card_f": int(c.CARD_F),
            "unit_f": int(c.UNIT_F),
            "n_cards": N_CARDS,
            "n_units": f["n_units"],
            "n_hazards": f["n_hazards"],
            "card_names": names,
            "card_supported": [bool(value) for value in f["supported"]],
            "unit_names": f["unit_names"],
            "card_table": f["card_table"],
            "unit_table": f["unit_table"],
            "pool_decks": len(self.sampler.pool),
            "random_deck_cards": len(self.sampler.cards),
        }

    # ---- matches ----

    def _refill(self, indices=None) -> None:
        """Hands a sampled match to every battle that has used its last one."""
        if indices is None:
            indices = np.flatnonzero(self._env_flags[:, 1])
        if not len(indices):
            return
        decks, seeds = [], []
        for index in indices:
            rng = np.random.default_rng([self.master_seed, int(index), int(self.episode[index])])
            self.episode[index] += 1
            decks.extend(self.sampler.pair(rng))
            seeds.append(int(rng.integers(1, 2**31 - 1)))
        self.env.setNext(self._jint([int(i) for i in indices]), self._jint(decks), self._jint(seeds))

    def set_mix(self, pool_prob: float, mutant_prob: float = 0.0) -> None:
        self.sampler.pool_prob, self.sampler.mutant_prob = pool_prob, mutant_prob

    def set_pool(self, decks: list[list[str]], weights: list[float] | None = None) -> None:
        self.sampler.set_pool([_tokens(deck) for deck in decks], weights)

    def queue_match(self, env: int, blue: list[int], red: list[int], seed: int | None = None) -> None:
        if not 0 <= env < self.num_envs:
            raise ValueError("no such battle")
        decks = self.sampler.check(blue) + self.sampler.check(red)
        if seed is None:
            seed = int(np.random.default_rng([self.master_seed, env, int(self.episode[env]), 1]).integers(2**31 - 1))
        self.env.queueMatch(env, self._jint(decks), int(seed) % (2**31 - 1))

    def random_decks(self, count: int, seed: int) -> list[list[int]]:
        rng = np.random.default_rng([int(seed), 0x5EED])
        return [self.sampler.random(rng) for _ in range(count)]

    def crashes(self):
        flat = np.array(self.env.crashes(), dtype=np.int64).reshape(-1, 2, 8)
        return [([CARD_NAMES[t] for t in pair[0]], [CARD_NAMES[t] for t in pair[1]]) for pair in flat]

    def deck_retries(self) -> int:
        return int(self.env.deckRetries())

    def refusals(self) -> list[dict]:
        """Plays the simulator refused although the legal mask offered them."""
        names = self._refusal_names
        rows = np.array(self.env.refusals(), dtype=np.int64).reshape(-1, 5)
        return [
            {"reason": names[code] if 0 <= code < len(names) and names[code] else str(code),
             "card": CARD_NAMES[card], "action": int(action), "tick": int(tick), "elixir": elixir / 10000}
            for code, card, action, tick, elixir in rows.tolist()
        ]

    def shots_dropped(self) -> int:
        """Units' and towers' shots left out of the hazard plane for want of a row. Spell areas
        and the projectiles cards cast come first; one of those left out counts in
        ``hazard_overflow`` instead."""
        return int(self.env.shotsDropped())

    def hazard_ranks(self) -> dict[str, int]:
        """Each hazard's rank for a row of the plane: 0 an area, 1 a card's projectile, 2 a
        shot, 3 decoration. A crown tower's shot reads 2 here and is ranked 3 in play."""
        return {str(name): int(rank) for name, rank in zip(self.env.hazardNames(), self.env.hazardRanks()) if name}

    def run_refusals(self) -> int:
        """Plays accepted when given that were refused when they ran, a second later."""
        return int(self.env.runRefusals())

    def run_refused(self) -> list[dict]:
        rows = np.array(self.env.runRefused(), dtype=np.int64).reshape(-1, 4)
        return [{"code": int(code), "card": CARD_NAMES[card], "action": int(action), "tick": int(tick)}
                for code, card, action, tick in rows.tolist()]

    # ---- stepping ----

    def reset(self, seed: int | None = None) -> dict[str, torch.Tensor]:
        if seed is not None:
            self.master_seed = int(seed)
            self.episode[:] = 0
            self._refill(np.arange(self.num_envs))
        self.env.reset()
        self._refill()
        return self._observe()

    def step(self, actions: torch.Tensor) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray]]:
        self._actions[:] = actions.detach().to("cpu", torch.int32).numpy()
        self.env.step()
        self._refill()
        floats, flags, ints = self._row_floats, self._row_flags, self._row_ints
        info = {
            "reward": floats[:, 0].copy(),
            "trade": floats[:, 1].copy(),
            "leak": floats[:, 2].copy(),
            "ticks": floats[:, 3].copy(),
            "final_towers": floats[:, 4:10].copy(),
            "done": flags[:, 1].astype(bool),
            "truncated": flags[:, 2].astype(bool),
            "verdict": flags[:, 3].astype(np.int8),
            "outcome": self._env_ints[:, 0].astype(np.int8),
            "episode_decisions": self._env_ints[:, 1].astype(np.uint32),
            "episode_ticks": self._env_ints[:, 2].astype(np.uint32),
            "entity_overflow": ints[:, 1].astype(np.uint32),
            "hazard_overflow": ints[:, 2].astype(np.uint32),
            "decks": self._decks.copy(),
            "queued": self._env_flags[:, 0].astype(bool),
        }
        return self._observe(), info

    def _views(self, floats: torch.Tensor, masks: torch.Tensor, width: int) -> dict[str, torch.Tensor]:
        """The planes as views of one float block and one byte block."""
        p = self._planes
        rows = floats.shape[0]
        out = {}
        for name, (offset, shape) in p["floats"].items():
            out[name] = floats[:, offset : offset + int(np.prod(shape))].view(rows, *shape)
        for name, (offset, shape) in p["bytes"].items():
            out[name] = masks[:, offset : offset + int(np.prod(shape))].view(rows, *shape)
        entity = floats[:, p["entity"] : p["entity"] + width * p["entity_row"]].view(rows, width, p["entity_row"])
        out["entities"] = entity[..., : p["entity_f"]]
        out["entity_extras"] = entity[..., p["entity_f"] :]
        out["entity_mask"] = masks[:, p["entity_mask"] : p["entity_mask"] + width]
        return out

    def _observe(self) -> dict[str, torch.Tensor]:
        p = self._planes
        counts = self._row_ints[:, 0]
        width = entity_width(int(counts.max()), self.max_entities)
        # One copy of each block, cut to the entity width in use: the JVM writes
        # over its buffers at the next step, and the collector keeps these.
        floats = torch.from_numpy(np.ascontiguousarray(self._floats[:, : p["entity"] + width * p["entity_row"]]))
        masks = torch.from_numpy(np.ascontiguousarray(self._bytes[:, : p["entity_mask"] + width]))
        self._blocks = (floats, masks, width)
        self.last_cpu = self._views(floats, masks, width)
        self.last_choice = torch.from_numpy(self._row_flags[:, 0].astype(bool))
        self.last_entity_count = torch.from_numpy(counts.astype(np.int64))
        return self.upload(self.last_cpu, packed=True)

    def upload(self, obs: dict[str, torch.Tensor], packed: bool = False) -> dict[str, torch.Tensor]:
        """Copy CPU planes to the device. This step's own planes go up as the
        two blocks they already are."""
        if self.device.type == "cpu" or not packed or obs is not self.last_cpu:
            return super().upload(obs, packed)
        floats, masks, width = self._blocks
        lazy = self.device.type == "cuda"
        floats = floats.to(self.device, non_blocking=lazy)
        masks = masks.to(self.device, non_blocking=lazy)
        out = self._views(floats, masks, width)
        self._uploaded = (out, floats, masks, width)
        self._packed = None
        return out

    def take(self, obs: dict[str, torch.Tensor], index: torch.Tensor, keys) -> dict[str, torch.Tensor]:
        uploaded = getattr(self, "_uploaded", None)
        if uploaded is None or obs is not uploaded[0]:
            return super().take(obs, index, keys)
        _, floats, masks, width = uploaded
        rows = self._views(floats.index_select(0, index), masks.index_select(0, index), width)
        return {key: rows[key] for key in keys}
