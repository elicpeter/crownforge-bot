"""Deterministic Clash Royale benchmark policies for the Arena simulator."""

from __future__ import annotations

import numpy as np
import torch

from . import actions as A


BRIDGE_X = (3.5, 14.5)
# Arena card feature offsets from arena/src/feat.rs. Troop stats use this
# same layout in the per-unit table, starting at column zero.
TROOP, BUILDING, SPELL = 2, 3, 4
HP, DAMAGE, RANGE, HIT_TIME, AIR_TARGET, GROUND_TARGET, FLYING, BUILDINGS_ONLY, SPLASH = (
    0, 2, 3, 5, 6, 7, 8, 9, 10
)
CARD_STATS = 8
# Travel times are estimates for aiming at visible moving units. Log instead
# sweeps a lane toward the opponent starting from the chosen tile.
SPELL_TRAVEL = {
    "fireball": 0.8, "zap": 0.2, "royal_delivery": 0.5,
    "rocket": 1.2, "lightning": 0.5, "tornado": 0.2, "the_log": 0.4, "poison": 0.4,
}
DEFENSIVE_BUILDINGS = {"cannon", "tesla", "inferno_tower"}
CHEAP_CYCLE = ("skeletons", "ice_spirit")


class ArenaBenchmarkBot:
    """A scripted player with one fixed strategy and no hidden-state inputs."""

    def __init__(self, name: str, spec: dict):
        if spec.get("simulator") != "arena":
            raise ValueError(f"{name} requires the arena simulator")
        self.name = name
        self.cards = spec["card_table"]
        self.units = spec["unit_table"]
        self.names = spec["card_names"]
        self.unit_names = spec["unit_names"]
        self.max_hp = np.rint(np.expm1(self.units[:, HP] * 9))
        self.damage = np.rint(np.expm1(self.units[:, DAMAGE] * 8))
        self.spell_damage = np.rint(np.expm1(self.cards[:, 22] * 8))
        self.double_elixir = spec["double_elixir_tick"] * spec["tick_ms"] / 1000
        self.overtime = spec["regular_ticks"] * spec["tick_ms"] / 1000

    def act(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        legal = obs["legal"]
        plays = legal[:, 1:A.PLAY_ACTION_COUNT].reshape(-1, A.HAND_SLOTS, A.CELLS)
        masks = plays.detach().cpu().numpy()
        actions = np.zeros(len(masks), dtype=np.int64)
        active = np.flatnonzero(masks.any(axis=(1, 2)))
        if len(active):
            # Only copy the rows with playable cards and the visible planes
            # used by these policies. Finished battles can reuse the same bot.
            rows = torch.as_tensor(active, device=legal.device)
            keys = ("global", "hand", "entities", "entity_extras", "entity_mask")
            visible = [obs[key].index_select(0, rows).detach().cpu().numpy() for key in keys]
            for row, (clock, hand, entities, extras, present) in zip(active, zip(*visible)):
                alive = present.astype(bool) & (entities[:, 23] > 0)
                actions[row] = self._choose(clock, hand, entities[alive], extras[alive], masks[row])
        return torch.from_numpy(actions).to(legal.device)

    def _choose(self, clock, hand, entities, extras, masks) -> int:
        seconds, elixir = clock[:2]
        # Use resolved card stats for evolutions and the base name for strategy.
        slots = {
            self.names[int(hand[slot, 0])]: slot
            for slot in np.flatnonzero(masks.any(axis=1))
        }
        kind, mine, token = entities[:, :3].astype(int).T
        xy = entities[:, 4:6] * (A.WIDTH, A.HEIGHT)
        left = xy[:, 0] < A.WIDTH / 2
        hp = np.rint(self.max_hp[token] * entities[:, 6])
        troops = kind == 3
        ours = mine == 1
        enemies = ~ours
        towers = enemies & (kind <= 2)
        princesses = towers & (kind == 2)
        targets = np.flatnonzero(princesses if princesses.any() else towers)
        if not len(targets):
            return 0
        tower = targets[np.argmin(hp[targets])]
        lane = int(not left[tower]) if princesses.any() else 0
        if self.name == "bridge-pressure" and len(targets) == 2:
            # Punish an expensive backline commitment in the other lane.
            backline = enemies & troops & (xy[:, 1] >= 21) & (self.cards[token, 1] >= 0.5)
            if backline.any():
                heavy = np.argmax(np.where(backline, hp, -1))
                lane = int(left[heavy])
                tower = next(index for index in targets if int(not left[index]) == lane)

        # Troops in or approaching our half. A building usually threatens only
        # from across the river if it is actual siege.
        incoming = enemies & (
            (troops & (xy[:, 1] < 17.3)) |
            ((kind == 4) & (self.units[token, RANGE] >= 1.0) & (xy[:, 1] < 20))
        )
        active = extras[:, 0] == 0
        enemy_air = self.units[token, FLYING] > 0
        if self.name == "graveyard-control":
            x = xy[tower, 0] if princesses.any() else 6.5
            in_range = abs(xy[:, 0] - x) < 5
            in_enemy_half = xy[:, 1] > 19
            enemy_splash = enemies & (kind >= 3) & in_range & in_enemy_half & (self.units[token, SPLASH] > 0.15)
            tanking = ours & troops & in_enemy_half & (abs(xy[:, 0] - x) < 4) & (hp > 350)
            close_hp = hp[enemies & (kind >= 3) & (xy[:, 1] < 12)].sum()
            if ("graveyard" in slots and close_hp < 2700
                    and (tanking.any() or (elixir >= 7 and not enemy_splash.any()))):
                tx, ty = xy[tower]
                return self._at(masks, slots["graveyard"], tx + (1 if tx > 9 else -1), ty + 1)
            if "goblin_hut" in slots and elixir >= 6 and close_hp < 950:
                return self._at(masks, slots["goblin_hut"], 7.5 if lane == 0 else 10.5, 9.5)
        defense = self._defense(
            slots, hand, incoming, ours & troops, kind, token, xy, hp, active, enemy_air,
        )
        spell = self._spell(
            slots, hand, masks, enemies, kind, token, xy, hp, active, extras[:, 3:5],
            incoming, enemy_air, tower, seconds, elixir,
        )
        if spell and (defense is None or spell[0] > defense[0]):
            return spell[1]
        if defense is not None:
            _, slot, x, y = defense
            play = self._at(masks, slot, x, y)
            if play:
                return play

        x = BRIDGE_X[lane]
        in_lane = left == (lane == 0)
        own_push = ours & troops & in_lane & (xy[:, 1] >= 10)
        lane_enemy = enemies & (kind >= 3) & in_lane & (xy[:, 1] < 24)
        # Keep the pressure on the remaining Princess. With neither standing,
        # use the open land pocket to shorten the walk toward the King.
        front = 14.5 if princesses.any() else 20.5
        multi = seconds >= self.double_elixir
        play = None

        if self.name == "hog-cycle":
            if self._affords("hogrider", slots, hand, elixir, reserve=1):
                # A cheap lead play can distract defenders without delaying
                # the Hog through another elixir cycle.
                if (self._affords("ice_spirit", slots, hand, elixir, reserve=5)
                        and not own_push.any() and lane_enemy.any()):
                    play = ("ice_spirit", x, front)
                else:
                    play = ("hogrider", x, front)
            elif elixir >= 8 and "hogrider" not in slots:
                play = self._cycle(slots, x, ("musketeer", "ice_golem"))

        elif self.name == "golem-beatdown":
            tanks = ours & troops & (token == self.names.index("golem"))
            if tanks.any():
                tank = np.argmax(np.where(tanks, xy[:, 1], -1))
                tank_x, tank_y = xy[tank]
                support = ("night_witch", "babydragon", "lumberjack", "mega_minion")
                for card in support:
                    if tank_y >= 7 and self._affords(card, slots, hand, elixir, reserve=0.5):
                        play = (card, tank_x, min(14.5, tank_y - 3))
                        break
            elif "golem" in slots and elixir >= (8 if multi else 9.7):
                play = ("golem", x, 2.5)
            elif elixir >= 9.5 and "golem" not in slots:
                play = self._cycle(slots, x, ("mega_minion", "babydragon", "night_witch", "lumberjack"))

        elif self.name == "bridge-pressure":
            if self._affords("battle_ram", slots, hand, elixir, reserve=1):
                play = ("battle_ram", x, front)
            elif own_push.any() or elixir >= (6 if multi else 7.5):
                support = ("dark_prince", "electro_wizard", "minions", "knight", "giant_skeleton")
                for card in support:
                    if self._affords(card, slots, hand, elixir, reserve=0.5 if own_push.any() else 2):
                        play = (card, x, front - (2 if own_push.any() else 0))
                        break

        elif self.name == "xbow-siege":
            siege = ours & (kind == 4) & (token == self.names.index("xbow"))
            if (self._affords("xbow", slots, hand, elixir, reserve=2 if multi else 2.8)
                    and not siege.any() and not lane_enemy.any()):
                play = ("xbow", x, front)
            elif siege.any() and lane_enemy.any():
                for card in ("knight", "archers", "skeletons", "ice_spirit"):
                    if self._affords(card, slots, hand, elixir, reserve=1):
                        play = (card, x + (1 if lane == 0 else -1), 14.5)
                        break
            elif elixir >= 9.5:
                play = self._cycle(slots, x, ("archers", "knight"))

        elif self.name == "log-bait":
            if self._affords("goblin_barrel", slots, hand, elixir, reserve=2):
                # Lead with a second threat when both fit. The delayed Barrel
                # then forces a separate answer at the Princess Tower.
                if (self._affords("princess", slots, hand, elixir, reserve=4)
                        and not own_push.any() and not lane_enemy.any()):
                    play = ("princess", x, 14.5)
                else:
                    play = ("goblin_barrel", *xy[tower])
            elif elixir >= 7:
                if "princess" in slots:
                    play = ("princess", x, 14.5 if not lane_enemy.any() else 9.5)
                else:
                    play = self._cycle(slots, x, ("goblin_gang", "knight"))

        elif self.name == "graveyard-control" and elixir >= 6:
            card = next((card for card in ("knight", "babydragon") if card in slots), None)
            if card is not None:
                play = (card, x, 14.5)
            elif elixir >= 8:
                play = self._cycle(slots, x, ("ice_wizard",))

        if play is None:
            return 0
        card, x, y = play
        return self._at(masks, slots[card], x, y)

    @staticmethod
    def _affords(card, slots, hand, elixir, reserve):
        return card in slots and elixir >= hand[slots[card], 3] + reserve

    @staticmethod
    def _cycle(slots, x, support):
        card = next((card for card in (*CHEAP_CYCLE, *support) if card in slots), None)
        if card is None:
            return None
        return (card, x, 14.5 if card == "ice_spirit" else 2.5)

    @staticmethod
    def _at(masks, slot, x, y) -> int:
        col = int(np.clip(x, 0, A.WIDTH - 1))
        row = int(np.clip(y, 0, A.HEIGHT - 1))
        cell = row * A.WIDTH + col
        return 1 + slot * A.CELLS + cell if masks[slot, cell] else 0

    def _defense(self, slots, hand, incoming, friendly, kind, token, xy, hp, active, air):
        if not incoming.any():
            return None
        # Advance, remaining health and damage determine which threat needs
        # attention first. Nearby troops that can respond reduce its priority.
        pressure = (2 + np.clip((16 - xy[:, 1]) * 0.27, -0.3, 2.7)
                    + np.minimum(2.2, hp / 950) + np.minimum(1.2, self.damage[token] / 330))
        best = None
        for target in np.flatnonzero(incoming):
            tx, ty = xy[target]
            unit = self.units[token[target]]
            if ty >= 15 and self.unit_names[token[target]] in {"bandit", "boss_bandit"}:
                continue  # The pinned simulator can leave an invulnerable dash stalled here.
            distance = np.hypot(xy[:, 0] - tx, xy[:, 1] - ty)
            aim = AIR_TARGET if air[target] else GROUND_TARGET
            nearby = (friendly & (self.units[token, aim] > 0)
                      & (self.units[token, BUILDINGS_ONLY] == 0) & (distance <= 4.5))
            cover = self.damage[token[nearby]] * np.where(active[nearby], 1.0, 0.7)
            cover = float(cover.sum()) / max(200, min(550, hp[target]))
            if unit[BUILDINGS_ONLY] > 0:
                cover *= 0.65
            nearby_threats = incoming & (distance <= 2.6)
            for name, slot in slots.items():
                card = int(hand[slot, 1])
                stats = self.units[card]
                cost = hand[slot, 3]
                if name in DEFENSIVE_BUILDINGS:
                    if ty < 8 or unit[BUILDINGS_ONLY] == 0 or stats[aim] == 0:
                        continue
                    # This central placement pulls riders and tanks away
                    # from the threatened Princess and into tower range.
                    x, y = (7.5 if tx < 9 else 10.5), 11.5
                    value = 3.6 + min(1.5, hp[target] / 950) - 0.28 * cost
                elif self.cards[card, TROOP] and stats[BUILDINGS_ONLY] == 0 and stats[aim] > 0:
                    ranged = stats[RANGE] >= 0.34
                    x, y = tx, np.clip(ty - (4.0 if ranged else 1.5), 2.5, 13.5)
                    if not ranged and abs(tx - 9) < 5:
                        x += 0.7 if tx < 9 else -0.7
                    hitters = np.rint(self.cards[card, 7] * 16)
                    cluster = 1 + min(2, int((nearby_threats & (air == air[target])).sum()) - 1)
                    damage = self.damage[card] * max(1, hitters)
                    damage *= cluster if stats[SPLASH] > 0.25 else 1
                    value = (min(3.4, damage / 90) * 0.73
                             + min(1.7, self.max_hp[card] * hitters / 640) * 0.45
                             + 0.45 * ranged - 0.42 * cost)
                    if name in {"ice_spirit", "skeletons"} and ty >= 15.5:
                        value -= 1.2
                    if not active[target] and ty >= 14.5:
                        value -= 0.45
                else:
                    continue
                score = pressure[target] + value - 2.6 * min(2.0, cover)
                if score >= 4.2 and (best is None or score > best[0]):
                    best = (score, slot, x, y)
        return best

    def _spell(self, slots, hand, masks, enemy, kind, token, xy, hp, active, velocity,
               incoming, air, tower, seconds, elixir):
        best = None
        for name, slot in slots.items():
            if name not in SPELL_TRAVEL:
                continue
            card = int(hand[slot, 1])
            cost = hand[slot, 3]
            damage = self.spell_damage[card]
            crown = int(damage * self.cards[card, 23] + 1e-4)
            # A moving troop may leave Poison after a few ticks. A tower can
            # take all eight. Tornado's damage also comes from its area effect.
            if name == "tornado":
                damage, crown = 180, 54
            elif name == "poison":
                damage, crown = 3 * 92, 8 * 20
            predicted = xy + velocity * (3 * SPELL_TRAVEL[name])
            candidates = np.flatnonzero(enemy & (kind >= 3))
            aims = [xy[tower], *predicted[candidates]]
            # Pairwise cluster centroids are sufficient to catch near misses
            # without evaluating every board tile on every decision.
            if len(candidates) > 1 and (incoming.any() or elixir >= 7):
                for index in candidates[:8]:
                    near = candidates[np.linalg.norm(predicted[candidates] - predicted[index], axis=1) <= 3.5]
                    if len(near) > 1:
                        aims.append(predicted[near].mean(axis=0))

            for aim in aims:
                x, y = np.floor(aim).clip((0, 0), (17, 31)) + 0.5
                if name == "the_log":
                    y = float(np.clip(y - 2, 0.5, 14.5))
                    hit = (abs(predicted[:, 0] - x) <= 2.0) & (predicted[:, 1] >= y - 0.4)
                    hit &= (predicted[:, 1] <= y + 10.1) & ~air
                else:
                    radius = float(self.cards[card, 24]) * 5
                    # Crown towers have a square hitbox. Troop centers are
                    # conservative for most splash spells.
                    extent = np.where(kind == 1, 2.0, np.where(kind == 2, 1.5, 0.0))
                    delta = np.maximum(abs(predicted - (x, y)) - extent[:, None], 0)
                    hit = (delta ** 2).sum(axis=1) <= radius ** 2
                hit &= enemy & active
                if name == "lightning":
                    hits = np.flatnonzero(hit)
                    hit[hits[np.argsort(-hp[hits], kind="stable")[3:]]] = False
                troops = hit & (kind >= 3)
                count = int(troops.sum())
                lethal = hit[tower] and crown >= hp[tower] and name != "the_log"
                if lethal:
                    score = 20.0
                else:
                    removed = np.minimum(1, damage / np.maximum(1, hp[troops]))
                    high_value = np.clip(self.max_hp[token[troops]] / 330, 0.45, 1.45)
                    score = float((removed * high_value).sum()) * 2.0 - 0.36 * cost
                    score += 1.1 * bool((troops & incoming).any())
                    if hit[tower] and crown > 0:
                        score += 1.0
                    # Late chip breaks an HP tie if we have spare elixir.
                    if count == 0 and hit[tower] and crown > 0 and elixir >= 9:
                        if seconds >= self.overtime and (kind[tower] == 2 or self.name == "log-bait"):
                            score = 4.35
                    if name == "tornado" and count < 3:
                        continue
                if score < 4.25 or (best is not None and score <= best[0]):
                    continue
                action = self._at(masks, slot, x, y)
                if action:
                    best = (score, action)
        return best
