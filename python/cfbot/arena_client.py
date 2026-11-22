"""Tournament client: plays a checkpoint on a cr-rl-arena server.

    python -m cfbot.arena_client wss://HOST NAME --checkpoint runs/main/checkpoint.pt \\
        --deck knight,archers,giant,musketeer,skeletons,fireball,arrows,zap

It joins the queue under ``NAME`` with the deck, answers every observation, and
keeps playing matches until stopped (Ctrl-C). If the connection drops it joins
again; a dropped connection during a match forfeits that match.

How it plays is how the checkpoint was trained. The server sends an observation
every tick; the policy decides on the ticks the training bridge decides on
(every ``decision_ticks``, 0.3 s of game time) and the client passes at once on
the others, as the server asks. The observation rows come from the same Rust
code as in training (``cfbot_arena.Featurizer``), fed with what the server
sends and nothing else.

A failure inside the policy never costs the turn: the client passes, prints the
error once, and goes on. Results go to ``--log`` as one JSON line per match.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import traceback

import torch

from .agent import Agent
from .arena import bridge
from .env import POLICY_KEYS
from .league import load_policy
from .search import Searcher


class Player:
    """One checkpoint and one deck: observation dict in, server action out."""

    def __init__(self, checkpoint: str, deck: list[str], device: str = "cpu", mode: str = "sample",
                 temperature: float = 1.0, seed: int = 0, ema: bool = False, search: dict | None = None):
        self.deck = list(deck)
        self.device = torch.device(device)
        data = torch.load(checkpoint, map_location="cpu", weights_only=False)
        config = data.get("config", {}) if isinstance(data, dict) else {}
        self.decision_ticks = int(config.get("decision_ticks", 6))
        max_entities = int(config.get("max_entities", 128))
        self.featurizer = bridge().Featurizer(self.deck, self.decision_ticks, max_entities)
        spec = self.featurizer.spec()
        model, _ = load_policy(checkpoint, spec, self.device, ema=ema)
        self.agent = Agent(model, mode=mode, seed=seed, temperature=temperature)
        # Look-ahead (cfbot.search): the settings of a Searcher, or None for the plain policy.
        self.searcher = None
        if search is not None:
            self.searcher = Searcher(model, temperature=temperature, greedy=mode == "greedy", seed=seed, **search)
        self.decisions = 0
        self.policy_seconds = 0.0
        self.slowest_seconds = 0.0
        self.failed = False

    def start(self, dealt: list[str] | None = None) -> list[str]:
        """Start a match on `dealt`, the deck the server handed out for it, else our own.

        Later games of a tournament series can be played on random decks the
        server deals in `start`; the observations then show those cards. Returns
        the deck the match is played on.
        """
        deck = list(dealt) if dealt else self.deck
        try:
            self.featurizer.reset(deck)
        except Exception:  # never lose the match to a deck we cannot read: say so, play our own
            traceback.print_exc()
            deck = self.deck
            self.featurizer.reset(deck)
        return deck

    def act(self, obs: dict):
        """The reply to one observation: ``[slot, x, y]`` or ``None``."""
        self.featurizer.update(obs)
        if obs["tick"] % self.decision_ticks or not self.featurizer.has_choice():
            return None
        started = time.perf_counter()
        rows = self.featurizer.observe()
        public = {key: torch.from_numpy(rows[key]).to(self.device) for key in POLICY_KEYS}
        if self.searcher is not None:
            lines = (lambda picked, k: self.featurizer.search_open(k), self.featurizer.search_advance)
            flat = int(self.searcher.act(public, self._rollouts, lines=lines)[0])
        else:
            flat = int(self.agent.act(public)[0])
        action = self.featurizer.act(flat)
        self.decisions += 1
        spent = time.perf_counter() - started
        self.policy_seconds += spent
        self.slowest_seconds = max(self.slowest_seconds, spent)
        return list(action) if action else None

    def _rollouts(self, picked, candidates):
        return self.featurizer.search(candidates[0].tolist(), self.searcher.horizon)

    def safe_act(self, obs: dict):
        try:
            return self.act(obs)
        except Exception:  # never lose the turn to a bug: pass and say so once
            if not self.failed:
                self.failed = True
                traceback.print_exc()
            return None


async def play(url: str, name: str, player: Player, log: str | None, matches: int | None) -> int:
    """One connection: join, then answer until it closes. Returns matches finished."""
    import websockets

    finished = 0
    async with websockets.connect(url, max_size=None) as socket:
        await socket.send(json.dumps({"t": "join", "name": name, "deck": player.deck}))
        opponent, ticks, plays, ignored = None, 0, 0, 0
        async for raw in socket:
            message = json.loads(raw)
            kind = message.get("t")
            if kind == "obs":
                obs = message["obs"]
                ticks += 1
                ignored += obs.get("last_action") not in ("ok", "none")
                action = player.safe_act(obs)
                plays += action is not None
                await socket.send(json.dumps({"t": "act", "tick": obs["tick"], "action": action}))
            elif kind == "joined":
                print(f"joined as {message['name']}, rating {message['rating']:.0f}", flush=True)
            elif kind == "start":
                opponent, ticks, plays, ignored = message.get("opponent"), 0, 0, 0
                deck = player.start(message.get("deck"))
                rated = "" if message.get("rated", True) else " (unrated)"
                dealt = f", dealt deck {', '.join(deck)}" if message.get("deck") else ""
                print(f"match against {opponent}{rated}{dealt}", flush=True)
            elif kind == "result":
                finished += 1
                mean_ms = 1000 * player.policy_seconds / max(1, player.decisions)
                print(f"  {message['result']} {message['crowns']} rating {message['rating']:.0f} "
                      f"({message['delta']:+.0f}) match {message['match']}: {ticks} ticks, {plays} plays, "
                      f"{ignored} ignored, {mean_ms:.1f} ms per decision, slowest so far "
                      f"{1000 * player.slowest_seconds:.0f} ms"
                      + (f", look-ahead cut short {player.searcher.cut_short} times" if player.searcher else ""),
                      flush=True)
                if log:
                    with open(log, "a") as handle:
                        handle.write(json.dumps({
                            "time": time.time(), "opponent": opponent, "ticks": ticks, "plays": plays,
                            "ignored": ignored, **{key: message.get(key) for key in
                                                   ("result", "crowns", "rating", "delta", "match", "seed", "rated")},
                        }) + "\n")
                if matches is not None and finished >= matches:
                    return finished
            elif kind == "aborted":
                print(f"  match void: {message.get('reason')}", flush=True)
            elif kind == "error":
                raise RuntimeError(f"server: {message.get('msg')}")
    return finished


async def run(args, player: Player) -> None:
    left = args.matches
    delay = 1.0
    while left is None or left > 0:
        try:
            done = await play(args.url, args.name, player, args.log, left)
            if left is not None:
                left -= done
            delay = 1.0
        except (OSError, RuntimeError) as error:
            print(f"connection lost: {error}; joining again in {delay:.0f} s", flush=True)
        except Exception as error:  # websockets' own closing errors
            print(f"connection closed: {type(error).__name__}: {error}; joining again in {delay:.0f} s", flush=True)
        if left is None or left > 0:
            await asyncio.sleep(delay)
            delay = min(delay * 2, 30.0)


def search_settings(text: str | None) -> dict | None:
    """``scope=tile,cells=4,horizon=40,weight=20`` as Searcher's keyword arguments."""
    if not text:
        return None
    kinds = {"scope": str, "cells": int, "horizon": int, "weight": float, "wait_floor": lambda v: v == "1",
             "depth": int, "samples": int, "rollout_temperature": float, "budget": float}
    settings = {}
    for item in text.split(","):
        key, _, value = item.partition("=")
        if key not in kinds:
            raise SystemExit(f"--search: unknown setting '{key}' (known: {', '.join(kinds)})")
        settings[key] = kinds[key](value)
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="ws://host:port or wss://host")
    parser.add_argument("name", help="the name the rating is kept under")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--deck", required=True, help="eight card names, separated by commas")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mode", default="sample", choices=["sample", "greedy"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--ema", action="store_true", help="play the averaged weights")
    parser.add_argument("--search", default=None, metavar="SETTINGS",
                        help="look ahead before moving (cfbot.search), e.g. scope=tile,cells=4,horizon=40,weight=20; "
                             "depth=5 lets both players go on playing in the look-ahead, budget=0.8 stops a "
                             "decision's look-ahead after that many seconds")
    parser.add_argument("--seed", type=int, default=int(time.time()))
    parser.add_argument("--matches", type=int, default=None, help="stop after this many matches")
    parser.add_argument("--log", default=None, help="append one JSON line per match here")
    args = parser.parse_args()
    torch.set_num_threads(max(1, min(4, torch.get_num_threads())))
    deck = [card.strip() for card in args.deck.split(",") if card.strip()]
    player = Player(args.checkpoint, deck, args.device, args.mode, args.temperature, args.seed, args.ema,
                    search_settings(args.search))
    print(f"{args.checkpoint}: deciding every {player.decision_ticks} ticks, deck {', '.join(deck)}", flush=True)
    try:
        asyncio.run(run(args, player))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
