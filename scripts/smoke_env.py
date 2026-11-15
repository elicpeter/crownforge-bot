"""Random-play smoke test and throughput check for the environment bridge."""
import argparse
import time

import numpy as np

from cfbot import _env

parser = argparse.ArgumentParser()
parser.add_argument("--envs", type=int, default=64)
parser.add_argument("--workers", type=int, default=8)
parser.add_argument("--episodes", type=int, default=200)
parser.add_argument("--play-prob", type=float, default=0.15)
args = parser.parse_args()

env = _env.VecEnv(num_envs=args.envs, workers=args.workers, seed=1)
spec = env.spec()
print({k: v for k, v in spec.items() if not isinstance(v, (list, np.ndarray))})
names = spec["card_names"]
print("supported cards:", sum(spec["card_supported"]))
obs = env.reset()
for key, value in obs.items():
    print(f"  {key:14s} {str(value.dtype):8s} {value.shape}")
print("deck row0:", [names[int(i)] for i in obs["own_deck"][0][:, 0]])
print("hand row0:", [names[int(i)] for i in obs["hand"][0][:, 0]], "legal", int(obs["legal"][0].sum()))

rng = np.random.default_rng(0)
rows = env.num_envs * 2


def random_actions(legal):
    actions = np.zeros(rows, dtype=np.int64)
    for row in np.nonzero(rng.random(rows) < args.play_prob)[0]:
        actions[row] = rng.choice(np.nonzero(legal[row])[0])
    return actions


start = time.time()
steps = episodes = max_entities = 0
ticks = []
outcomes = np.zeros(4, int)
verdicts = np.zeros(4, int)
lengths = []
while episodes < args.episodes:
    obs = env.step(random_actions(obs["legal"]))
    steps += 1
    max_entities = max(max_entities, int(obs["entity_mask"].sum(1).max()))
    ticks.append(float(obs["ticks"].mean()))
    verdicts += np.bincount(obs["verdict"], minlength=4)
    finished = obs["outcome"] >= 0
    episodes += int(finished.sum())
    outcomes += np.bincount(obs["outcome"][finished], minlength=4)
    lengths += list(obs["episode_decisions"][finished])
elapsed = time.time() - start
print(
    f"{steps} steps, {steps * rows / elapsed:.0f} agent-steps/s, {episodes} episodes, "
    f"blue/red/draw/crash {outcomes}, mean decisions {np.mean(lengths):.0f}, "
    f"max entities {max_entities}, mean ticks per step {np.mean(ticks):.1f}, "
    f"wait/ok/rejected/invalid {verdicts}, entity overflow {obs['entity_overflow'].max()}, "
    f"deck retries {env.deck_retries()}"
)
for blue, red in env.crashes()[:10]:
    print("crash:", blue, "vs", red)
