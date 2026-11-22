# Training

How to start, resume and monitor a training run, how strength is measured, and
what sets the throughput. What is being trained is described in
[design.md](design.md).

## Running

```sh
python -m cfbot.train --run runs/main --set budget_hours=330   # start a two-week run
python -m cfbot.train --run runs/main --resume                 # continue after a stop
scripts/run_forever.sh runs/main --set budget_hours=330        # restart after crashes
```

Ctrl-C checkpoints and exits. A checkpoint holds the model, its averaged
weights, the optimizer, the league and the deck statistics, and a run directory
can be moved between machines: a run started on Apple Silicon continues on an
NVIDIA GPU with `--resume`. The model size is fixed when a run starts
(`--preset small|base|large`); every other setting can change on resume with
`--set key=value`. [config.py](../python/cfbot/config.py) lists them all. Values
are read as JSON, so switches are `true`, `false` and `null`; `True`, `False`
and `None` are understood too.

One difference between devices comes from PyTorch, not from this code. On Apple
GPUs PyTorch 2.9 also sends gradient to the padding row of an embedding table;
on NVIDIA GPUs and the CPU that row stays at zero. Some of those rows are read
by real tokens: row 0 of the unit table by an ability button that holds no
unit, row 0 of the tower table by every unit that is not a tower, row 0 of the
card table by an empty hand slot and by a card with no second form. So on a Mac
those rows are learned and elsewhere they are not. A checkpoint moved from a Mac to an NVIDIA
GPU keeps the values it has and stops updating them; a run started on the
NVIDIA GPU never has them. Nothing breaks either way, but the two are not
exactly the same model, and a short run on a Mac is not an exact preview of
the long one.

Every schedule (learning rate, entropy, reward shaping, deck mix, scripted-bot
share) follows the run's progress from 0 to 1. With `budget_hours`, progress is
training time spent over the budget, and the run stops when the budget is
spent. Without it, progress is `iteration / total_iters`.

Each line of `runs/<name>/metrics.jsonl` is one iteration. The numbers to
watch:

| Field | Meaning |
|---|---|
| `gauntlet_elo`, `g_*` | Rating and scores from the gauntlet (below). The strength measurement. |
| `lineage_elo` | Rating against frozen copies of the agent's own past. It shows the agent still beats its history; it cannot tell progress from a cycle. |
| `win_random`, `win_spend8.0`, `win_spend5.0` | Score against the scripted bots in training games. All three place random cards, so these saturate early. |
| `explained_variance` | How well the critic predicts outcomes; rises toward 1. |
| `entropy`, `kl`, `kl_stop` | Policy randomness and step size on decisions that had a choice. `kl_stop` is 1 when the policy stopped updating early in an iteration. |
| `play_rate`, `choice_frac` | Share of the learner's real decisions that were a play, and share of its rows that had a choice at all. |
| `trade`, `leak` | Elixir value of damage dealt per decision, and elixir wasted at the cap. |
| `return_clamped` | Share of value targets outside the value range. Should stay near 0. |
| `rejected`, `crashes`, `overflow`, `hazard_overflow`, `deck_retries` | Should stay 0. |
| `alarms` | Present when a check failed: non-finite gradients, a policy that stopped playing, collapsed entropy, halved throughput, simulator crashes. Also printed as `ALARM`. |

## Measuring strength

Every 2% of the run the agent plays a gauntlet against scripted bots and
frozen snapshots of itself taken at doubling iterations, always on the same 32
deck pairs and seeds, each pair played from both seats. The arena has additional
[fixed opponents](arena-simulator.md#opponents). Results go to
`gauntlet.jsonl`, and one Bradley-Terry rating scale is fitted over all of them
with the random bot at 0.

```sh
python -m cfbot.evaluate --ratings runs/main                       # rating of every gauntlet so far
python -m cfbot.evaluate runs/main/checkpoint.pt random --pairs 200
python -m cfbot.evaluate A.pt B.pt --ema                           # averaged weights
```

A direct comparison plays each deck pair twice with seats swapped and the same
seed, counts exactly one game per battle slot, and reports simulator crashes
separately instead of scoring them.

To probe how exploitable a checkpoint is, train a fresh agent against it alone
and watch `win_fixed`:

```sh
python -m cfbot.train --run runs/probe --set fixed_opponent=runs/main/checkpoint.pt
```

## Throughput and memory

On an M2 Max (32 GB) the `base` preset (3.8M parameters) learns about 2,900
decisions per second, counting collection and the update: about 8 s to collect
256 battles x 128 decisions and 10 s to train on them. That was measured in
the first iterations of a run and again sixty iterations in; busier boards
later in training are slower.

The network sets the pace, not the simulator. Stepping all 256 battles takes
about 5 ms of a 60 ms collection step. The rest is four small actor passes, one
for the learner and one for each league opponent, and on an Apple GPU a pass
over 21 rows costs almost half as much as one over 250. The update is
nine-tenths forward and backward passes. The code is arranged around that:

- **Collection.** The actor runs only on learner rows that have a choice; a row
  that can only wait is settled without it. The critic does not run during
  collection: it values the stored rollout afterwards in a few large passes.
  Scripted bots, rewards and bookkeeping stay on the CPU. Per step the
  observation goes to the device as two blocks and every sampled action comes
  back in one transfer.
- **Update.** Cross-attention projects the seven queries instead of the whole
  memory. The placement head runs only on rows whose stored action was a play.
  On Apple GPUs embedding tables take their gradient through a matrix product,
  because the lookup's own gradient is very slow there.

Each of these gives the same numbers as the plain way, to rounding (to
half-precision rounding where mixed precision is on).
[tests/test_equivalence.py](../tests/test_equivalence.py) holds the collector, the
update and the network to that against frozen copies of the originals, on the
CPU and on an Apple GPU where there is one.

`scripts/profile_iteration.py` prints where one iteration's time goes;
`scripts/bench_env.py` times the simulator alone.

Two things to know on Apple GPUs. Kernels are compiled per tensor shape, so the
code pads to a few fixed sizes. And `minibatch` must fit in memory: going over
physical memory makes training hundreds of times slower instead of failing. The
base model keeps about 4 MB of activations per decision at full entity width,
so 2,048 is the limit on a 16 GB GPU. Half precision is not used there: it
measured slower than float32, several times slower when acting.

On an NVIDIA GPU mixed precision is used automatically, AdamW runs as fused
kernels, the finished rollout stays on the GPU (`rollout_on_device`, about
1 GB) and peak memory is logged as `gpu_mem_gb`. `--set compile=true` also
sends the two training passes through `torch.compile`. What was measured on an
RTX 5070 Ti, where `compile=true` did not help, is in
[arena-simulator.md](arena-simulator.md#measured-training-runs).
