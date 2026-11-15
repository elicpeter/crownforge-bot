"""Export a trained policy to ONNX for use outside Python.

    python -m cfbot.export runs/main/checkpoint.pt policy.onnx

The exported graph is the deployment interface of :class:`cfbot.agent.Agent`:

    inputs   the public observation planes, in ``cfbot.env.POLICY_KEYS`` order
    outputs  ``log_probs`` [rows, 4611]  log-probability of every simulator
                                         action, -inf where illegal
             ``value``     [rows]        public value estimate: the expected
                                         return, which includes the shaped
                                         reward while shaping is on, over a
                                         support of about [-1.5, 1.5]

The policy has no hidden memory, so each call is independent. A caller picks an
action from ``log_probs`` and passes its index to the simulator's
``grid::decode_observed``. Needs ``pip install onnx onnxscript onnxruntime``.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn

from .config import DRAFT_CARDS
from .env import POLICY_KEYS, Env
from .league import load_policy


class Deploy(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *planes):
        return self.model.full_log_probs(dict(zip(POLICY_KEYS, planes)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--rows", type=int, default=2, help="rows in the example batch")
    parser.add_argument("--ema", action="store_true", help="export the averaged weights")
    args = parser.parse_args()

    env = Env(max(1, args.rows // 2), workers=1, exclude_cards=DRAFT_CARDS)
    model, _ = load_policy(args.checkpoint, env.spec, torch.device("cpu"), ema=args.ema)
    deploy = Deploy(model).eval()

    # Play a little so the example has units on the board.
    obs = env.reset(seed=1)
    for _ in range(80):
        obs, _ = env.step(model.act(obs, with_value=False)["flat"])
    example = tuple(obs[key] for key in POLICY_KEYS)

    names = list(POLICY_KEYS)
    dynamic = {name: {0: "rows"} for name in names}
    for name in ("entities", "entity_extras", "entity_mask"):
        dynamic[name][1] = "entities"
    dynamic.update({"log_probs": {0: "rows"}, "value": {0: "rows"}})
    torch.onnx.export(
        deploy,
        example,
        args.output,
        input_names=names,
        output_names=["log_probs", "value"],
        dynamic_axes=dynamic,
        opset_version=18,
        dynamo=False,
    )
    print(f"wrote {args.output}")

    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed; skipped verification")
        return
    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])
    log_probs, value = session.run(None, {name: plane.numpy() for name, plane in zip(names, example)})
    with torch.no_grad():
        expected, expected_value = deploy(*example)
    legal = torch.isfinite(expected).numpy()
    worst = np.abs(log_probs[legal] - expected.numpy()[legal]).max()
    same_mask = bool((np.isfinite(log_probs) == legal).all())
    same_choice = bool((log_probs.argmax(-1) == expected.argmax(-1).numpy()).all())
    print(
        f"onnxruntime check: max log-prob difference {worst:.2e}, value difference "
        f"{np.abs(value - expected_value.numpy()).max():.2e}, legal mask equal {same_mask}, "
        f"greedy action equal {same_choice}"
    )


if __name__ == "__main__":
    main()
