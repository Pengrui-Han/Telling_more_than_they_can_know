"""Sweep the injection scale on the no-persona prompt.

Goal: the smallest scale that flips the recommendation to a water destination while the
text stays coherent. Larger scales degrade into lists and then repetition loops.

    python scripts/calibrate.py [--results results/qwen2.5-7b] [--scales 0 20 40 60 80 100]
"""

import argparse
import json
from pathlib import Path

import torch

from introspection.model import chat, load_model
from introspection.prompts import build_prompt
from introspection.steering import steer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/qwen2.5-7b", help="directory with water_direction.pt")
    p.add_argument("--scales", type=float, nargs="+", default=[0, 20, 40, 60, 80, 100])
    args = p.parse_args()

    results = Path(args.results)
    vec = torch.load(results / "water_direction.pt", map_location="cpu", weights_only=True)
    tokenizer, model = load_model(vec["model"])
    prompt = build_prompt("no_persona")

    rows = []
    for scale in args.scales:
        with steer(model, vec["layer"], vec["direction"], scale):
            reply = chat(tokenizer, model, prompt)
        rows.append({"scale": scale, "reply": reply})
        print(f"\n=== scale {scale:g} ===\n{reply}")

    out = results / "calibration.json"
    with open(out, "w") as f:
        json.dump({"model": vec["model"], "layer": vec["layer"], "prompt": prompt, "runs": rows},
                  f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
