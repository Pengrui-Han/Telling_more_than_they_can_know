"""Build the "water" direction from contrast pairs and save it.

    python scripts/build_vector.py [--model ...] [--layer 14] [--results results/qwen2.5-7b]
"""

import argparse
from pathlib import Path

import torch

from introspection.model import DEFAULT_MODEL, load_model
from introspection.prompts import CONTRAST_PAIRS
from introspection.steering import build_direction


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--layer", type=int, default=14, help="decoder block whose output is steered")
    p.add_argument("--results", default="results/qwen2.5-7b", help="output directory")
    args = p.parse_args()

    tokenizer, model = load_model(args.model)
    direction, raw_norm = build_direction(tokenizer, model, CONTRAST_PAIRS, args.layer)

    out = Path(args.results)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model": args.model, "layer": args.layer, "direction": direction,
         "raw_norm": raw_norm, "pairs": CONTRAST_PAIRS},
        out / "water_direction.pt",
    )
    print(f"saved {out / 'water_direction.pt'}  layer={args.layer}  dim={direction.numel()}  raw_norm={raw_norm:.2f}")


if __name__ == "__main__":
    main()
