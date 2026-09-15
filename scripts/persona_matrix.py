"""Personas x {baseline, injected}.

Same hidden cause (the water direction) under four different visible contexts (personas).
This measures recommendation shifts and persona-conditioned explanations; it does not
by itself establish whether the model can detect or report the intervention.

    python scripts/persona_matrix.py [--results results/qwen2.5-7b] [--scale 20]
"""

import argparse
import json
from pathlib import Path

import torch

from introspection.model import chat, load_model
from introspection.prompts import PERSONAS, build_prompt
from introspection.steering import steer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="results/qwen2.5-7b", help="directory with water_direction.pt")
    p.add_argument("--scale", type=float, default=20, help="injection scale (from calibrate.py)")
    p.add_argument("--format", choices=["paragraph", "sentence"], default="paragraph",
                   help="response format requested in the prompt")
    args = p.parse_args()

    results = Path(args.results)
    vec = torch.load(results / "water_direction.pt", map_location="cpu", weights_only=True)
    tokenizer, model = load_model(vec["model"])

    rows = []
    for persona in PERSONAS:
        prompt = build_prompt(persona, args.format)
        for condition, scale in [("baseline", 0), ("inject", args.scale)]:
            with steer(model, vec["layer"], vec["direction"], scale):
                reply = chat(tokenizer, model, prompt)
            rows.append({"persona": persona, "condition": condition, "scale": scale,
                         "prompt": prompt, "reply": reply})
            print(f"\n=== {persona} / {condition} (scale {scale:g}) ===\n{reply}")

    out = results / ("persona_matrix.json" if args.format == "paragraph" else "persona_matrix_sentence.json")
    with open(out, "w") as f:
        json.dump({"model": vec["model"], "layer": vec["layer"], "scale": args.scale,
                   "response_format": args.format, "runs": rows},
                  f, indent=2)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
