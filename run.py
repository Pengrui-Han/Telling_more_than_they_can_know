"""Run recommendation or fixed-Bali thought-detection trials with one model load."""
import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from introspection.model import DEFAULT_MODEL, chat, load_model
from introspection.prompts import CONTRAST_PAIRS, DETECT_QUESTION, PERSONAS, QUESTIONS, build_prompt
from introspection.steering import build_direction, steer


def trial_messages(persona, wording, experiment):
    messages = [{"role": "user", "content": build_prompt(persona, wording)}]
    if experiment == "detect":
        # Supplied history, not a claim that this answer was generated in an earlier trial.
        messages += [{"role": "assistant", "content": "I recommend Bali."},
                     {"role": "user", "content": DETECT_QUESTION}]
    return messages


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=["recommend", "detect"], default="recommend")
    p.add_argument("--model", help="Hugging Face ID; defaults to the vector's model or Qwen2.5-7B-Instruct")
    p.add_argument("--layer", type=int, help="zero-based block; defaults to the vector's layer or middle block")
    p.add_argument("--vector", type=Path, help="reuse an existing water_direction.pt")
    p.add_argument("--scales", type=float, nargs="+", default=[0, 20])
    p.add_argument("--wordings", choices=list(QUESTIONS), nargs="+", help="default: all three for recommend; because for detect")
    p.add_argument("--output", type=Path, default=Path("runs/latest"))
    args = p.parse_args()
    if not all(math.isfinite(s) for s in args.scales):
        p.error("scales must be finite")
    output = args.output / f"{args.experiment}.json"
    if output.exists():
        p.error(f"{output} already exists; choose a new --output directory")
    vec = torch.load(args.vector, map_location="cpu", weights_only=True) if args.vector else None
    if vec and ((args.model and args.model != vec["model"]) or
                (args.layer is not None and args.layer != vec["layer"])):
        p.error("model/layer must match the saved vector; omit --vector to rebuild")
    name = args.model or (vec["model"] if vec else DEFAULT_MODEL)
    tokenizer, model = load_model(name)
    layer = vec["layer"] if vec else (args.layer if args.layer is not None else model.config.num_hidden_layers // 2)
    args.output.mkdir(parents=True, exist_ok=True)
    vector_path = args.vector
    if vec is None:
        vector_path = args.output / "water_direction.pt"
        if vector_path.exists():
            p.error(f"{vector_path} exists; pass --vector to reuse it or choose a new --output")
        direction, norm = build_direction(tokenizer, model, CONTRAST_PAIRS, layer)
        vec = {"model": name, "layer": layer, "direction": direction, "raw_norm": norm,
               "pairs": CONTRAST_PAIRS}
        torch.save(vec, vector_path)
    wordings = args.wordings or (list(QUESTIONS) if args.experiment == "recommend" else ["because"])
    report = {
        "experiment": args.experiment, "model": name, "layer": layer,
        "scales": args.scales, "wordings": wordings,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "model_commit": getattr(model.config, "_commit_hash", None),
        "vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "generation": {"max_new_tokens": 200, "do_sample": False, "repetition_penalty": 1.0,
                       "num_beams": 1, "num_return_sequences": 1, "use_cache": True},
        "injection": "All prompt and generated positions at the selected block output.",
        "supplied_answer": "I recommend Bali." if args.experiment == "detect" else None,
        "runs": [],
    }
    for wording in wordings:
        for persona in PERSONAS:
            messages = trial_messages(persona, wording, args.experiment)
            for scale in args.scales:
                with steer(model, layer, vec["direction"], scale):
                    reply = chat(tokenizer, model, messages)
                report["runs"].append({"wording": wording, "persona": persona, "scale": scale,
                                       "condition": "baseline" if scale == 0 else "inject",
                                       "messages": messages, "reply": reply})
                output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                print(f"\n=== {wording} / {persona} / scale={scale:g} ===\n{reply}", flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
