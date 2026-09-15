"""Run paragraph, structured, and thought-detection experiments with one model load."""
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


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment", choices=["all", "paragraph", "structured", "detect"], default="all")
    p.add_argument("--model", help="Hugging Face ID; defaults to the vector's model or Qwen2.5-7B-Instruct")
    p.add_argument("--layer", type=int, help="zero-based block; defaults to the vector's layer or middle block")
    p.add_argument("--vector", type=Path, help="reuse an existing water_direction.pt")
    p.add_argument("--scales", type=float, nargs="+", help="defaults: 0/20 for 7B, 0/60 for 72B; required for a new model")
    p.add_argument("--wordings", choices=list(QUESTIONS), nargs="+", help="structured experiment only; default: all three")
    p.add_argument("--output", type=Path, help="default: runs/<model-name>")
    p.add_argument("--max-new-tokens", type=int, default=200)
    args = p.parse_args(argv)
    if args.max_new_tokens < 1:
        p.error("max-new-tokens must be positive")
    vec = torch.load(args.vector, map_location="cpu", weights_only=True) if args.vector else None
    if vec and ((args.model and args.model != vec["model"]) or
                (args.layer is not None and args.layer != vec["layer"])):
        p.error("model/layer must match the saved vector; omit --vector to rebuild")
    name = args.model or (vec["model"] if vec else DEFAULT_MODEL)
    defaults = {DEFAULT_MODEL: [0, 20], "Qwen/Qwen2.5-72B-Instruct": [0, 60]}
    scales = args.scales if args.scales is not None else defaults.get(name)
    if scales is None:
        p.error("For a new model, specify --scales (e.g. 0 20 40); inspect the outputs to calibrate")
    if not all(math.isfinite(s) for s in scales):
        p.error("scales must be finite")
    args.output = args.output or Path("runs") / name.rstrip("/").split("/")[-1].lower()
    plans = {"paragraph": ["paragraph"], "structured": args.wordings or list(QUESTIONS),
             "detect": ["because"]}
    experiments = list(plans) if args.experiment == "all" else [args.experiment]
    for experiment in experiments:
        output = args.output / f"{experiment}.json"
        if output.exists():
            p.error(f"{output} already exists; choose a new --output directory")
    if not args.vector and (args.output / "water_direction.pt").exists():
        p.error("Output vector already exists; use --vector to reuse it or choose a new --output")
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
    metadata = {
        "model": name, "layer": layer, "scales": scales,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "model_commit": getattr(model.config, "_commit_hash", None),
        "vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "generation": {"max_new_tokens": args.max_new_tokens, "do_sample": False, "repetition_penalty": 1.0,
                       "num_beams": 1, "num_return_sequences": 1, "use_cache": True},
        "injection": "All prompt and generated positions at the selected block output.",
    }
    for experiment in experiments:
        output = args.output / f"{experiment}.json"
        report = {**metadata, "experiment": experiment, "wordings": plans[experiment],
                  "supplied_answer": "I recommend Bali." if experiment == "detect" else None,
                  "runs": []}
        for wording in plans[experiment]:
            for persona in PERSONAS:
                messages = trial_messages(persona, wording, experiment)
                for scale in scales:
                    with steer(model, layer, vec["direction"], scale):
                        reply = chat(tokenizer, model, messages, max_new_tokens=args.max_new_tokens)
                    report["runs"].append({"wording": wording, "persona": persona, "scale": scale,
                                           "condition": "baseline" if scale == 0 else "inject",
                                           "messages": messages, "reply": reply})
                    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                    print(f"\n=== {experiment} / {wording} / {persona} / scale={scale:g} ===\n{reply}", flush=True)
        print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
