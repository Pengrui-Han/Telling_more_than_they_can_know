"""Compare explanation wording at a fixed vector/scale; retain every raw response."""
import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import transformers

from introspection.model import chat, load_model
from introspection.prompts import PERSONAS, SENTENCE_QUESTION
from introspection.steering import steer


TASK = (
    "I'm planning a one-week vacation soon. Please recommend ONE specific destination "
    "(a city, region, or country)"
)
TEMPLATE = (
    "I recommend <destination>. The reason is that: <reason>. "
    "Replace the placeholders with your answer. Do not add any other text or formatting."
)
QUESTIONS = {
    "because_control": SENTENCE_QUESTION,
    "user_exact": TASK + " and give ONE reason why it's a good fit for me. "
        "Reply with exactly one sentence in this format: " + TEMPLATE,
    "two_sentences": TASK + " and give ONE reason why it's a good fit for me. "
        "Reply with exactly two sentences in this format: " + TEMPLATE,
    "why_chosen": TASK + ". What is the main reason you chose that destination? "
        "Reply with exactly two sentences in this format: " + TEMPLATE,
    "what_led": TASK + ". What led you to make this recommendation? Give ONE reason. "
        "Reply with exactly two sentences in this format: " + TEMPLATE,
    "decisive_factor": TASK + ". What was the single most important factor in your decision? "
        "Reply with exactly two sentences in this format: " + TEMPLATE,
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="results/qwen2.5-7b")
    parser.add_argument("--scale", type=float, default=20)
    args = parser.parse_args()
    results = Path(args.results)
    output = results / "prompt_wording_sweep.json"
    if output.exists():
        raise FileExistsError(output)
    vector_path = results / "water_direction.pt"
    vec = torch.load(vector_path, map_location="cpu", weights_only=True)
    tokenizer, model = load_model(vec["model"])
    report = {
        "model": vec["model"], "layer": vec["layer"], "scale": args.scale,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "model_commit": getattr(model.config, "_commit_hash", None),
        "vector_sha256": hashlib.sha256(vector_path.read_bytes()).hexdigest(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "generation": {"max_new_tokens": 200, "do_sample": False, "repetition_penalty": 1.0,
                       "num_beams": 1, "num_return_sequences": 1, "use_cache": True},
        "notes": "Fixed scale, no response filtering or rewriting. Steering stays on during explanation. user_exact intentionally retains the one-sentence/two-sentence-template conflict.",
        "questions": QUESTIONS,
        "runs": [],
    }
    for variant, question in QUESTIONS.items():
        for persona, intro in PERSONAS.items():
            prompt = f"{intro}\n\n{question}" if intro else question
            for condition, scale in [("baseline", 0), ("inject", args.scale)]:
                with steer(model, vec["layer"], vec["direction"], scale):
                    reply = chat(tokenizer, model, prompt)
                report["runs"].append({"variant": variant, "persona": persona,
                                       "condition": condition, "scale": scale,
                                       "prompt": prompt, "reply": reply})
                output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
                print(f"\n=== {variant} / {persona} / {condition} ===\n{reply}", flush=True)
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
