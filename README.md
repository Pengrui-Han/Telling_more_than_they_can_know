# Telling more than they can know

A minimal activation-steering experiment: does an LLM explain a changed recommendation
using the user’s context? A water-themed direction steers vacation recommendations;
four persona conditions test how the accompanying explanations change.

## Method

Average the last-token residual differences from 10 water-versus-land contrast pairs
and normalize the vector. Add `scale × direction` after a zero-indexed decoder block
at every prompt and generated-token position, including during the explanation.
Compare baseline and steered answers with greedy decoding (bf16, repetition penalty 1).

[Prompts](introspection/prompts.py) · [Steering](introspection/steering.py) · [Saved outputs](results/)

## Run

Python 3.10+ and sufficient GPU memory are required for the full experiments.

```bash
pip install -e '.[test]'
python scripts/build_vector.py --results runs/7b
python scripts/calibrate.py --results runs/7b
python scripts/persona_matrix.py --results runs/7b --scale 20
```

For single-sentence answers (`I recommend <destination>, because <reason>.`),
add `--format sentence` to calibration and matrix commands. Results are saved as
`calibration_sentence.json` and `persona_matrix_sentence.json`. This requests the
format through prompting; it does not constrain decoding or rewrite answers.

Calibration writes a scale sweep to `calibration.json`. Inspect it and choose the
smallest tested scale that changes the recommendation coherently; selection is manual.
The final script writes all prompts and replies to `persona_matrix.json`.

For another model, set `--model`, `--layer`, and a new `--results` directory in
`build_vector.py`. Pass that directory to the other scripts; they load the model and
layer from the saved vector. Rebuild and recalibrate for each model.

| Model | Layer (zero-based) | Scale used in saved runs |
|---|---:|---:|
| Qwen/Qwen2.5-7B-Instruct | 14 | 20 |
| Qwen/Qwen2.5-72B-Instruct | 40 | 60 |

Loading uses Hugging Face `device_map="auto"` across GPUs visible to one process.
Qwen2/Llama-style decoder layouts are supported. Each script reloads the model.
405B has not been tested; bf16 weights alone require about 810 GB. Multi-node
inference and vLLM are not implemented.

For SLURM, edit the resource directives in [run.sbatch](run.sbatch), activate your
Python environment, then run `mkdir -p logs && sbatch run.sbatch`.
Set `MODEL`, `LAYER`, `SCALE`, and `RESULTS` as needed.

## Results and scope

The saved 7B/72B runs show recommendations changing toward beach destinations,
with explanations tailored to each persona and no spontaneous mention of injection.
These are exploratory examples, not proof of a specific confabulation mechanism.
This repository contains the recommendation matrix only; the original demo’s
injection-detection, causal follow-up, and framing experiments are not included.

The 7B matrix was rerun with the original student wording ("I want a memorable trip
but I need..."): all four injected recommendations are Bali. The previous wording's
outputs are retained in `persona_matrix_previous_prompt.json`.

The 7B sentence-format experiment at the same scale (20) followed the requested
format in all 8 replies. Injected destinations were Bali in all four conditions;
3/4 destinations changed because the stressed-worker baseline already chose Bali.
Changing the answer format also changes the experimental prompt.

A [six-wording comparison](results/qwen2.5-7b/prompt_wording_summary.md) retained
beach/water content in all 24 injected explanations, alongside persona-specific themes.

The 72B outputs are historical and use the shorter student wording; they have not
been rerun with the restored prompt or sentence format. Exact text can vary across
hardware and library versions.

## Tests

```bash
python -m pytest tests/test_offline.py  # tiny Qwen2/Llama models; no downloads
python -m pytest tests/                # also loads Qwen2.5-0.5B-Instruct on CPU
```

Checks cover extraction/injection alignment, generation hooks, cleanup, invalid
vectors, and chat special tokens. Reviewed with torch 2.10 / transformers 5.3.
