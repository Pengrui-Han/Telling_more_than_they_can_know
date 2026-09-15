# Telling more than they can know

Activation steering of vacation recommendations and explanations, followed by thought detection.

| Experiment | Task | Output |
|---|---|---|
| `paragraph` | Recommend a destination and explain in 3–4 sentences | `paragraph.json` |
| `structured` | Three short recommendation-and-reason formats | `structured.json` |
| `detect` | Given the same Bali answer, report whether an injected concept is noticed | `detect.json` |

Each experiment compares unsteered and steered responses across four personas.

## Run

```bash
pip install -e '.[test]'
introspection --output runs/7b
```

This loads Qwen2.5-7B once, builds the direction, and runs all three experiments.
`python run.py` is equivalent. Defaults: bf16, greedy decoding, middle block,
scales 0/20 for Qwen2.5-7B and 0/60 for Qwen2.5-72B.

```bash
# Reuse the saved 72B vector; run all three experiments.
introspection --vector results/qwen2.5-72b/water_direction.pt --output runs/72b

# New model: build its own vector and inspect several strengths.
introspection --model <HF_MODEL_ID> --scales 0 20 40 --output runs/new-model
```

Use `--experiment paragraph|structured|detect` for one experiment, `--layer N` to
choose a zero-based block, or `--max-new-tokens N` for a longer generation budget.
Scale selection is manual; changing model or prompt can change the effect.

Requires sufficient memory and a Hugging Face Qwen2/Llama-style `model.model.layers`
decoder. Loading uses GPUs visible to one process. Qwen2.5-7B/72B are tested;
new architectures may need an adapter. Multi-node inference is not implemented.

## Structure

```text
run.py                 source-checkout entry point
introspection/         prompts, model loading, steering, runner
results/               all three experiments for 7B and 72B; pilot records
tests/                 hook, chat, and runner checks
```

[Exact prompts](introspection/prompts.py) · [Results and provenance](results/README.md)

```bash
python -m pytest tests/  # includes a cached/downloaded 0.5B model test
```
