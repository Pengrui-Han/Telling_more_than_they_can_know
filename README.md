# Telling more than they can know

Steer a model’s vacation recommendation, then examine its explanation and thought-detection report.

## Run

```bash
pip install -e '.[test]'
python run.py --output runs/7b
python run.py --experiment detect --vector runs/7b/water_direction.pt --output runs/7b
```

To rerun the 72B experiments with its saved vector and previous calibration:

```bash
python run.py --vector results/qwen2.5-72b/water_direction.pt --scales 0 60 --output runs/72b
python run.py --experiment detect --vector results/qwen2.5-72b/water_direction.pt --scales 0 60 --output runs/72b
```

The first command builds a water-themed vector and runs **3 wordings × 4 personas ×
no injection / injection**. The second supplies the same `I recommend Bali.` history
in both conditions and asks whether the model notices an injected concept.

Defaults: Qwen2.5-7B-Instruct, middle decoder block (index 14), scales 0 and 20,
bf16, greedy decoding. One model load per command.

- `--wordings because why led`: choose the explanation questions.
- `--scales 0 10 20 40`: inspect other strengths; selection is manual.
- `--model <HF_ID> --layer <index>`: build a vector for another model.
- `--vector results/qwen2.5-7b/water_direction.pt`: reuse the tested 7B vector.

Requires enough memory and a supported Hugging Face Qwen2/Llama-style decoder.
Multi-node inference and 400B Qwen compatibility have not been implemented or tested.

## Read

- [Exact prompts](introspection/prompts.py)
- [Experiment spec and preliminary results](SPEC.md)
- [Raw 7B results](results/qwen2.5-7b/) · [Raw 72B results](results/qwen2.5-72b/)

The three wordings were selected from a six-wording pilot in which all injected
replies were Bali. This is exploratory selection, not a guarantee for other models.
The complete pilot is retained. Explanations still contain beach/water content.
The 72B rerun changed 6/12 destinations at scale 60; only 4/12 injected replies were
Bali. Both models answered NO in all fixed-Bali detection trials.

```bash
python -m pytest tests/  # includes a downloaded/cached Qwen2.5-0.5B CPU test
```
