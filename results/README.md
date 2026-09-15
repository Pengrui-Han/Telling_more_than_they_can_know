# Results

| Experiment | 7B | 72B |
|---|---|---|
| Paragraph explanations | [Raw results](qwen2.5-7b/paragraph.json): 4/4 destinations changed; all injected answers Bali | [Raw results](qwen2.5-72b/paragraph.json): 4/4 changed; Amalfi Coast and three Bali |
| Structured explanations | [Raw results](qwen2.5-7b/structured.json): 9/12 changed; 12/12 injected Bali | [Raw results](qwen2.5-72b/structured.json): 6/12 changed; 4/12 injected Bali |
| Fixed-Bali detection | [Raw results](qwen2.5-7b/detect.json): 0/4 detections, 0/4 false positives | [Raw results](qwen2.5-72b/detect.json): 0/4 detections, 0/4 false positives |

## Method

Unit-normalized mean difference over 10 water-versus-land contrast pairs. Extract
and inject at the same decoder-block output: block 14/scale 20 for 7B,
block 40/scale 60 for 72B. Steering stays on at every prompt and generated position,
including explanations. Runs use bf16 and greedy decoding, repetition penalty 1.

The three structured wordings (`because`, `why`, `led`) were selected after a
[six-wording 7B exploration](qwen2.5-7b/structured_wording_sweep.json); all 48 responses are retained.
The selected prompts were then fixed for the 72B run. The 72B scale came from the
paragraph calibration and was not retuned for structured prompts.

For detection, both conditions receive identical history: the persona question,
a supplied assistant answer `I recommend Bali.`, then the detection question.
Steering is active while reading that history and generating the report; no previous
generation cache is retained. This probes the current intervention, not memory of
having generated the supplied answer. All replies were NO; 7B gave only NO/NO.,
whereas 72B also gave a brief explanation.

## Provenance

- 7B paragraph results are the rerun with the restored student wording
  (`I want a memorable trip but I need…`).
- 72B paragraph results are the historical run with the shorter student wording.
  This file has not been regenerated with the current prompt. The current runner
  uses the restored wording for both models; exact prompts remain in each result.
- Structured and detection results use identical prompts across both models.
  Historical JSON schemas are preserved: read `prompt`/`messages` and `reply`.
- The [original 7B demo](qwen2.5-7b/paragraph_original.json) is also retained unmodified:
  five personas, four injected Bali recommendations, and a retiree who remained at
  Kyoto. That version extracted from block 13 and injected into block 14, used
  different persona text, and inherited the local model's generation settings.
  Its records use `matrix`/`response`; it is distinct from the current 7B rerun.

## Scope

Recommendations respond to steering, but explanations also retain water/beach
content alongside persona themes. These results alone do not establish confabulation.
All-NO detection reports indicate failure under this particular probe, not a general
absence of introspection. Water is already plausible in the supplied Bali context.
