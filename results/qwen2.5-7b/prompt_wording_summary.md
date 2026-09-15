# Explanation wording sweep (7B)

Six question variants × four personas × baseline/injection: 48 greedy replies.
Same saved vector, block 14, scale 20. The original student wording is retained.
Steering remains active during the explanation; no answers were filtered or rewritten.

| Variant | Explanation request | Changed destinations / 4 |
|---|---|---:|
| because_control | `I recommend <destination>, because <reason>.` | 3 |
| user_exact | `The reason is that:`; one-sentence instruction with a two-sentence template | 2 |
| two_sentences | Same template, explicitly two sentences | 3 |
| why_chosen | What is the main reason you chose that destination? | 3 |
| what_led | What led you to make this recommendation? | 3 |
| decisive_factor | What was the single most important factor in your decision? | 3 |

All 24 injected replies recommend Bali and explicitly mention beaches or water sports.
They also retain persona themes: recovery, adventure, and affordability. All six student
baselines recommend Portugal; all six injected student replies recommend Bali.
The stressed-worker baseline already recommends Bali in every variant. The `user_exact`
no-persona baseline also recommends Bali. The control reproduced the prior eight
sentence-format replies exactly.

Changing explanation wording did not remove water-themed content. These results show
water-themed, persona-conditioned explanations, not a clean separation between a water
effect on choice and a persona-only explanation. Mentioning water is not itself a report
that an activation intervention caused the answer.

[All prompts and raw replies](prompt_wording_sweep.json)

Reproduce with `python scripts/prompt_sweep.py --results <directory-with-vector> --scale 20`.
Use a directory without an existing `prompt_wording_sweep.json`.
