"""Checks that the steering hook does exactly what steering.py's docstring says.

    TEST_MODEL=Qwen/Qwen2.5-0.5B-Instruct python -m pytest tests/

Any small Qwen2-style model works; runs on CPU in float32 in well under a minute.
"""

import os
from contextlib import contextmanager

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from introspection.steering import build_direction, last_token_residual, steer

MODEL = os.environ.get("TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
LAYER = 6
TEXT = "I love the ocean and sandy beaches."


@pytest.fixture(scope="module")
def tm():
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32).eval()
    return tokenizer, model


def forward(model, inputs):
    with torch.no_grad():
        return model(**inputs, output_hidden_states=True, use_cache=False)


def unit(model, seed=0):
    d = torch.randn(model.config.hidden_size, generator=torch.Generator().manual_seed(seed))
    return d / d.norm()


def block_output(o):
    return o[0] if isinstance(o, tuple) else o


@contextmanager
def capture_output(module, store):
    """Record a block's output tensor (clone) in store["out"]. Hook returns None."""
    def hook(_m, _i, o):
        store["out"] = block_output(o).clone()
    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def capture_input(module, store):
    """Record the hidden_states a block receives, via a pre-hook. Hook returns None."""
    def hook(_m, args, kwargs):
        x = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        store["in"] = x.clone()
    handle = module.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        yield
    finally:
        handle.remove()


def n_hooks(model):
    return len(model.model.layers[LAYER]._forward_hooks)


def test_read_and_write_locations_match(tm):
    """hidden_states[L+1] (what build_direction reads) is the output of layers[L] (where steer writes)."""
    tokenizer, model = tm
    inputs = tokenizer(TEXT, return_tensors="pt")
    seen = {}
    with capture_output(model.model.layers[LAYER], seen):
        out = forward(model, inputs)
    assert torch.equal(seen["out"], out.hidden_states[LAYER + 1])
    assert torch.equal(last_token_residual(tokenizer, model, TEXT, LAYER),
                       out.hidden_states[LAYER + 1][0, -1])


def test_hook_adds_exactly_scale_times_direction(tm):
    """Block LAYER+1 receives base + scale*direction at every position, up to float32 rounding
    (Qwen residual streams contain activations in the thousands, so the tolerance scales)."""
    tokenizer, model = tm
    inputs = tokenizer("Where should I go on vacation?", return_tensors="pt")
    d, scale = unit(model), 3.0
    nxt = model.model.layers[LAYER + 1]
    before = n_hooks(model)

    base_in, steered_in = {}, {}
    with capture_input(nxt, base_in):
        base = forward(model, inputs)
    with steer(model, LAYER, d, scale):
        assert n_hooks(model) == before + 1
        with capture_input(nxt, steered_in):
            steered = forward(model, inputs)
    assert n_hooks(model) == before, "our hook was not removed"

    err = (steered_in["in"] - base_in["in"] - scale * d).abs()
    tol = 1e-4 + 1e-6 * base_in["in"].abs()
    print(f"\nmax |delta - scale*d| = {err.max():.2e}  (max |base| = {base_in['in'].abs().max():.0f})")
    assert (err <= tol).all()
    assert torch.equal(steered.hidden_states[LAYER], base.hidden_states[LAYER])  # below: untouched
    assert not torch.allclose(steered.logits, base.logits)                         # downstream: changed


def test_scale_zero_is_exact_baseline(tm):
    tokenizer, model = tm
    inputs = tokenizer(TEXT, return_tensors="pt")
    before = n_hooks(model)
    base = forward(model, inputs)
    with steer(model, LAYER, unit(model), 0):
        assert n_hooks(model) == before
        again = forward(model, inputs)
    assert torch.equal(again.logits, base.logits)


def test_hook_fires_on_prompt_and_every_generated_token(tm):
    """With KV cache, generate() runs one forward per new token; the hook must see each one."""
    tokenizer, model = tm
    text = tokenizer.apply_chat_template([{"role": "user", "content": "Say something."}],
                                         tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    seen_lengths = []

    def counter(_m, _i, o):
        seen_lengths.append(block_output(o).shape[1])

    handle = model.model.layers[LAYER].register_forward_hook(counter)
    try:
        n_new = 5
        with steer(model, LAYER, unit(model), 3.0), torch.no_grad():
            model.generate(**inputs, max_new_tokens=n_new, min_new_tokens=n_new,
                           do_sample=False, repetition_penalty=1.0)
    finally:
        handle.remove()
    assert seen_lengths == [inputs["input_ids"].shape[1]] + [1] * (n_new - 1)


def test_build_direction_is_unit_mean_difference(tm):
    tokenizer, model = tm
    pairs = [(TEXT, "I love mountains and dense forests."),
             ("Swimming in clear water.", "Hiking on rocky trails.")]
    d, raw_norm = build_direction(tokenizer, model, pairs, LAYER)
    expected = torch.stack([last_token_residual(tokenizer, model, p, LAYER)
                            - last_token_residual(tokenizer, model, n, LAYER) for p, n in pairs]).mean(0)
    assert torch.allclose(d * raw_norm, expected, atol=1e-4)
    assert abs(d.norm().item() - 1) < 1e-5
