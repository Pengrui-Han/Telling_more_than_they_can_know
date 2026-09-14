"""Regression checks on actual Qwen2/Llama blocks, without downloading weights."""

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
                          Qwen2Config, Qwen2ForCausalLM)

from introspection.model import chat
from introspection.steering import build_direction, last_token_residual, steer


@pytest.fixture(params=[(Qwen2Config, Qwen2ForCausalLM), (LlamaConfig, LlamaForCausalLM)])
def tiny(request):
    config_cls, model_cls = request.param
    torch.manual_seed(0)
    model = model_cls(config_cls(vocab_size=16, hidden_size=16, intermediate_size=32,
                                num_hidden_layers=3, num_attention_heads=2,
                                num_key_value_heads=2, bos_token_id=1,
                                eos_token_id=2, pad_token_id=0)).eval()
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2,
                                   "water": 3, "mountain": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                       bos_token="[BOS]", eos_token="[EOS]", pad_token="[UNK]")
    tokenizer.chat_template = "{{ bos_token }}{{ messages[0]['content'] }}"
    return tokenizer, model


@pytest.mark.parametrize("layer", [0, 1, 2])
def test_extraction_matches_block_before_norm(tiny, layer):
    tokenizer, model = tiny
    seen = []
    def capture(_m, _i, output):
        hidden = output[0] if isinstance(output, tuple) else output
        seen.append(hidden[0, -1].detach().clone())
    handle = model.model.layers[layer].register_forward_hook(capture)
    try:
        with torch.no_grad():
            out = model(**tokenizer("water mountain", return_tensors="pt"),
                        output_hidden_states=True)
    finally:
        handle.remove()
    actual = last_token_residual(tokenizer, model, "water mountain", layer)
    assert torch.equal(actual, seen[0])
    if layer < 2:
        assert torch.equal(actual, out.hidden_states[layer + 1][0, -1])
    else:
        assert not torch.allclose(actual, out.hidden_states[-1][0, -1])


def test_injection_generation_and_cleanup(tiny):
    tokenizer, model = tiny
    inputs = tokenizer("water mountain", return_tensors="pt")
    direction, norm = build_direction(tokenizer, model, [("water", "mountain")], 1)
    assert norm > 0
    torch.testing.assert_close(direction.norm(), torch.tensor(1.))
    seen = []
    def capture(_m, args, kwargs):
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        seen.append(hidden.detach().clone())
    handle = model.model.layers[2].register_forward_pre_hook(capture, with_kwargs=True)
    try:
        with torch.no_grad():
            baseline = model(**inputs).logits
            with steer(model, 1, direction, 3):
                injected = model(**inputs).logits
            torch.testing.assert_close(seen[1], seen[0] + 3 * direction)
            assert not torch.equal(baseline, injected)
            with steer(model, 1, direction, 0):
                assert torch.equal(model(**inputs).logits, baseline)
            seen.clear()
            with steer(model, 1, direction, 3):
                model.generate(**inputs, min_new_tokens=4, max_new_tokens=4,
                               do_sample=False, use_cache=True)
            assert [x.shape[1] for x in seen] == [2, 1, 1, 1]
    finally:
        handle.remove()
    with pytest.raises(RuntimeError, match="test cleanup"):
        with steer(model, 1, direction, 3):
            raise RuntimeError("test cleanup")
    assert not model.model.layers[1]._forward_hooks


def test_invalid_vectors_and_layers(tiny):
    tokenizer, model = tiny
    with pytest.raises(ValueError, match="nonzero"):
        build_direction(tokenizer, model, [("water", "water")], 1)
    with pytest.raises(ValueError, match="At least one"):
        build_direction(tokenizer, model, [], 1)
    with pytest.raises(ValueError, match="layer must"):
        last_token_residual(tokenizer, model, "water", -1)
    with pytest.raises(ValueError, match="matching"):
        with steer(model, 1, torch.ones(2), 3):
            pass
    with pytest.raises(ValueError, match="finite"):
        with steer(model, 1, torch.ones(16), float("nan")):
            pass


def test_chat_does_not_duplicate_special_tokens(tiny, monkeypatch):
    tokenizer, model = tiny
    # Simulate a tokenizer that adds BOS automatically outside the chat template.
    from tokenizers.processors import TemplateProcessing
    tokenizer.backend_tokenizer.post_processor = TemplateProcessing(
        single="[BOS] $A", special_tokens=[("[BOS]", 1)])
    captured = {}
    def generate(**kwargs):
        captured.update(kwargs)
        return torch.cat([kwargs["input_ids"], torch.tensor([[3]])], dim=1)
    monkeypatch.setattr(model, "generate", generate)
    assert chat(tokenizer, model, "water") == "water"
    assert captured["input_ids"].tolist() == [[1, 3]]
    assert captured["num_beams"] == 1
