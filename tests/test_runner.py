"""CLI orchestration checks; model math is covered by the steering tests."""
import json
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from introspection import runner
from introspection.model import DEFAULT_MODEL


@pytest.fixture
def backend(monkeypatch):
    calls = {"loads": [], "builds": [], "replies": []}
    def load(name):
        calls["loads"].append(name)
        return object(), SimpleNamespace(config=SimpleNamespace(num_hidden_layers=28))
    def build(tokenizer, model, pairs, layer):
        calls["builds"].append(layer)
        return torch.tensor([1., 0.]), 2.
    @contextmanager
    def steering(*args):
        yield
    def reply(tokenizer, model, messages, max_new_tokens):
        calls["replies"].append(messages)
        return "raw reply"
    monkeypatch.setattr(runner, "load_model", load)
    monkeypatch.setattr(runner, "build_direction", build)
    monkeypatch.setattr(runner, "steer", steering)
    monkeypatch.setattr(runner, "chat", reply)
    return calls


def test_all_experiments_share_one_load_and_preserve_histories(tmp_path, backend):
    runner.main(["--output", str(tmp_path)])
    assert backend["loads"] == [DEFAULT_MODEL]
    assert backend["builds"] == [14]
    for experiment, count in [("paragraph", 8), ("structured", 24), ("detect", 8)]:
        report = json.loads((tmp_path / f"{experiment}.json").read_text())
        assert len(report["runs"]) == count
        assert all(r["reply"] == "raw reply" for r in report["runs"])
        for a, b in zip(report["runs"][::2], report["runs"][1::2]):
            assert a["messages"] == b["messages"]
            assert (a["scale"], b["scale"]) == (0, 20)
            if experiment == "detect":
                assert a["messages"][1] == {"role": "assistant", "content": "I recommend Bali."}
            elif experiment == "paragraph":
                assert "explain in 3-4 sentences" in a["messages"][0]["content"]
    with pytest.raises(SystemExit):
        runner.main(["--output", str(tmp_path)])
    assert len(backend["loads"]) == 1  # reject overwrites before expensive loading


def test_saved_72b_vector_selects_72b_scale_without_rebuilding(tmp_path, backend):
    vector = tmp_path / "vector.pt"
    torch.save({"model": "Qwen/Qwen2.5-72B-Instruct", "layer": 40,
                "direction": torch.tensor([1., 0.])}, vector)
    runner.main(["--experiment", "detect", "--vector", str(vector), "--output", str(tmp_path)])
    report = json.loads((tmp_path / "detect.json").read_text())
    assert report["layer"] == 40 and report["scales"] == [0, 60]
    assert backend["builds"] == []
    with pytest.raises(SystemExit):
        runner.main(["--vector", str(vector), "--model", DEFAULT_MODEL,
                     "--output", str(tmp_path / "wrong-model")])
    assert len(backend["loads"]) == 1


def test_new_model_requires_explicit_scales(tmp_path, backend):
    with pytest.raises(SystemExit):
        runner.main(["--model", "example/new-model", "--output", str(tmp_path)])
    assert backend["loads"] == []
    runner.main(["--model", "example/new-model", "--scales", "0", "5",
                 "--experiment", "paragraph", "--output", str(tmp_path)])
    report = json.loads((tmp_path / "paragraph.json").read_text())
    assert report["scales"] == [0, 5]
