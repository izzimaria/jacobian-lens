"""Regression tests for experiments/persona_drift resume/caching and metrics."""

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parents[1] / "experiments" / "persona_drift"))
pytest.importorskip("matplotlib")
import analysis  # noqa: E402
import run  # noqa: E402

import jlens  # noqa: E402


def _rec(first_hit, first_emit, n=6):
    ranks = np.full((3, n), 100)
    if first_hit is not None:
        ranks[1, first_hit] = 0
    emitted = [False] * n
    if first_emit is not None:
        emitted[first_emit] = True
    return {
        "layers": [0, 1, 2],
        "ranks": {"persona": ranks},
        "emitted_in_set": {"persona": emitted},
    }


def test_lead_stats_ignores_missing_emissions():
    meta = {"band": [1, 1]}
    group = [_rec(1, 3), _rec(4, 2), _rec(0, None), _rec(None, None)]
    m = analysis.lead_stats(meta, group, hit_rank=10)
    assert m["persona_frac_band_before_emit"] == 0.5
    assert m["persona_frac_band_hit_no_emit"] == 0.5
    m = analysis.lead_stats(meta, [_rec(0, None)], hit_rank=10)
    assert m["persona_frac_band_before_emit"] is None


def test_check_stage_rejects_changed_inputs(tmp_path):
    args = SimpleNamespace(out=tmp_path)
    run.check_stage(args, "sweep", {"strengths": [-1, 1]}, exists=False)
    run.check_stage(args, "sweep", {"strengths": [-1, 1]}, exists=True)
    with pytest.raises(SystemExit, match="strengths"):
        run.check_stage(args, "sweep", {"strengths": [-4, 4]}, exists=True)
    # no outputs yet: new inputs are recorded instead of rejected
    run.check_stage(args, "sweep", {"strengths": [-4, 4]}, exists=False)
    run.check_stage(args, "sweep", {"strengths": [-4, 4]}, exists=True)


def test_scratch_is_namespaced_by_out(tmp_path):
    a = SimpleNamespace(out=tmp_path / "a", scratch_dir=tmp_path / "s")
    b = SimpleNamespace(out=tmp_path / "b", scratch_dir=tmp_path / "s")
    assert run.scratch(a) != run.scratch(b)
    assert run.scratch(a).parent == (tmp_path / "s").resolve()
    assert run.scratch(SimpleNamespace(out=tmp_path, scratch_dir=None)) is None


def test_atomic_write_keeps_old_file_on_failure(tmp_path):
    p = tmp_path / "x.json"
    run.save_json({"a": 1}, p)

    def boom(t):
        t.write_text("partial")
        raise RuntimeError

    with pytest.raises(RuntimeError):
        run.atomic_write(p, boom)
    assert p.read_text() == '{\n "a": 1\n}'
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_lens_save_is_atomic(tmp_path):
    lens = jlens.JacobianLens(jacobians={0: torch.eye(4)}, n_prompts=1, d_model=4)
    p = tmp_path / "lens.pt"
    lens.save(str(p))
    assert [f.name for f in tmp_path.iterdir()] == ["lens.pt"]
    assert jlens.JacobianLens.load(str(p)).n_prompts == 1


def test_stage_fit_scratch_resume_guard(tmp_path, monkeypatch):
    import jlens.examples
    from tests.tiny import TinyDecoder

    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5, "uvwxyzabcd " * 5]
    monkeypatch.setattr(jlens.examples, "load_wikitext_prompts", lambda n: prompts[:n])
    models = SimpleNamespace(
        get=lambda: (None, None, TinyDecoder(n_layers=3, d_model=8))
    )

    def args(out, n):
        return SimpleNamespace(
            out=tmp_path / out,
            scratch_dir=tmp_path / "scratch",
            model="tiny",
            n_fit_prompts=n,
            fit_seq_len=64,
            dim_batch=4,
            fit_sync_every=0,
        )

    a = args("a", 3)
    a.out.mkdir()
    run.stage_fit(a, models)
    assert jlens.JacobianLens.load(str(a.out / "lens.pt")).n_prompts == 3
    assert not (a.out / "lens_ckpt.pt").exists()

    # a new --out in the same runtime must not inherit run a's checkpoint
    b = args("b", 2)
    b.out.mkdir()
    run.stage_fit(b, models)
    assert jlens.JacobianLens.load(str(b.out / "lens.pt")).n_prompts == 2

    # changing fit inputs for an existing lens is refused
    with pytest.raises(SystemExit, match="n_fit_prompts"):
        run.stage_fit(args("a", 2), models)

    # a corrupt lens is refit instead of reused
    (b.out / "lens.pt").write_bytes(b"partial")
    run.stage_fit(b, models)
    assert jlens.JacobianLens.load(str(b.out / "lens.pt")).n_prompts == 2
