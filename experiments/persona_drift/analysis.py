"""Metrics and figures for the persona-drift experiment (reads ``run.py`` outputs).

``run(out_dir)`` writes ``summary.json`` and ``figures/*.png``; the notebook calls
the individual ``fig_*`` functions.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SETS = ("persona", "assistant")


def load(out_dir) -> tuple[dict, list[dict]]:
    out_dir = Path(out_dir)
    meta = json.loads((out_dir / "readout_meta.json").read_text())
    rs = [
        json.loads(p.read_text()) for p in sorted((out_dir / "readout").glob("*.json"))
    ]
    for r in rs:
        for k in r["ranks"]:
            r["ranks"][k] = np.asarray(r["ranks"][k])
        r["proj"] = np.asarray(r["proj"])
    return meta, rs


def band_rows(r, meta) -> list[int]:
    lo, hi = meta["band"]
    return [i for i, l in enumerate(r["layers"]) if lo <= l <= hi]


def band_min(r, meta, name) -> np.ndarray:
    """Best (lowest) lens rank of a token set over the workspace band, per response position."""
    return r["ranks"][name][band_rows(r, meta)].min(0)


def steered(rs):
    return [r for r in rs if r["category"] != "organic"]


def by_s(rs):
    groups = defaultdict(list)
    for r in steered(rs):
        groups[r["s"]].append(r)
    return dict(sorted(groups.items()))


def first_true(mask) -> int | None:
    idx = np.flatnonzero(mask)
    return int(idx[0]) if idx.size else None


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def per_s_metrics(meta, rs, hit_rank: int) -> dict:
    L = meta["steer_layer"]
    out = {}
    for s, group in by_s(rs).items():
        m = {
            "n": len(group),
            "proj_steer_layer": float(np.mean([r["proj"][L].mean() for r in group])),
            "proj_steer_layer_sem": float(
                np.std([r["proj"][L].mean() for r in group]) / np.sqrt(len(group))
            ),
        }
        for name in SETS:
            band = np.concatenate([band_min(r, meta, name) for r in group])
            final = np.concatenate([r["ranks"][name][-1] for r in group])
            m[f"{name}_band_hit_rate"] = float((band < hit_rank).mean())
            m[f"{name}_band_median_rank"] = float(np.median(band))
            m[f"{name}_output_hit_rate"] = float((final < hit_rank).mean())
            m[f"{name}_emitted_rate"] = float(
                np.mean(np.concatenate([r["emitted_in_set"][name] for r in group]))
            )
        m.update(lead_stats(meta, group, hit_rank))
        out[s] = m
    return out


def lead_stats(meta, group, hit_rank: int, name: str = "persona") -> dict:
    """Does the persona set reach the band (rank < hit_rank) before a persona word is emitted?"""
    t_ws, t_out, t_final = [], [], []
    for r in group:
        t_ws.append(first_true(band_min(r, meta, name) < hit_rank))
        t_out.append(first_true(np.asarray(r["emitted_in_set"][name])))
        t_final.append(first_true(r["ranks"][name][-1] < hit_rank))
    both = [
        (w, o)
        for w, o in zip(t_ws, t_out, strict=True)
        if w is not None and o is not None
    ]
    return {
        f"{name}_frac_band_hit": float(np.mean([t is not None for t in t_ws])),
        f"{name}_frac_emitted": float(np.mean([t is not None for t in t_out])),
        f"{name}_frac_output_hit": float(np.mean([t is not None for t in t_final])),
        f"{name}_frac_band_before_emit": float(
            np.mean(
                [
                    w is not None and (o is None or w < o)
                    for w, o in zip(t_ws, t_out, strict=True)
                ]
            )
        ),
        f"{name}_median_lead_tokens": float(np.median([o - w for w, o in both]))
        if both
        else None,
        f"{name}_first_band_hit_pos": t_ws,
        f"{name}_first_emit_pos": t_out,
    }


def layer_profiles(meta, rs, hit_rank: int) -> dict:
    """Per-layer axis projection and persona/assistant readability (hit rate) for each s."""
    prof = {}
    for s, group in by_s(rs).items():
        prof[s] = {
            "proj": np.mean([r["proj"].mean(1) for r in group], 0),
            **{
                f"{n}_hit": np.mean(
                    [(r["ranks"][n] < hit_rank).mean(1) for r in group], 0
                )
                for n in SETS
            },
        }
    return prof


def onset_layers(
    meta, rs, hit_rank: int, frac: float = 0.5, min_delta: float = 0.02
) -> dict:
    """First layer where |delta proj| (vs s=0) reaches ``frac`` of its max, and where the persona-set
    hit rate gain reaches ``frac`` of its max — 'axis changes' vs 'persona verbally readable'."""
    prof = layer_profiles(meta, rs, hit_rank)
    if 0.0 not in prof:
        return {}
    layers = steered(rs)[0]["layers"]
    out = {}
    for s, p in prof.items():
        if s == 0:
            continue
        dproj = np.abs(p["proj"] - prof[0.0]["proj"])
        dhit = p["persona_hit"] - prof[0.0]["persona_hit"]
        out[s] = {
            "axis_onset_layer": first_true(dproj >= frac * dproj.max())
            if dproj.max() > 0
            else None,
            "persona_readable_onset_layer": (
                layers[first_true(dhit >= frac * dhit.max())]
                if dhit.max() > min_delta
                else None
            ),
            "max_persona_hit_gain": float(dhit.max()),
        }
    return out


def organic_metrics(meta, rs, hit_rank: int) -> dict:
    L = meta["steer_layer"]
    out = {}
    for r in rs:
        if r["category"] != "organic":
            continue
        turn = np.asarray(r["turn"])
        rows = []
        for t in np.unique(turn):
            sel = turn == t
            rows.append(
                {
                    "turn": int(t),
                    "proj_steer_layer": float(r["proj"][L][sel].mean()),
                    **{
                        f"{n}_band_hit_rate": float(
                            (band_min(r, meta, n)[sel] < hit_rank).mean()
                        )
                        for n in SETS
                    },
                    "persona_emitted_rate": float(
                        np.asarray(r["emitted_in_set"]["persona"])[sel].mean()
                    ),
                }
            )
        out[r["probe_id"]] = rows
    return out


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #


def fig_projection(meta, rs, ax=None):
    """(a) drift strength vs projection of the mean response residual on the axis (steer layer)."""
    ax = ax or plt.subplots(figsize=(5, 3.5))[1]
    L = meta["steer_layer"]
    cats = sorted({r["category"] for r in steered(rs)})
    for c in cats:
        groups = by_s([r for r in rs if r["category"] == c])
        xs = list(groups)
        ys = [np.mean([r["proj"][L].mean() for r in g]) for g in groups.values()]
        ax.plot(xs, ys, "o-", label=c)
    ax.set(
        xlabel="steering strength s",
        ylabel=f"mean response projection (layer {L})",
        title="(a) axis projection vs drift strength",
    )
    ax.legend(fontsize=8)
    return ax


def fig_band_heatmaps(meta, rs, hit_rank: int, n_bins: int = 10, axes=None):
    """(b) band-min lens rank of persona / assistant sets vs s x relative response position."""
    groups = by_s(rs)
    if axes is None:
        _, axes = plt.subplots(1, len(SETS), figsize=(5 * len(SETS), 3.5))
    for ax, name in zip(np.atleast_1d(axes), SETS, strict=False):
        grid = np.full((len(groups), n_bins), np.nan)
        for i, g in enumerate(groups.values()):
            cols = defaultdict(list)
            for r in g:
                bm = band_min(r, meta, name)
                bins = np.minimum(
                    (np.arange(len(bm)) * n_bins) // max(len(bm), 1), n_bins - 1
                )
                for b, v in zip(bins, bm, strict=True):
                    cols[b].append(np.log10(1 + v))
            for b, v in cols.items():
                grid[i, b] = np.median(v)
        im = ax.imshow(grid, aspect="auto", cmap="viridis_r", origin="lower")
        ax.set_yticks(range(len(groups)), [f"{s:+g}" for s in groups])
        ax.set_xticks(
            range(n_bins), [f"{b / n_bins:.1f}" for b in range(n_bins)], fontsize=7
        )
        ax.set(
            xlabel="relative response position",
            ylabel="s",
            title=f"(b) {name} set: median log10(1+band-min rank)",
        )
        plt.colorbar(im, ax=ax)
    return axes


def fig_band_hit_vs_position(
    meta, rs, hit_rank: int, name="persona", max_pos: int = 128, ax=None
):
    """(b') fraction of transcripts whose band-min rank < hit_rank at each absolute response position."""
    ax = ax or plt.subplots(figsize=(6, 3.5))[1]
    cmap = plt.get_cmap("coolwarm")
    groups = by_s(rs)
    ss = list(groups)
    for s, g in groups.items():
        hits = np.full((len(g), max_pos), np.nan)
        for i, r in enumerate(g):
            bm = band_min(r, meta, name)[:max_pos] < hit_rank
            hits[i, : len(bm)] = bm
        ax.plot(
            np.nanmean(hits, 0),
            color=cmap((s - min(ss)) / (max(ss) - min(ss) + 1e-9)),
            label=f"s={s:+g}",
        )
    ax.set(
        xlabel="response token position",
        ylabel=f"P(band rank < {hit_rank})",
        title=f"(b') {name} set surfacing in the band",
    )
    ax.legend(fontsize=7, ncol=2)
    return ax


def fig_layer_profiles(meta, rs, hit_rank: int, axes=None):
    """(d) per-layer delta projection vs delta persona readability (relative to s=0)."""
    prof = layer_profiles(meta, rs, hit_rank)
    layers = steered(rs)[0]["layers"]
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    cmap = plt.get_cmap("coolwarm")
    ss = list(prof)
    for s, p in prof.items():
        if s == 0:
            continue
        c = cmap((s - min(ss)) / (max(ss) - min(ss) + 1e-9))
        axes[0].plot(p["proj"] - prof[0.0]["proj"], color=c, label=f"s={s:+g}")
        axes[1].plot(
            layers,
            p["persona_hit"] - prof[0.0]["persona_hit"],
            color=c,
            label=f"s={s:+g}",
        )
    lo, hi = meta["band"]
    for ax in axes:
        ax.axvspan(lo, hi, color="grey", alpha=0.15)
        ax.axvline(meta["steer_layer"], color="k", ls=":", lw=1)
        ax.set_xlabel("layer")
    axes[0].set(
        ylabel="Δ mean projection on axis",
        title="(d) where the axis projection changes",
    )
    axes[1].set(
        ylabel=f"Δ P(persona rank < {hit_rank})",
        title="(d) where persona becomes lens-readable",
    )
    axes[1].legend(fontsize=7, ncol=2)
    return axes


def fig_organic(meta, rs, hit_rank: int, axes=None):
    org = organic_metrics(meta, rs, hit_rank)
    if not org:
        return None
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for name, rows in org.items():
        t = [x["turn"] for x in rows]
        axes[0].plot(t, [x["proj_steer_layer"] for x in rows], "o-", label=name)
        axes[1].plot(t, [x["persona_band_hit_rate"] for x in rows], "o-", label=name)
    axes[0].set(
        xlabel="assistant turn",
        ylabel="mean projection (steer layer)",
        title="organic drift: axis projection",
    )
    axes[1].set(
        xlabel="assistant turn",
        ylabel=f"P(persona band rank < {hit_rank})",
        title="organic drift: persona in band",
    )
    axes[0].legend(fontsize=7)
    return axes


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def run(out_dir, hit_rank: int | None = None) -> dict:
    out_dir = Path(out_dir)
    meta, rs = load(out_dir)
    hit_rank = hit_rank or meta["hit_rank"]
    figs = out_dir / "figures"
    figs.mkdir(exist_ok=True)

    for fname, fn in [
        ("a_projection", lambda: fig_projection(meta, rs)),
        ("b_band_heatmaps", lambda: fig_band_heatmaps(meta, rs, hit_rank)),
        ("b_persona_vs_position", lambda: fig_band_hit_vs_position(meta, rs, hit_rank)),
        (
            "b_assistant_vs_position",
            lambda: fig_band_hit_vs_position(meta, rs, hit_rank, "assistant"),
        ),
        ("d_layer_profiles", lambda: fig_layer_profiles(meta, rs, hit_rank)),
        ("organic", lambda: fig_organic(meta, rs, hit_rank)),
    ]:
        if fn() is not None:
            plt.tight_layout()
            plt.savefig(figs / f"{fname}.png", dpi=130)
        plt.close("all")

    per_s = per_s_metrics(meta, rs, hit_rank)
    headline = {
        f"{s:+g}": {k: v for k, v in m.items() if not isinstance(v, list)}
        for s, m in per_s.items()
    }
    summary = {
        "model": meta["model"],
        "band": meta["band"],
        "steer_layer": meta["steer_layer"],
        "hit_rank": hit_rank,
        "headline": headline,
        "per_s": per_s,
        "onset_layers": onset_layers(meta, rs, hit_rank),
        "organic": organic_metrics(meta, rs, hit_rank),
    }
    summary = _jsonable(summary)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary
