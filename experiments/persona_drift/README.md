# Persona drift along the Assistant Axis, read through the Jacobian lens

Does persona drift become readable in the workspace band (mid layers, via the
Jacobian lens) before it shows up in the model's output tokens?

This experiment links two codebases:

- [`assistant-axis`](https://github.com/safety-research/assistant-axis)
  (fork: `izzimaria/assistant-axis`): axis extraction pipeline, role/default
  prompt data, `ActivationSteering`, organic drift transcripts.
- `jlens` (this repo): `jlens.fit`, `ActivationRecorder`, `JacobianLens.transport`,
  `jlens.vis` slice pages.

## Files

| File | Purpose |
| --- | --- |
| `run.py` | Pipeline: `fit -> axis -> sweep -> readout -> analyze` (each stage resumable, skipped if its output exists) |
| `analysis.py` | Metrics + figures (`summary.json`, `figures/*.png`) |
| `compat.py` | transformers-5 shim for assistant-axis `apply_chat_template` |
| `persona_drift.ipynb` | Reproduces the figures from a run's `--out` directory |
| `probes.json` | Fixed probe battery: identity, open-ended, harmlessness (4 each) |
| `token_sets.json` | Tracked token sets: `persona` and `assistant` words |

## Setup

```bash
pip install -e .            # jlens
git clone https://github.com/izzimaria/assistant-axis ../assistant-axis
pip install -e ../assistant-axis   # vllm optional (fast step-1 generation on GPU)
pip install datasets        # wikitext prompts for fitting
export OPENAI_API_KEY=...   # optional: assistant-axis step-3 judge
```

## Running

```bash
python experiments/persona_drift/run.py \
    --model Qwen/Qwen2.5-7B-Instruct --assistant_axis_dir ../assistant-axis \
    --out out --n_fit_prompts 100 --dim_batch 8
```

Outputs in `--out`:

| Path | Content |
| --- | --- |
| `lens.pt` | Jacobian lens, fitted once on the unsteered model (wikitext, 128 tokens/prompt) |
| `axis.pt` | `unit` (per-layer unit axis), `raw`, `axis_norm`, `resid_norm`, judge mode, roles |
| `axis_pipeline/` | assistant-axis `responses/ activations/ scores/ vectors/ axis_raw.pt` |
| `transcripts.json` | Every generated transcript with `s`, `coeff`, steer layer, category |
| `readout/<id>.json` | Per transcript: response positions, emitted tokens, top-1 lens token per (layer, position), tracked-set ranks (`persona`, `assistant`, `emitted`), axis projection per (layer, position) |
| `readout/<id>_topk.npz` | Top-k lens token ids per (layer, position) |
| `slices/<condition>.html` | `jlens.vis` slice page per condition (probe `identity_0`; organic: whole conversation's last turn) |
| `summary.json`, `figures/` | Metrics (a)-(d), onset layers, lead statistics, organic per-turn curves |

## Design decisions

- **Model.** assistant-axis has explicit configs only for Gemma-2-27B, Qwen3-32B
  and Llama-3.3-70B (with precomputed axes on HF). Neither Qwen2.5-7B-Instruct nor
  Llama-3.1-8B-Instruct is configured, so for either one the axis has to be
  extracted with the pipeline, and `get_config` infers `target_layer = n_layers // 2`.
  Default is **Qwen2.5-7B-Instruct**: ungated, its chat template is handled by
  assistant-axis's Qwen span code, and jlens examples are Qwen.
- **Axis.** Steps 1-5 of the assistant-axis pipeline on its verbatim role
  instructions (`data/roles/instructions`, 275 roles + `default`) and the 240
  extraction questions. `axis = mean(default) - mean(role)` per layer (positive =
  more Assistant-like), then unit-normalised. Step 1 uses `pipeline/1_generate.py`
  (vLLM) on GPU, otherwise an HF fallback with the same prompts and JSONL schema.
  Step 3 needs `OPENAI_API_KEY`; with `--judge none` every role sample counts as
  score 3 (no filtering of responses that failed to role-play), which makes the
  axis noisier than the paper's.
- **Steering.** `assistant_axis.ActivationSteering` (addition, all positions) at
  the assistant-axis target layer, vector = unit axis, coefficient
  `s * ||raw axis[L]||` (`--scale axis_norm`). That is their convention:
  coefficient 1 on the raw axis = one full default-minus-role mean difference.
  Their `steer.ipynb` uses coefficient -10 on Qwen3-32B, so `|s| <= 2` may be too weak
  to change behaviour; widen `--strengths` or use `--scale resid_norm` (s in units
  of mean residual norm).
- **Organic drift.** assistant-axis ships four multi-turn drift transcripts
  (`transcripts/persona_drift/{coding,writing,therapy,philosophy}.json`) but no
  driver. The organic condition replays their first `--organic_turns` user turns
  and lets the model write its own replies (s = 0).
- **Readout.** Each transcript is re-encoded with the chat template and forwarded
  under the same steering. Residuals at all layers are recorded; fitted layers are
  transported (`lens.transport`) and unembedded, the final layer is the model's
  own output. Response positions come from assistant-axis
  `ConversationEncoder.response_indices`. The residual at position *p* is scored against
  the token emitted at *p+1*.
- **Workspace band.** Default `[n/3, 2n/3]` (layers 9-18 for 28-layer Qwen2.5-7B),
  override with `--band`. A token set is "readable" at a cell if its best member
  has lens rank < `--hit_rank` (10).

## Metrics

- **(a)** `proj_steer_layer`: mean over response positions of `resid[L] · unit[L]` vs `s`.
- **(b)** band-min rank (best rank over band layers) of the persona / assistant sets per response
  position; heatmap over `s` × relative position, and P(rank < k) vs absolute position.
- **(c)** slice pages per condition.
- **(d)** per layer, Δprojection(s) − Δprojection(0) vs Δ P(persona rank < k); `onset_layers` reports
  the first layer reaching half the maximum of each.
- **Lead.** Per transcript: first response position where the persona set enters the band
  (`first_band_hit_pos`) vs first position whose emitted token is a persona word
  (`first_emit_pos`); `frac_band_before_emit` and `median_lead_tokens` per `s`.

## Results

**Status: only a CPU smoke run so far. No 7B results yet.** The smoke run used
`Qwen/Qwen2.5-0.5B-Instruct` (24 layers, band 8–16, steering layer 12), a lens fitted on 2 wikitext
prompts of 48 tokens, an unjudged axis from 4 roles (pirate, mystic, ghost, poet) × 2 questions,
`s ∈ {-8, -2, 0, +2}`, 12 probes per strength, 40-token responses, and 2 organic therapy turns.
It shows that every stage runs end to end. The numbers below show the plumbing works; they are not
findings.

| s | proj @ L12 | persona band hit rate | persona band median rank | persona emitted rate | assistant band median rank |
|---|---|---|---|---|---|
| −8 | −17.3 | 0.69 | 4 | 0.035 | 1866 |
| −2 | −3.7 | 0.26 | 27 | 0.018 | 110 |
| 0 | 0.9 | 0.15 | 77 | 0.004 | 72 |
| +2 | 5.0 | 0.14 | 91 | 0.012 | 108 |

- (a) The projection is linear in `s`, so steering and readout use the same axis.
- (d) The axis projection jumps at the injection layer (12), as expected. Persona lens-readability
  rises only after it: it starts at layer 13 and peaks near layer 19, with a +0.66 hit-rate gain at s = −8.
- At s = −8, persona words are lens-readable in the band about 20× more often than they are emitted
  (0.69 vs 0.035), and the assistant set falls out of the band.
- **The lead question is not answered yet.** `frac_band_before_emit` is already 0.92 at s = 0,
  because a 2-prompt lens makes many band hits on its own. So "band before emission" does not
  separate steered from unsteered runs at this scale. For the real run, compare against the s = 0
  baseline, not the absolute fraction.

Next: the full run on one GPU (Qwen2.5-7B-Instruct, `--n_fit_prompts 100`+, default strengths, all
roles, `OPENAI_API_KEY` set so the judge filters role samples).

**transformers 5 note.** assistant-axis expects `apply_chat_template(tokenize=True)` to return
`list[int]` (transformers 4). `compat.py` restores that default, both in-process and for the pipeline
scripts, so the fork stays unmodified.
