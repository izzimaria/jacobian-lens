"""Persona drift along the Assistant Axis, read out through the Jacobian lens.

Stages (run all, or a subset with ``--stages``):

  fit      Fit the Jacobian lens once on the unsteered model (generic text),
           checkpoint to ``{out}/lens.pt``.
  axis     Run the assistant-axis pipeline (generate -> activations -> judge ->
           vectors -> axis) on its verbatim role/default prompt data, then
           unit-normalize the per-layer axis -> ``{out}/axis.pt``.
  sweep    Additive steering along the axis at each strength ``s`` on a fixed
           probe battery, plus the organic multi-turn drift condition (replaying
           the user turns of ``assistant-axis/transcripts/persona_drift``)
           -> ``{out}/transcripts.json``.
  readout  Forward each transcript (under the same steering) with
           ``ActivationRecorder``, transport + unembed the response-span residual
           at every fitted layer, record top-k lens tokens, tracked-set ranks and
           axis projections -> ``{out}/readout/*.json|npz``, ``{out}/slices/*.html``.
  analyze  Metrics + figures -> ``{out}/summary.json``, ``{out}/figures/*.png``.

Example (one GPU, Qwen2.5-7B-Instruct)::

    python experiments/persona_drift/run.py --model Qwen/Qwen2.5-7B-Instruct \
        --assistant_axis_dir ../assistant-axis --out out --n_fit_prompts 100
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import jlens
from jlens.hooks import ActivationRecorder
from jlens.vis import _ranks_of, build_page, compute_slice

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compat import patch_apply_chat_template  # noqa: E402

patch_apply_chat_template()

HERE = Path(__file__).resolve().parent
log = logging.getLogger("persona_drift")

STAGES = ["fit", "axis", "sweep", "readout", "analyze"]


# --------------------------------------------------------------------------- #
# Setup helpers
# --------------------------------------------------------------------------- #


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument(
        "--assistant_axis_dir", default=str(HERE.parents[2] / "assistant-axis")
    )
    p.add_argument("--out", default="out")
    p.add_argument("--stages", nargs="+", default=["all"], choices=["all", *STAGES])
    p.add_argument("--device", default=None, help="cuda / cpu (default: auto)")
    # fit
    p.add_argument("--n_fit_prompts", type=int, default=100)
    p.add_argument("--fit_seq_len", type=int, default=128)
    p.add_argument("--dim_batch", type=int, default=8)
    p.add_argument(
        "--scratch_dir",
        type=Path,
        default=None,
        help="Local dir for bulky intermediates (fit checkpoint, axis activations) "
        "when --out is a synced folder such as Colab Drive (default: --out)",
    )
    p.add_argument(
        "--fit_sync_every",
        type=int,
        default=0,
        help="With --scratch_dir: also copy the fit checkpoint to --out every N "
        "prompts (0 = never)",
    )
    # axis
    p.add_argument(
        "--axis_roles",
        nargs="+",
        default=None,
        help="Role subset (default: all roles in data/roles/role_list.json). 'default' is always added.",
    )
    p.add_argument("--axis_question_count", type=int, default=240)
    p.add_argument(
        "--axis_prompt_indices",
        type=int,
        nargs="+",
        default=None,
        help="Instruction variants per role (default: all 5)",
    )
    p.add_argument("--axis_max_tokens", type=int, default=512)
    p.add_argument(
        "--axis_batch_size", type=int, default=16, help="HF generation batch"
    )
    p.add_argument(
        "--act_batch_size",
        type=int,
        default=4,
        help="2_activations.py batch; it stacks all layers on GPU, so keep small",
    )
    p.add_argument(
        "--min_count",
        type=int,
        default=50,
        help="4_vectors.py: min score=3 samples per role",
    )
    p.add_argument(
        "--judge",
        choices=["auto", "openai", "none"],
        default="auto",
        help="'none' skips 3_judge.py and keeps every role sample (score=3); auto = openai iff OPENAI_API_KEY set",
    )
    p.add_argument("--generator", choices=["auto", "vllm", "hf"], default="auto")
    # sweep
    p.add_argument(
        "--strengths", type=float, nargs="+", default=[-2, -1, -0.5, 0, 0.5, 1, 2]
    )
    p.add_argument(
        "--steer_layer",
        type=int,
        default=None,
        help="default: assistant-axis target_layer",
    )
    p.add_argument(
        "--scale",
        choices=["axis_norm", "resid_norm"],
        default="axis_norm",
        help="coefficient = s * scale[layer] applied to the unit axis. axis_norm = assistant-axis convention "
        "(coefficient multiplies the raw mean-difference axis); resid_norm = mean residual norm.",
    )
    p.add_argument("--probes", default=str(HERE / "probes.json"))
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument(
        "--organic",
        nargs="*",
        default=None,
        help="Organic transcripts to replay (default: all in transcripts/persona_drift)",
    )
    p.add_argument(
        "--organic_turns",
        type=int,
        default=8,
        help="user turns replayed per organic transcript",
    )
    p.add_argument("--organic_max_new_tokens", type=int, default=192)
    # readout
    p.add_argument("--token_sets", default=str(HERE / "token_sets.json"))
    p.add_argument("--top_k", type=int, default=10)
    p.add_argument(
        "--band",
        type=int,
        nargs=2,
        default=None,
        help="workspace band [lo, hi] inclusive (default: n/3..2n/3)",
    )
    p.add_argument(
        "--hit_rank",
        type=int,
        default=10,
        help="rank threshold (0-based <) counted as 'readable'",
    )
    p.add_argument(
        "--slice_probe",
        default="identity_0",
        help="probe id rendered as a slice page per condition",
    )
    p.add_argument("--max_seq_len", type=int, default=4096)
    args = p.parse_args(argv)
    if "all" in args.stages:
        args.stages = STAGES
    args.out = Path(args.out)
    args.assistant_axis_dir = Path(args.assistant_axis_dir).resolve()
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    return args


def chat_kwargs(model_name: str) -> dict:
    return {"enable_thinking": False} if "qwen" in model_name.lower() else {}


class Models:
    """Lazily loaded HF model + jlens wrapper (freed before pipeline subprocesses)."""

    def __init__(self, args):
        self.args = args
        self.hf = self.tok = self.lens_model = None

    def get(self):
        if self.hf is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            dtype = torch.bfloat16 if self.args.device == "cuda" else torch.float32
            log.info("loading %s (%s, %s)", self.args.model, self.args.device, dtype)
            self.tok = AutoTokenizer.from_pretrained(self.args.model)
            self.hf = AutoModelForCausalLM.from_pretrained(
                self.args.model, dtype=dtype
            ).to(self.args.device)
            self.lens_model = jlens.from_hf(self.hf, self.tok)
        return self.hf, self.tok, self.lens_model

    def free(self):
        self.hf = self.tok = self.lens_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False))


def chat_ids(tok, conversation, model_name) -> list[int]:
    out = tok.apply_chat_template(
        conversation,
        tokenize=True,
        add_generation_prompt=False,
        **chat_kwargs(model_name),
    )
    if hasattr(out, "keys"):
        out = out["input_ids"]
    return list(out)


# --------------------------------------------------------------------------- #
# Stage 1: fit
# --------------------------------------------------------------------------- #


def stage_fit(args, models: Models):
    path = args.out / "lens.pt"
    if path.exists():
        log.info("lens exists at %s, skipping fit", path)
        return
    from jlens.examples import load_wikitext_prompts

    _, _, lens_model = models.get()
    prompts = load_wikitext_prompts(args.n_fit_prompts)
    log.info(
        "fitting lens on %d wikitext prompts (seq_len=%d, dim_batch=%d)",
        len(prompts),
        args.fit_seq_len,
        args.dim_batch,
    )
    # On a synced folder (Colab Drive) every rewrite of the ~1.3 GB checkpoint
    # leaves a trashed copy, so checkpoint to --scratch_dir and mirror to
    # --out only every --fit_sync_every prompts.
    synced = args.out / "lens_ckpt.pt"
    ckpt = args.scratch_dir / "lens_ckpt.pt" if args.scratch_dir else synced
    step = len(prompts)
    done = 0
    if ckpt != synced:
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        if synced.exists() and not ckpt.exists():
            log.info("copying %s -> %s", synced, ckpt)
            shutil.copyfile(synced, ckpt)
        if ckpt.exists():
            done = torch.load(ckpt, map_location="cpu", mmap=True, weights_only=True)[
                "n_done"
            ]
        step = args.fit_sync_every or len(prompts)
    for end in range(step, len(prompts) + step, step):
        end = min(end, len(prompts))
        lens = jlens.fit(
            lens_model,
            prompts[:end],
            dim_batch=args.dim_batch,
            max_seq_len=args.fit_seq_len,
            checkpoint_path=str(ckpt),
        )
        if ckpt != synced and end < len(prompts) and lens.n_prompts > done:
            log.info("syncing checkpoint (%d prompts) -> %s", lens.n_prompts, synced)
            shutil.copyfile(ckpt, synced)
            done = lens.n_prompts
    lens.save(str(path))
    log.info("saved %r -> %s", lens, path)
    if ckpt != synced and synced.exists():
        synced.unlink()


# --------------------------------------------------------------------------- #
# Stage 2: assistant axis (assistant-axis pipeline)
# --------------------------------------------------------------------------- #


def _run_pipeline_script(args, script: str, *script_args):
    cmd = [
        sys.executable,
        str(HERE / "compat.py"),
        str(args.assistant_axis_dir / "pipeline" / script),
        *map(str, script_args),
    ]
    log.info("$ %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=args.assistant_axis_dir / "pipeline")


def _axis_roles(args) -> list[str]:
    roles = args.axis_roles
    if roles is None:
        roles = list(
            json.loads(
                (args.assistant_axis_dir / "data/roles/role_list.json").read_text()
            )
        )
    return sorted(set(roles) | {"default"})


@torch.no_grad()
def generate_batch(
    hf, tok, convs, model_name, *, max_new_tokens, temperature=0.7, top_p=0.9
) -> list[str]:
    """Left-padded batched sampling; same template/sampling as assistant_axis.generate_response."""
    prompts = [
        tok.apply_chat_template(
            c, tokenize=False, add_generation_prompt=True, **chat_kwargs(model_name)
        )
        for c in convs
    ]
    side, tok.padding_side = tok.padding_side, "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    finally:
        tok.padding_side = side
    enc = enc.to(hf.device)
    out = hf.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tok.pad_token_id,
    )
    return [
        tok.decode(o[enc.input_ids.shape[1] :], skip_special_tokens=True).strip()
        for o in out
    ]


def _generate_hf(args, models: Models, roles, responses_dir: Path):
    """HF fallback for 1_generate.py (no vLLM): same prompts, same JSONL schema."""
    import jsonlines
    from assistant_axis import get_config
    from assistant_axis.generation import format_conversation

    hf, tok, _ = models.get()
    short_name = get_config(args.model)["short_name"]
    questions = [
        q["question"]
        for q in jsonlines.open(
            args.assistant_axis_dir / "data/extraction_questions.jsonl"
        )
    ]
    questions = questions[: args.axis_question_count]
    for role in roles:
        out_file = responses_dir / f"{role}.jsonl"
        if out_file.exists():
            continue
        data = json.loads(
            (
                args.assistant_axis_dir / f"data/roles/instructions/{role}.json"
            ).read_text()
        )
        instructions = [
            i.get("pos", "").replace("{model_name}", short_name)
            for i in data["instruction"]
        ]
        idxs = args.axis_prompt_indices or range(len(instructions))
        jobs = [
            (
                p_idx,
                q_idx,
                question,
                format_conversation(instructions[p_idx], question, tok),
            )
            for p_idx in idxs
            for q_idx, question in enumerate(questions)
        ]
        rows = []
        for b in range(0, len(jobs), args.axis_batch_size):
            batch = jobs[b : b + args.axis_batch_size]
            resps = generate_batch(
                hf,
                tok,
                [conv for *_, conv in batch],
                args.model,
                max_new_tokens=args.axis_max_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            for (p_idx, q_idx, question, conv), resp in zip(batch, resps, strict=True):
                rows.append(
                    {
                        "system_prompt": instructions[p_idx],
                        "prompt_index": p_idx,
                        "question_index": q_idx,
                        "question": question,
                        "conversation": conv + [{"role": "assistant", "content": resp}],
                        "label": "pos",
                    }
                )
        with jsonlines.open(out_file, "w") as w:
            w.write_all(rows)
        log.info("generated %d responses for role %s", len(rows), role)


def _write_unjudged_scores(roles, responses_dir: Path, scores_dir: Path):
    """--judge none: keep every role sample (score=3) so 4_vectors.py runs unchanged."""
    import jsonlines

    scores_dir.mkdir(parents=True, exist_ok=True)
    for role in roles:
        if role == "default":
            continue
        scores = {
            f"{r['label']}_p{r['prompt_index']}_q{r['question_index']}": 3
            for r in jsonlines.open(responses_dir / f"{role}.jsonl")
        }
        save_json(scores, scores_dir / f"{role}.json")


def stage_axis(args, models: Models):
    path = args.out / "axis.pt"
    if path.exists():
        log.info("axis exists at %s, skipping", path)
        return
    work = (args.out / "axis_pipeline").resolve()
    responses, acts, scores, vectors = (
        work / d for d in ("responses", "activations", "scores", "vectors")
    )
    if args.scratch_dir:
        acts = (args.scratch_dir / "axis_activations").resolve()
    for d in (responses, acts, scores, vectors):
        d.mkdir(parents=True, exist_ok=True)
    roles = _axis_roles(args)
    data = args.assistant_axis_dir / "data"

    use_vllm = args.generator == "vllm" or (
        args.generator == "auto"
        and torch.cuda.is_available()
        and importlib.util.find_spec("vllm") is not None
    )
    if use_vllm:
        if args.axis_prompt_indices:
            log.warning(
                "1_generate.py uses all instruction variants; ignoring --axis_prompt_indices"
            )
        models.free()
        _run_pipeline_script(
            args,
            "1_generate.py",
            "--model",
            args.model,
            "--output_dir",
            responses,
            "--roles_dir",
            data / "roles/instructions",
            "--questions_file",
            data / "extraction_questions.jsonl",
            "--question_count",
            args.axis_question_count,
            "--max_tokens",
            args.axis_max_tokens,
            "--roles",
            *roles,
        )
    else:
        _generate_hf(args, models, roles, responses)
    models.free()

    _run_pipeline_script(
        args,
        "2_activations.py",
        "--model",
        args.model,
        "--responses_dir",
        responses,
        "--output_dir",
        acts,
        "--batch_size",
        args.act_batch_size,
        "--roles",
        *roles,
    )

    judge = (
        args.judge
        if args.judge != "auto"
        else ("openai" if os.environ.get("OPENAI_API_KEY") else "none")
    )
    if judge == "openai":
        _run_pipeline_script(
            args,
            "3_judge.py",
            "--responses_dir",
            responses,
            "--output_dir",
            scores,
            "--roles_dir",
            data / "roles/instructions",
            "--roles",
            *roles,
        )
    else:
        log.warning("no judge: all role samples count as fully role-playing (score=3)")
        _write_unjudged_scores(roles, responses, scores)

    _run_pipeline_script(
        args,
        "4_vectors.py",
        "--activations_dir",
        acts,
        "--scores_dir",
        scores,
        "--output_dir",
        vectors,
        "--min_count",
        args.min_count,
    )
    _run_pipeline_script(
        args, "5_axis.py", "--vectors_dir", vectors, "--output", work / "axis_raw.pt"
    )

    from assistant_axis import load_axis

    raw = load_axis(
        str(work / "axis_raw.pt")
    ).float()  # (n_layers, d) = mean(default) - mean(role)
    axis_norm = raw.norm(dim=-1)
    unit = raw / axis_norm.clamp_min(1e-8)[:, None]
    default_acts = torch.stack(
        list(torch.load(acts / "default.pt", weights_only=False).values())
    ).float()
    resid_norm = default_acts.norm(dim=-1).mean(
        0
    )  # mean per-layer residual norm over default responses
    torch.save(
        {
            "unit": unit,
            "axis_norm": axis_norm,
            "resid_norm": resid_norm,
            "raw": raw,
            "judge": judge,
            "roles": roles,
            "model": args.model,
        },
        path,
    )
    log.info(
        "saved unit axis %s (judge=%s, %d roles) -> %s",
        tuple(unit.shape),
        judge,
        len(roles) - 1,
        path,
    )


# --------------------------------------------------------------------------- #
# Stage 3: drift sweep
# --------------------------------------------------------------------------- #


def load_axis_bundle(args):
    bundle = torch.load(args.out / "axis.pt", weights_only=False)
    from assistant_axis import get_config

    layer = (
        args.steer_layer
        if args.steer_layer is not None
        else get_config(args.model)["target_layer"]
    )
    scale = float(bundle[args.scale][layer])
    return bundle, layer, scale


def steering_ctx(hf, bundle, layer, coeff):
    """assistant-axis additive steering on the unit axis (no-op at coeff 0)."""
    from contextlib import nullcontext

    from assistant_axis import ActivationSteering

    if coeff == 0:
        return nullcontext()
    return ActivationSteering(
        hf,
        steering_vectors=[bundle["unit"][layer]],
        coefficients=[coeff],
        layer_indices=[layer],
        intervention_type="addition",
        positions="all",
    )


def condition_name(s: float) -> str:
    return f"steer_{s:+g}".replace("+", "p").replace("-", "m")


def stage_sweep(args, models: Models):
    path = args.out / "transcripts.json"
    if path.exists():
        log.info("transcripts exist at %s, skipping", path)
        return
    from assistant_axis import generate_response

    organic_dir = args.assistant_axis_dir / "transcripts/persona_drift"
    names = (
        args.organic
        if args.organic is not None
        else sorted(p.stem for p in organic_dir.glob("*.json"))
    )
    organic = {}
    for name in names:
        src = json.loads((organic_dir / f"{name}.json").read_text())
        user_turns = [t["content"] for t in src["conversation"] if t["role"] == "user"][
            : args.organic_turns
        ]
        organic[name] = (src, user_turns)

    hf, tok, _ = models.get()
    bundle, layer, scale = load_axis_bundle(args)
    probes = json.loads(Path(args.probes).read_text())
    records = []
    for s in args.strengths:
        coeff = s * scale
        with torch.no_grad(), steering_ctx(hf, bundle, layer, coeff):
            for category, items in probes.items():
                for i, prompt in enumerate(items):
                    conv = [{"role": "user", "content": prompt}]
                    resp = generate_response(
                        hf,
                        tok,
                        conv,
                        max_new_tokens=args.max_new_tokens,
                        do_sample=False,
                    )
                    records.append(
                        {
                            "id": f"{condition_name(s)}__{category}_{i}",
                            "condition": condition_name(s),
                            "s": s,
                            "coeff": coeff,
                            "steer_layer": layer,
                            "category": category,
                            "probe_id": f"{category}_{i}",
                            "conversation": conv
                            + [{"role": "assistant", "content": resp}],
                        }
                    )
                    log.info(
                        "[s=%+g] %s_%d: %s",
                        s,
                        category,
                        i,
                        resp[:80].replace("\n", " "),
                    )

    for name, (src, user_turns) in organic.items():
        conv = []
        with torch.no_grad():
            for u in user_turns:
                conv.append({"role": "user", "content": u})
                resp = generate_response(
                    hf,
                    tok,
                    conv,
                    max_new_tokens=args.organic_max_new_tokens,
                    do_sample=False,
                )
                conv.append({"role": "assistant", "content": resp})
        records.append(
            {
                "id": f"organic__{name}",
                "condition": f"organic_{name}",
                "s": 0.0,
                "coeff": 0.0,
                "steer_layer": layer,
                "category": "organic",
                "probe_id": name,
                "user_persona": src.get("persona"),
                "topic": src.get("topic"),
                "conversation": conv,
            }
        )
        log.info("organic %s: %d turns", name, len(user_turns))
    save_json(
        {
            "model": args.model,
            "steer_layer": layer,
            "scale_mode": args.scale,
            "scale": scale,
            "strengths": args.strengths,
            "records": records,
        },
        path,
    )


# --------------------------------------------------------------------------- #
# Stage 4: J-space readout
# --------------------------------------------------------------------------- #


def token_set_ids(tok, words: list[str]) -> dict[int, str]:
    """Single-token ids for each word in its bare / space-prefixed / capitalised forms."""
    ids = {}
    for w in words:
        for v in {
            w,
            " " + w,
            w.capitalize(),
            " " + w.capitalize(),
            w.lower(),
            " " + w.lower(),
        }:
            enc = tok.encode(v, add_special_tokens=False)
            if len(enc) == 1:
                ids[enc[0]] = v
    return ids


def default_band(n_layers: int) -> tuple[int, int]:
    return n_layers // 3, (2 * n_layers) // 3


@torch.no_grad()
def readout_one(rec, models, lens, bundle, token_sets, args) -> dict:
    hf, tok, lm = models.get()
    from assistant_axis.internals import ConversationEncoder

    conv = rec["conversation"]
    ids = chat_ids(tok, conv, args.model)[: args.max_seq_len]
    enc = ConversationEncoder(tok, args.model)
    turns = enc.response_indices(conv, per_turn=True, **chat_kwargs(args.model))
    positions, turn_of = [], []
    for t, idx in enumerate(turns):
        keep = [p for p in idx if p + 1 < len(ids)]
        positions += keep
        turn_of += [t] * len(keep)
    n_layers = lm.n_layers
    final = n_layers - 1
    read_layers = sorted(set(lens.source_layers) | {final})
    pos = torch.tensor(positions, dtype=torch.long)
    emitted = torch.tensor([ids[p + 1] for p in positions], dtype=torch.long)

    input_ids = torch.tensor([ids], device=lm.input_device)
    with steering_ctx(hf, bundle, rec["steer_layer"], rec["coeff"]):
        with ActivationRecorder(lm.layers, at=range(n_layers)) as recorder:
            lm.forward(input_ids)
    acts = {l: recorder.activations[l][0].float() for l in range(n_layers)}

    unit = bundle["unit"].to(acts[0].device)
    proj_layers = torch.stack(
        [acts[l][pos] @ unit[l] for l in range(n_layers)]
    )  # [n_layers, n_pos]

    set_ids = {
        name: torch.tensor(sorted(d), dtype=torch.long)
        for name, d in token_sets.items()
    }
    ranks = {
        name: np.zeros((len(read_layers), len(positions)), dtype=np.int32)
        for name in set_ids
    }
    ranks["emitted"] = np.zeros((len(read_layers), len(positions)), dtype=np.int32)
    top_ids = np.zeros((len(read_layers), len(positions), args.top_k), dtype=np.int32)
    for li, l in enumerate(read_layers):
        resid = acts[l][pos]
        if l in lens.jacobians:
            resid = lens.transport(resid, l)
        logits = lm.unembed(resid).float()
        top_ids[li] = logits.topk(args.top_k, dim=-1).indices.cpu().numpy()
        for name, sid in set_ids.items():
            ranks[name][li] = (
                _ranks_of(logits, sid.to(logits.device)).min(-1).values.cpu().numpy()
            )
        ranks["emitted"][li] = (
            _ranks_of(logits, emitted.to(logits.device)[:, None])
            .squeeze(-1)
            .cpu()
            .numpy()
        )
        del logits

    def dec(t):
        return tok.decode([int(t)], clean_up_tokenization_spaces=False)

    emitted_in = {
        name: [int(e) in token_sets[name] for e in emitted.tolist()]
        for name in token_sets
    }
    return {
        "id": rec["id"],
        "condition": rec["condition"],
        "s": rec["s"],
        "category": rec["category"],
        "probe_id": rec["probe_id"],
        "layers": read_layers,
        "n_layers": n_layers,
        "positions": positions,
        "turn": turn_of,
        "emitted_tokens": [dec(e) for e in emitted.tolist()],
        "emitted_in_set": emitted_in,
        "top1": [[dec(t) for t in top_ids[li, :, 0]] for li in range(len(read_layers))],
        "ranks": {k: v.tolist() for k, v in ranks.items()},
        "proj": proj_layers.cpu().tolist(),
        "_top_ids": top_ids,
    }


def stage_readout(args, models: Models):
    out_dir = args.out / "readout"
    out_dir.mkdir(parents=True, exist_ok=True)
    hf, tok, lm = models.get()
    lens = jlens.JacobianLens.load(str(args.out / "lens.pt"))
    bundle, _, _ = load_axis_bundle(args)
    transcripts = json.loads((args.out / "transcripts.json").read_text())
    words = json.loads(Path(args.token_sets).read_text())
    token_sets = {name: token_set_ids(tok, ws) for name, ws in words.items()}
    band = args.band or default_band(lm.n_layers)
    save_json(
        {
            "token_sets": {
                k: {str(i): v for i, v in d.items()} for k, d in token_sets.items()
            },
            "band": list(band),
            "hit_rank": args.hit_rank,
            "source_layers": lens.source_layers,
            "n_layers": lm.n_layers,
            **{
                k: transcripts[k]
                for k in ("model", "steer_layer", "scale_mode", "scale")
            },
        },
        args.out / "readout_meta.json",
    )
    for rec in transcripts["records"]:
        f = out_dir / f"{rec['id']}.json"
        if f.exists():
            continue
        r = readout_one(rec, models, lens, bundle, token_sets, args)
        np.savez_compressed(
            out_dir / f"{rec['id']}_topk.npz",
            top_ids=r.pop("_top_ids"),
            layers=r["layers"],
            positions=r["positions"],
        )
        save_json(r, f)
        log.info("readout %s (%d positions)", rec["id"], len(r["positions"]))

    pinned = set().union(*[set(d) for d in token_sets.values()])
    slice_dir = args.out / "slices"
    slice_dir.mkdir(parents=True, exist_ok=True)
    for rec in transcripts["records"]:
        if rec["probe_id"] != args.slice_probe and rec["category"] != "organic":
            continue
        page_path = slice_dir / f"{rec['condition']}.html"
        if page_path.exists():
            continue
        conv = rec["conversation"]
        text = tok.apply_chat_template(
            conv, tokenize=False, add_generation_prompt=False, **chat_kwargs(args.model)
        )
        prefix = tok.apply_chat_template(
            conv[:-1],
            tokenize=False,
            add_generation_prompt=True,
            **chat_kwargs(args.model),
        )
        n_resp = len(lm.encode(text, max_length=args.max_seq_len)[0]) - len(
            lm.encode(prefix, max_length=args.max_seq_len)[0]
        )
        with steering_ctx(hf, bundle, rec["steer_layer"], rec["coeff"]):
            sd = compute_slice(
                lm,
                lens,
                text,
                pinned_token_ids=pinned,
                last_n_tokens=n_resp + 1,
                max_seq_len=args.max_seq_len,
                top_n=args.top_k,
            )
        page, _, _ = build_page(
            sd,
            text,
            title=f"persona drift: {rec['condition']}",
            description=f"{args.model} | s={rec['s']:+g} (coeff {rec['coeff']:.2f} @ layer "
            f"{rec['steer_layer']}) | probe {rec['probe_id']} | last assistant turn",
            pinned_token_ids=pinned,
        )
        page_path.write_text(page)
        log.info("slice page -> %s", page_path)


# --------------------------------------------------------------------------- #
# Stage 5: analyze
# --------------------------------------------------------------------------- #


def stage_analyze(args, models: Models):
    sys.path.insert(0, str(HERE))
    import analysis

    summary = analysis.run(args.out, hit_rank=args.hit_rank)
    log.info(
        "summary -> %s\n%s",
        args.out / "summary.json",
        json.dumps(summary["headline"], indent=1),
    )


def main(argv=None):
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    for noisy in ("httpx", "httpx2", "huggingface_hub", "datasets"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        args.out / "config.json",
    )
    models = Models(args)
    for stage in STAGES:
        if stage in args.stages:
            log.info("=== stage: %s ===", stage)
            globals()[f"stage_{stage}"](args, models)


if __name__ == "__main__":
    main()
