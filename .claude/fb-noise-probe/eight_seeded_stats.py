#!/usr/bin/env python3
"""The 8b fp8 seeded leg's CARD VERDICT (front-b-noise, auditor order
2026-10-27, on the tree rebased onto 8143cf0ebe): "8b seeded P3 (the pair ×2,
deterministic) and P5 (200 draws within 3σ of main's 8b logits)".

Two parts, read from the leg's own artifacts (a verdict is never computed
over runs that did not happen):
  P3. 2-concurrent (1234 + 9999), run TWICE -> each request's text is
      IDENTICAL across the two pair runs (auditor ruling (b), 2026-10-27, the
      2b precedent: the batched-numerics class — the rows=2 rung is a
      different baked tape than solo, the same class main's own batched greedy
      shows, and main has no seeded-batched reference — so the pair is gated
      against ITSELF, never equated with solo; the pair-vs-solo comparison is
      reported as information only).
  P5. 200 seeded single-token draws at temp 1.0 -> each top-10 token's
      frequency vs softmax(MAIN's 8b host logits) — main8-logits-leg.sh's
      dump at /work/fb-chat-main8/logits-0000.bin — within 3 sigma (binomial).

Plus the ORDERED LOG EVIDENCE: the server log must SHOW the seat binding both
seeds (SEAT_BIND lines, both in ONE launch for the pair) and the batched rung
launching (PREFILL_PATH=batched / decode-batch lines), including the 2-row
launch. Exit 0 iff both gated parts pass and the evidence lines exist.
"""
import glob
import json
import math
import pathlib
from collections import Counter

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REF = pathlib.Path("/work/fb-chat-main8")
SRVLOG = HERE / "eight-seeded-server.log"
DRAWS = 200
Z_MAX = 3.0


def content(name: str) -> str:
    r = json.loads((HERE / name).read_text())
    return r["choices"][0]["message"]["content"]


def first_div(x: str, y: str) -> dict:
    i = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]),
             min(len(x), len(y)))
    return {"at_char": i,
            "run_a": x[max(0, i - 20):i + 20],
            "run_b": y[max(0, i - 20):i + 20]}


def softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    e = np.exp(x - m)
    return e / e.sum()


def main() -> int:
    tok_path = sorted(
        glob.glob(
            "/root/.cache/huggingface/hub/models--RedHatAI--granite-3.1-8b-instruct-FP8-dynamic/"
            "snapshots/*/tokenizer.json"
        )
        + glob.glob(
            str(pathlib.Path.home() / ".cache/huggingface/hub/"
                "models--RedHatAI--granite-3.1-8b-instruct-FP8-dynamic/snapshots/*/tokenizer.json")
        )
        + glob.glob(
            "/work/huggingface/hub/"
            "models--RedHatAI--granite-3.1-8b-instruct-FP8-dynamic/snapshots/*/tokenizer.json"
        )
    )
    if not tok_path:
        print("no 8b tokenizer.json in the HF cache — cannot map draw texts to ids")
        return 1
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(tok_path[0])

    report: dict = {
        "probe": "8b-fp8-seeded",
        "model": "RedHatAI/granite-3.1-8b-instruct-FP8-dynamic",
        "prompt": "Write two sentences about the ocean.",
        "note": "rebased onto 8143cf0ebe; P3 + P5 only per the auditor's order",
    }
    ok = True

    # ── P3: the concurrent pair, run twice — the pair against ITSELF ──
    da = content("eight-p3-seed1234-a.json")
    db = content("eight-p3-seed1234-b.json")
    ea = content("eight-p3-seed9999-a.json")
    eb = content("eight-p3-seed9999-b.json")
    p3a = da == db
    p3b = ea == eb
    report["p3_concurrent_pair_deterministic"] = {
        "seed1234_run_a_equals_run_b": p3a,
        "seed9999_run_a_equals_run_b": p3b,
        "gate": "the concurrent pair against itself across two runs (ruling "
                "(b), 2026-10-27 — batched numerics, never equated with solo)",
    }
    if not p3a:
        report.setdefault("p3_first_divergence", {})["seed1234"] = first_div(da, db)
    if not p3b:
        report.setdefault("p3_first_divergence", {})["seed9999"] = first_div(ea, eb)
    report["texts"] = {
        "concurrent_seed1234_run_a": da, "concurrent_seed1234_run_b": db,
        "concurrent_seed9999_run_a": ea, "concurrent_seed9999_run_b": eb,
    }
    ok = ok and p3a and p3b

    # ── P5: the draws vs softmax(main's 8b host logits) ──
    logits = np.fromfile(REF / "logits-0000.bin", dtype="<f4")
    p = softmax(logits.astype(np.float64))
    top10 = np.argsort(p)[::-1][:10]
    # text -> id over the reference's own top 1000 (draws outside stay counted
    # against the complement, never silently dropped)
    top1000 = np.argsort(p)[::-1][:1000]
    text2id: dict = {}
    ambiguous = []
    for tid in top1000:
        s = tok.decode([int(tid)], skip_special_tokens=True)
        if s in text2id:
            ambiguous.append(s)
        text2id[s] = int(tid)

    draws = [json.loads(l) for l in (HERE / "eight-p5-draws.txt").read_text().splitlines()]
    assert len(draws) == DRAWS, f"expected {DRAWS} draws, read {len(draws)}"
    counts = Counter(draws)
    rows = []
    for tid in top10:
        s = tok.decode([int(tid)], skip_special_tokens=True)
        obs = counts.get(s, 0) if s not in ambiguous else None
        exp = DRAWS * p[tid]
        sig = math.sqrt(DRAWS * p[tid] * (1 - p[tid]))
        z = (obs - exp) / max(sig, 0.5) if obs is not None else None
        rows.append({"id": int(tid), "text": s, "p": float(p[tid]),
                     "expected": exp, "observed": obs, "sigma": sig, "z": z,
                     "ambiguous_text": s in ambiguous})
    in_top1000 = sum(n for t, n in counts.items() if t in text2id)
    outside = {t: n for t, n in counts.items() if t not in text2id}
    report["p5_distribution"] = {
        "draws": DRAWS, "seeds": "100000..100199", "temperature": 1.0,
        "reference": "main's 8b host logits (main8-logits-leg, main e63c6f8ac8)",
        "ambiguous_texts": ambiguous,
        "top10": rows,
        "pass_within_3sigma": all(
            r["z"] is not None and abs(r["z"]) <= Z_MAX for r in rows
        ),
        "draws_in_ref_top1000": in_top1000,
        "expected_in_ref_top1000": DRAWS * float(p[top1000].sum()),
        "draws_outside_top1000": {t: n for t, n in list(outside.items())[:10]},
        "n_outside": sum(outside.values()),
    }
    p5 = report["p5_distribution"]["pass_within_3sigma"]
    ok = ok and p5

    # ── the ordered log evidence ──
    log = SRVLOG.read_text(errors="replace")
    lines = log.splitlines()
    seat_lines = [l for l in lines if "SEAT_BIND" in l]
    ev = {
        "seat_bind_lines": len(seat_lines),
        "seat_bound_seed_1234": any("seed=1234" in l for l in seat_lines),
        "seat_bound_seed_9999": any("seed=9999" in l for l in seat_lines),
        "both_seeds_one_launch": any("seed=1234" in l and "seed=9999" in l
                                     for l in seat_lines),
        "prefill_batched_lines": log.count("PREFILL_PATH=batched"),
        "decode_batch_lines": sum(1 for l in lines
                                  if "decode batch:" in l and "rung" in l),
        "two_row_rung_line": any("2 live on the 2-row rung" in l for l in lines),
        "seat_readback_lines": log.count("SEAT_READBACK"),
        "refusals_or_panics": [l.strip()[:200] for l in lines
                               if "REFUSING" in l or "panicked" in l][:10],
        "sample_seat_lines": [l.strip()[:200] for l in seat_lines[:6]],
    }
    report["log_evidence"] = ev
    ok = ok and ev["seat_bind_lines"] > 0 and ev["both_seeds_one_launch"] \
        and ev["prefill_batched_lines"] > 0 and ev["decode_batch_lines"] > 0 \
        and not ev["refusals_or_panics"]

    (HERE / "eight-seeded-verdict.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not ok:
        print("EIGHT-SE-VERDICT-FAILED (see report above)")
        return 1
    print("EIGHT-SE-VERDICT-PASS (P3 pair x2 deterministic, P5 within 3"
          " sigma of main's 8b logits, log evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
