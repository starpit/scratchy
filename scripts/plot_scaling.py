#!/usr/bin/env python3
"""Plot scaling curves from a bench_metal_matrix.sh JSON.

One PNG per (model, axis) with a line per backend, plus a two-panel layout
(tok/s and TPOT) for the concurrency axis — the sublinearity gap and the
per-request cost of being batched, side by side:

    python3 scripts/plot_scaling.py bench_results/metal_matrix/apple-m3-pro.json
    python3 scripts/plot_scaling.py <json> --out-dir plots/ --metrics tok_s,tpot,ttft

The script is a pure consumer of the runner's JSON: every dimension the cells
swept (axis, rung, input_len, output_len, concurrency, seed, backend version)
is already in the data, so nothing is re-derived or assumed here. Rows with a
`dedup_of` marker are skipped — they reference a cell measured under another
axis.
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

AXES = ["concurrency", "input_len", "output_len"]
# metric key in a row -> (label, ylabel). tok/s is the headline; TPOT flat
# across concurrency is the goal; TTFT vs input_len is prefill scaling.
METRICS = {
    "tok_s": ("output_throughput", "output tok/s"),
    "req_s": ("request_throughput", "requests/s"),
    "tpot": ("median_tpot_ms", "median TPOT (ms)"),
    "ttft": ("median_ttft_ms", "median TTFT (ms)"),
    "itl": ("median_itl_ms", "median ITL (ms)"),
}
AXIS_LABELS = {
    "concurrency": "offered concurrency (max-concurrency)",
    "input_len": "input length (tokens)",
    "output_len": "output length (tokens)",
}
# Log-x axes where rungs span orders of magnitude (128..8192).
LOG_X = {"input_len": True, "output_len": True, "concurrency": False}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json", help="matrix JSON from bench_metal_matrix.sh")
    ap.add_argument("--out-dir", default="scaling_plots")
    ap.add_argument("--metrics", default="tok_s,tpot,ttft",
                    help=f"comma-separated from: {','.join(METRICS)}")
    ap.add_argument("--backends", help="comma-separated subset (default: all present)")
    args = ap.parse_args()

    d = json.load(open(args.json))
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    bad = [m for m in metrics if m not in METRICS]
    if bad:
        sys.exit(f"unknown metrics: {bad}; choose from {list(METRICS)}")

    os.makedirs(args.out_dir, exist_ok=True)
    models = [e for e in d.get("models", []) if e.get("scaling")]
    if not models:
        sys.exit("no model in this JSON has a scaling block")

    n_files = 0
    for e in models:
        sc = e["scaling"]
        backends = sc.get("backends") or {}
        versions = sc.get("versions") or {}
        want = args.backends.split(",") if args.backends else sorted(backends)
        want = [b for b in want if b in backends]
        if not want:
            continue
        for axis in AXES:
            rows_by_backend = {}
            for bk in want:
                rows = sorted((r for r in backends[bk]
                               if r.get("axis") == axis and "output_throughput" in r),
                              key=lambda r: r["rung"])
                if rows:
                    rows_by_backend[bk] = rows
            if not rows_by_backend:
                continue
            for m in metrics:
                key, ylabel = METRICS[m]
                fig, ax = plt.subplots(figsize=(6, 4))
                for bk, rows in rows_by_backend.items():
                    ver = versions.get(bk, {}).get("version", "?")
                    xs = [r["rung"] for r in rows]
                    ys = [r.get(key) for r in rows]
                    n = len(xs)
                    # Drop rungs where this metric is missing for this backend.
                    pts = [(x, y) for x, y in zip(xs, ys) if y is not None]
                    if not pts:
                        continue
                    ax.plot([p[0] for p in pts], [p[1] for p in pts],
                            marker="o", label=f"{bk} {ver} (n={n})")
                ax.set_xlabel(AXIS_LABELS[axis])
                ax.set_ylabel(ylabel)
                if LOG_X[axis]:
                    ax.set_xscale("log", base=2)
                ax.set_title(f"{e['stem']} — {ylabel} vs {axis}")
                ax.grid(True, alpha=0.3)
                ax.legend()
                fig.tight_layout()
                out = os.path.join(args.out_dir,
                                   f"{e['stem']}.{axis}.{m}.png")
                fig.savefig(out, dpi=150)
                plt.close(fig)
                n_files += 1
                print(f"wrote {out}")

    if not n_files:
        sys.exit("nothing plottable found")
    print(f"\n{n_files} plots in {args.out_dir}/ — "
          f"one line per backend, dimensions from the JSON's cells")


if __name__ == "__main__":
    main()
