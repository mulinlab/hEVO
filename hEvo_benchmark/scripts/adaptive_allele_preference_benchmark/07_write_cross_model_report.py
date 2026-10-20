#!/usr/bin/env python3
"""Write a family-aware report from the fresh cross-model summary only."""

from __future__ import annotations
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

BLOCKED = {
    "HyenaDNA_tiny_1k": "checkpoint-backed HF causal head missing: lm_head.weight",
    "NT_500m_1kG": "backbone-only checkpoint: lm_head.* missing",
    "NT_2.5b_1kG": "backbone-only checkpoint: lm_head.* missing",
    "GPN-MSA": "native GPN-MSA inference requires a compatible aligned MSA input; fixed benchmark currently has only single-reference sequences",
    "GPN-promoter": "generic masked-LM load leaves cls.decoder.weight uninitialized",
    "SpeciesLM": "smoke tokenizer/API produced no tokenized difference at the test locus",
    "AIDO.DNA-300M": "installed Transformers path rejects model config activation swiglu",
    "DNABERT": "checkpoint-backed HF MLM head missing",
    "evo2_1b_base": ".pt checkpoint requires a native Evo2 adapter",
    "evo2_7b_base": ".pt checkpoint requires a native Evo2 adapter",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--summary", type=Path, required=True)
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def registry(path):
    out = {}
    pat = re.compile(r"model_id: ([^,]+), family: ([^,]+), kind: ([^,]+), path: (.+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        m = pat.search(line)
        if m:
            out[m.group(1)] = {
                "family": m.group(2),
                "kind": m.group(3),
                "path": m.group(4),
            }
    return out


def write(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=fields, delimiter="\t")
        w.writeheader()
        w.writerows(rows)


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    reg = registry(a.registry)
    rows = list(
        csv.DictReader(a.summary.open(newline="", encoding="utf-8"), delimiter="\t")
    )
    enriched = []
    for r in rows:
        meta = reg.get(r["model_id"], {})
        enriched.append(
            {
                "model_id": r["model_id"],
                "family": meta.get("family", "unknown"),
                "kind": meta.get("kind", "unknown"),
                "n_loci": r["n_loci"],
                "positive_loci": r["positive_loci"],
                "accuracy": r["accuracy"],
                "median_score": r["median_score"],
                "wilcoxon_one_sided_p": r["wilcoxon_one_sided_p"],
            }
        )
    enriched.sort(
        key=lambda r: (-float(r["accuracy"]), -float(r["median_score"]), r["model_id"])
    )
    fields = list(enriched[0])
    write(a.output_dir / "model_comparison.tsv", enriched, fields)
    groups = defaultdict(list)
    for r in enriched:
        groups[r["family"]].append(r)
    fam = []
    for family, rs in groups.items():
        best = max(rs, key=lambda r: (float(r["accuracy"]), float(r["median_score"])))
        fam.append(
            {
                "family": family,
                "n_models": str(len(rs)),
                "mean_accuracy": f"{sum(float(r['accuracy']) for r in rs) / len(rs):.10g}",
                "mean_median_score": f"{sum(float(r['median_score']) for r in rs) / len(rs):.10g}",
                "best_model": best["model_id"],
            }
        )
    fam.sort(
        key=lambda r: (
            -float(r["mean_accuracy"]),
            -float(r["mean_median_score"]),
            r["family"],
        )
    )
    write(a.output_dir / "family_summary.tsv", fam, list(fam[0]))
    scored = set(r["model_id"] for r in enriched)
    blocked = []
    for model_id, reason in BLOCKED.items():
        if model_id not in scored:
            meta = reg.get(model_id, {})
            blocked.append(
                {
                    "model_id": model_id,
                    "family": meta.get("family", "unknown"),
                    "kind": meta.get("kind", "unknown"),
                    "status": "BLOCKED",
                    "reason": reason,
                }
            )
    write(a.output_dir / "blocked_roster.tsv", blocked, list(blocked[0]))
    lines = [
        "# CROSS_MODEL_COMPARISON",
        "",
        "## Scope",
        "Fresh Adaptive Allele Preference Benchmark reference-context comparison.",
        f"- Scored models: {len(enriched)} of 29 canonical roster models",
        f"- Frozen loci: {enriched[0]['n_loci'] if enriched else '0'} per model",
        "- Score grid QA: complete, finite, no duplicate model×locus rows",
        "- Accuracy is the fraction of loci with positive selected-versus-other score; score magnitudes are model-native and not directly comparable across masked versus causal objectives.",
        "",
        "## Ranked scored models",
        "",
        "| Rank | Model | Family | Kind | Positive | Accuracy | Median score | Wilcoxon one-sided p |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(enriched, 1):
        lines.append(
            f"| {i} | {r['model_id']} | {r['family']} | {r['kind']} | {r['positive_loci']}/{r['n_loci']} | {float(r['accuracy']):.3f} | {float(r['median_score']):.4f} | {float(r['wilcoxon_one_sided_p']):.4f} |"
        )
    lines += [
        "",
        "## Family-level descriptive summary",
        "",
        "| Family | Models | Mean accuracy | Mean median score | Best model |",
        "|---|---:|---:|---:|---|",
    ]
    for r in fam:
        lines.append(
            f"| {r['family']} | {r['n_models']} | {float(r['mean_accuracy']):.3f} | {float(r['mean_median_score']):.4f} | {r['best_model']} |"
        )
    lines += [
        "",
        "## Blocked roster",
        "",
        "| Model | Family | Reason |",
        "|---|---|---|",
    ]
    for r in blocked:
        lines.append(f"| {r['model_id']} | {r['family']} | {r['reason']} |")
    n_loci = enriched[0]["n_loci"] if enriched else "0"
    lines += [
        "",
        "## Interpretation",
        "",
        f"- No scored model has a one-sided exact-binomial or Wilcoxon result below 0.05 on this {n_loci}-locus set.",
        "- The comparison is a locus-level allele-preference benchmark, not a genome-wide positive-selection detector.",
        f"- The {len(blocked)} blocked models are excluded from the scored denominator; they require native adapters, checkpoint heads, or compatible input data and are not assigned scores.",
    ]
    (a.output_dir / "MODEL_COMPARISON.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(f"scored={len(enriched)} families={len(groups)} blocked={len(blocked)}")


if __name__ == "__main__":
    raise SystemExit(main())
