"""Fixed held-out HAQER loci for the Fig. 4e hEvo-large/GPN-Star-M comparison."""
from __future__ import annotations

import json
import os
from pathlib import Path

HERE = Path(__file__).resolve()
BENCHMARK = HERE.parents[3]
HEVO_RUN = BENCHMARK / "results/haqer_large_vs_evo2_v3_refined"
GPN_RUN = Path(os.environ.get(
    "GPN_HAQER_RUN",
    str(Path(os.environ.get("DNAMODELS_BENCHMARK_ROOT", Path.home() / "DNAModelsBenchmark"))
        / "runs/gpn_star_m_har_haqer_v1"),
))
GOLD_BED = BENCHMARK / "datasets/HAQER_golden_windwos/haqer.hg38.bed"

# Twelve high-contrast held-out loci, with HAQER0218 retained as a COL4A2-near
# example. At the shared 0.59 cutoff, selection requires hEvo summit-in-gold,
# no GPN called summit within the displayed +/-2 kb context, and a smoothed
# local-score margin >= 0.15.
SELECTED_IDS = (
    "HAQER1223", "HAQER1230", "HAQER0333", "HAQER1262", "HAQER0258",
    "HAQER0521", "HAQER1272", "HAQER1411", "HAQER0679", "HAQER0608",
    "HAQER0412", "HAQER0218",
)
LITERATURE_CONTEXT = {
    "HAQER0218": "near COL4A2; nearby-gene context only",
}
FLANK_BP = 2_000
DISPLAY_GAP_BP = 800


def selected_segments() -> list[dict]:
    hevo_regions = json.loads((HEVO_RUN / "regions.json").read_text())
    gpn_regions = json.loads((GPN_RUN / "regions.json").read_text())
    gold = {}
    for line in GOLD_BED.read_text().splitlines():
        fields = line.split()
        if len(fields) >= 4:
            gold[fields[3]] = (fields[0], int(fields[1]), int(fields[2]))

    result = []
    offset = 0
    for index, ident in enumerate(SELECTED_IDS):
        chrom, gold_start, gold_end = gold[ident]
        gpn_region = next(
            region for region in gpn_regions
            if region.get("task") == "haqer"
            and region.get("split") in ("test", "val")
            and any(item["id"] == ident for item in region.get("known", []))
        )
        hevo_region = next(
            region for region in hevo_regions
            if region["chrom"] == chrom
            and region["split"] == gpn_region["split"]
            and region["start"] < gold_end and region["end"] > gold_start
        )
        midpoint = (gold_start + gold_end) // 2
        start = max(
            int(gpn_region["scan_start"]), int(hevo_region["scan_start"]),
            midpoint - FLANK_BP,
        )
        end = min(
            int(gpn_region["scan_end"]), int(hevo_region["scan_end"]),
            midpoint + FLANK_BP,
        )
        if start >= end:
            raise ValueError(f"Empty display interval for {ident}")
        if hevo_region["split"] == "train" or gpn_region["split"] == "train":
            raise ValueError(f"Training locus selected: {ident}")
        segment = {
            "id": ident,
            "index": index,
            "chrom": chrom,
            "start": start,
            "end": end,
            "gold_start": gold_start,
            "gold_end": gold_end,
            "split": gpn_region["split"],
            "hevo_region_id": hevo_region["id"],
            "gpn_region_id": gpn_region["id"],
            "display_start": offset,
            "offset": offset,
            "display_end": offset + end - start,
            "literature_context": LITERATURE_CONTEXT.get(ident, ""),
        }
        result.append(segment)
        offset = segment["display_end"] + DISPLAY_GAP_BP

    for i, first in enumerate(result):
        for second in result[i + 1:]:
            if first["chrom"] == second["chrom"] and not (
                first["end"] + DISPLAY_GAP_BP <= second["start"]
                or second["end"] + DISPLAY_GAP_BP <= first["start"]
            ):
                raise ValueError(f"Overlapping display contexts: {first['id']} and {second['id']}")
    return result


def write_manifest(path: Path) -> dict:
    segments = selected_segments()
    document = {
        "schema": "fig4e_hevo_gpn_selected_examples.v1",
        "assembly": "hg38",
        "coordinate_system": "0-based half-open",
        "selection": (
            "12 held-out HAQERs at shared score cutoff 0.59; hEvo summit in gold; no GPN called peak in +/-2 kb; "
            "smoothed local-score difference >= 0.15; narrow hEvo peaks; includes COL4A2-near HAQER0218"
        ),
        "primary_seed": 20260515,
        "display_flank_bp": FLANK_BP,
        "display_gap_bp": DISPLAY_GAP_BP,
        "score_note": "Both models use a common 0.59 calling cutoff; their HAQER heads are calibrated independently, so their full score distributions are not guaranteed to be identically calibrated.",
        "segments": segments,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return document


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK / "tables/fig4e_haqer_tracks/selected_segments.json",
    )
    args = parser.parse_args()
    print(json.dumps(write_manifest(args.output), indent=2))
