#!/usr/bin/env python3
"""Apply the approved manuscript corrections to tracked Word copies.

Ablation paragraphs and Fig5 a/b/c caption clauses are explicitly preserved
following the user's 2026-10-10 scope correction.
"""

from pathlib import Path
import argparse
import json
import zipfile

import pandas as pd
from lxml import etree as ET
from revise_docx import revise, visible, NS


def run(root, output):
    tables = root / "hEvo_benchmark/tables"

    def read(tid):
        return pd.read_csv(next(tables.glob(tid + "_*.csv")))

    resource = read("S43").set_index("model")
    shared = pd.read_csv(
        tables / "revision_consistency/common_task_rank_sensitivity.csv"
    ).set_index("model")
    inventory = json.loads(
        (tables / "revision_consistency/rank_inventory.json").read_text()
    )
    mpra = read("S29").set_index("model").auprc_mean
    improvement = (mpra["hEvo-large"] / mpra["GPN-promoter"] - 1) * 100
    panel = json.loads(
        pd.read_csv(tables / "fig4e_haqer_tracks/Fig4e_track_source_data.csv")
        .iloc[0]
        .payload_json
    )
    tm = panel["test_metrics"]
    h, g = tm["hevo"], tm["gpn"]
    hapref = read("S38").set_index("model_id")
    ncommon = inventory["common_task_units"]
    main = root / "article/Manuscript to XQY.docx"
    with zipfile.ZipFile(main) as z:
        doc = ET.fromstring(z.read("word/document.xml"))
    paragraphs = doc.findall(".//w:p", NS)
    patches = []

    def edit(i, old, new, reason):
        patches.append((i, old, new, reason))

    edit(
        20,
        "at least one of the two hEVO models achieved the highest aggregate score in 14 groups under the evaluated protocols.",
        "the two hEVO models showed task-dependent predictive strengths under the evaluated protocols. A source-linked endpoint inventory reports the metric and eligible comparator set for each endpoint; no single aggregate winner is inferred for the retrospective ClinVar analysis.",
        "Remove untraceable 14/15 group-winner claim while preserving the 48/15 inventory",
    )
    edit(
        32,
        "the better performing hEVO model achieved the highest aggregate score in 14 groups and the highest score at 34 endpoints under the evaluated protocols",
        "comparisons used the recorded primary metric and actual eligible model roster for each endpoint. The endpoint inventory is distinct from the 45 task units used for the cross-task rank analysis; aggregate group-winner and endpoint-winner counts are withheld until their original counting rules can be reproduced",
        "Separate endpoint inventory and rank universe; do not invent a retrospective endpoint aggregate",
    )
    old = "Across the six projections, hEVO-Large ranked first in the ENCODE and RefSeq PCA comparisons. GENA-LM-base led ENCODE t-SNE, GPN-Star-V led both GENCODE projections, and GPN-Star-P led RefSeq t-SNE"
    new = "The main Fig. 2c comparison shows CH indices from two-dimensional PCA for 19 representative models across ENCODE, GENCODE and RefSeq. hEVO-Large ranked first in the ENCODE and RefSeq PCA comparisons, whereas GPN-Star-V led the GENCODE PCA comparison. The separate PCA and t-SNE results are reported in Supplementary Fig. S4 and the source tables"
    edit(35, old, new, "Main Fig2c is PCA-only; t-SNE remains supplementary")
    edit(
        37,
        "compared with 0.766 for Genos-1.2B and 0.748 for Evo 2-7B",
        "compared with 0.840 for the strongest external comparator, GPN-Star-P",
        "Use the current S19 strongest external comparator",
    )
    edit(
        44,
        "compared with 0.284 for GPN-Star-P, the strongest external comparator, representing a relative improvement of 14.5%",
        f"compared with {mpra['GPN-promoter']:.3f} for GPN-Promoter, the strongest external comparator, representing a relative improvement of {improvement:.1f}%",
        "Correct MPRA comparator and recompute uplift from S29",
    )
    edit(
        44,
        "Expanded comparisons incorporating all tools retained these advantages",
        "Expanded comparisons under the specified primary scoring protocols are reported",
        "Avoid claiming invariance across scoring protocols",
    )
    rare = read("S27").set_index("model").auprc_mean
    rare_gain = (rare["hEvo-large"] / rare["GPN-MSA"] - 1) * 100
    edit(
        44,
        "compared with 0.150 and 0.433 for GPN-Star-M, the strongest external comparator in both analyses. These corresponded to relative improvements of 19.6% and 29.4%, respectively.",
        f"compared with 0.150 for GPN-Star-M in the broader set and {rare['GPN-MSA']:.3f} for GPN-MSA in the rare-causal set, the strongest external comparators under the reported raw protocols. These corresponded to relative improvements of 19.6% and {rare_gain:.1f}%, respectively.",
        "Use the current rare-causal GPN-MSA comparator and recompute uplift",
    )
    edit(
        39,
        "Full sequence benchmark comparisons include GPN-MSA and GPN-Star where applicable, alongside DeepSEA precision-recall results (Supplementary Figs. S5 and Supplementary Table S21).",
        "Expanded DeepSEA category-AUROC and macro-AUPRC comparisons, including public-data variants and GPN-Star-M, are reported in Supplementary Fig. S6 and Supplementary Table S21.",
        "Point DeepSEA precision-recall results to the actual supplement",
    )
    edit(46, "(Fig. 3i,k)", "(Fig. 3j,k)", "Current Fig3j is the upgrade trajectory")
    edit(46, "(Fig. 3j)", "(Fig. 3i)", "Current Fig3i is the downgrade trajectory")
    edit(
        44,
        "Together, hEVO combines strong clinical discrimination with improved prioritization of regulatory and trait-modulating variants, extending variant interpretation beyond pathogenicity to the allelic effects that shape human phenotypic diversity.",
        "These comparisons use the reported model-specific coverage and the primary raw scoring protocols. ClinVar discrimination is metric dependent: GPN-Star-M achieved AUROC 0.948, compared with 0.941 for hEVO-Base, although hEVO-Base had the higher reported raw-protocol AUPRC. GPN-Star cLLR results are provided separately as a sensitivity analysis; the primary raw scores are retained. On identical eligible variants, calibration changed GPN-Star-M AUPRC from 0.150 to 0.159 for CAUSALdb and from 0.662 to 0.720 for DDD external controls; its candidate-trio top-1 fraction changed from 0.278 to 0.480. Thus raw-protocol relative gains cannot be assumed to apply unchanged after calibration. The results support task-specific variant prioritization rather than uniform superiority across metrics or scoring protocols.",
        "Disclose ClinVar AUROC ordering, coverage and cLLR sensitivity",
    )
    ancestry = read("S34").set_index("model")
    edit(
        49,
        "(Fig. 4a and Supplementary Table S33)",
        "(Supplementary Fig. S8a and Supplementary Table S33)",
        "Unsupervised ancestry is supplementary in the current six-panel Fig4 layout",
    )
    edit(
        49,
        "achieved 98.73% accuracy, compared with 98.53% for Evo 2-7B",
        f"achieved {ancestry.loc['hEvo-large', 'classification_accuracy'] * 100:.2f}% accuracy and MCC {ancestry.loc['hEvo-large', 'classification_mcc']:.3f}, compared with {ancestry.loc['evo2_7b_base', 'classification_accuracy'] * 100:.2f}% and {ancestry.loc['evo2_7b_base', 'classification_mcc']:.3f} for Evo 2-7B",
        "Update ancestry classification values from current S34",
    )
    edit(
        49,
        "(Fig. 4b and Supplementary Table S34)",
        "(Fig. 4a and Supplementary Table S34)",
        "Current Fig4a is ancestry classification",
    )
    edit(
        49,
        "(Fig. 4c and Supplementary Table S35)",
        "(Fig. 4b and Supplementary Table S35)",
        "Current Fig4b is the few-shot ancestry comparison",
    )
    old52 = visible(paragraphs[52])
    start52 = "Across five background control resamples,"
    tail52 = old52[old52.index(start52) :]
    new52 = (
        "Across five background-control resamples, hEVO-Large achieved mean MCC 0.871 for HARs, compared with 0.870 for GPN-Star-M, a near tie. For HAQERs, hEVO-Large achieved 0.839 compared with 0.738 for GPN-Star-M, an absolute difference of 0.102. hEVO-Base achieved 0.813 and 0.680, respectively (Fig. 4c,d and Supplementary Tables S36 and S37). "
        "A separate supervised window-scoring analysis compares hEvo-large with GPN-Star-M in Fig. 4e. Its 12 examples comprise six validation and six test intervals selected for an hEvo summit inside the gold interval, no GPN summit call in the displayed 4-kb context, and a local smoothed-score margin of at least 0.15; they illustrate selected contrasts. Absence of a peak call does not imply absence of a GPN score signal. "
        f"Across the complete shared test set of {tm['n_common_haqers']} HAQERs and {tm['n_test_windows']:,} labelled windows, the three-seed mean window AUPRC was {h['window_auprc']:.3f} for hEvo and {g['window_auprc']:.3f} for GPN; AUROC was {h['window_auroc']:.3f} and {g['window_auroc']:.3f}. "
        f"Summit precision, recall and F1 were {h['summit_precision']:.3f}, {h['summit_recall']:.3f} and {h['summit_f1']:.3f} for hEvo, and {g['summit_precision']:.3f}, {g['summit_recall']:.3f} and {g['summit_f1']:.3f} for GPN. Thus hEvo had higher recall and window AUPRC, GPN had higher precision and AUROC, and summit F1 was similar."
    )
    edit(
        52,
        tail52,
        new52,
        "Update HAR/HAQER comparisons and separate selected examples from full-test performance",
    )
    a, b = hapref.loc["hEvo-base"], hapref.loc["hEvo-large"]
    edit(
        54,
        "Within this curated panel, both hEVO models favoured selected alleles more frequently than Evo 2-7B, GPN-MSA and the three GPN-Star models.",
        f"Within this curated panel, hEVO-Base favoured the selected allele at {int(a.positive_loci)}/30 loci ({a.accuracy:.3f}; 95% exact interval {a.accuracy_ci_low:.3f}–{a.accuracy_ci_high:.3f}), and hEVO-Large at {int(b.positive_loci)}/30 ({b.accuracy:.3f}; {b.accuracy_ci_low:.3f}–{b.accuracy_ci_high:.3f}). Both intervals include the 0.5 directional-reference level, so these descriptive preferences do not establish reliable detection of positive selection.",
        "Report small-sample uncertainty and directional concordance",
    )
    edit(
        54,
        "These results suggest that learning naturally occurring human haplotypes may capture aspects of allelic variation not reflected in pathogenicity prediction alone.",
        "These analyses describe score concordance at the prespecified loci and do not directly estimate evolutionary fitness or selection coefficients.",
        "Limit biological inference from allele preference",
    )
    edit(
        54,
        "Complementary Calinski-Harabasz analyses assessed the greater separation of selected and alternative allele scores in hEVO",
        "Complementary Calinski-Harabasz analyses describe separation of selected and other raw allele-score distributions. CH does not encode which allele has the higher score, so strong separation can accompany low selected-allele preference; it is interpreted alongside the directional fractions",
        "Remove unsupported hEvo CH superiority and distinguish score separation from directional agreement",
    )
    edit(
        54,
        "(Fig. 4g and Supplementary Tables S38 and S39)",
        "(Fig. 4f and Supplementary Tables S38 and S39)",
        "Current Fig4f is adaptive allele preference",
    )
    rb, rl, re, rg = [
        resource.loc[m]
        for m in ["hEvo-base", "hEvo-large", "evo2_7b_base", "Genos-1.2B"]
    ]
    new60 = (
        "We next examined predictive performance and resource use across the evaluated models. hEVO-Large contains 319,462,410 checkpoint parameters; S43 counts total predictor parameters including the task-specific probe. "
        f"Across 18 NT profiling tasks, mean MCC was {rb.mean_mcc:.3f} for hEVO-Base, {rl.mean_mcc:.3f} for hEVO-Large and {re.mean_mcc:.3f} for Evo 2-7B. "
        f"The Fig. 5e–h vertical axis instead uses equal-weight mean within-task ranks from 45 S49 task units: hEVO-Large {rl.mean_rank:.3f}, hEVO-Base {rb.mean_rank:.3f} and Evo 2-7B {re.mean_rank:.3f}; lower ranks are better and missing scores are not imputed. "
        f"On the {ncommon} tasks shared by all {inventory['models']} models, mean ranks were {shared.loc['hEvo-large', 'common_task_mean_rank']:.3f} for hEVO-Large, {shared.loc['evo2_7b_base', 'common_task_mean_rank']:.3f} for Evo 2-7B and {shared.loc['hEvo-base', 'common_task_mean_rank']:.3f} for hEVO-Base, changing the latter two models' order. NT contributes 18/45 rank units (40%), so this is task-unit weighting rather than equal weighting of biological domains. "
        f"Arithmetic-mean extraction latency across the NT tasks was {rb.latency_ms:.2f} ms per sequence for hEVO-Base, {rl.latency_ms:.2f} for hEVO-Large, {re.latency_ms:.2f} for Evo 2-7B and {rg.latency_ms:.2f} for Genos-1.2B. "
        f"The hEVO-Large/Evo 2-7B comparison corresponds to approximately {re.params_m / rl.params_m:.1f}-fold fewer predictor parameters and a {re.latency_ms / rl.latency_ms:.1f}-fold difference in recorded latency. Incremental peak allocation was {rl.vram_mb:,.0f} MiB for hEVO-Large, {rb.vram_mb:,.0f} MiB for hEVO-Base and {re.vram_mb:,.0f} MiB for Evo 2-7B. "
        "These are measurements under the retained model-specific configurations; surviving records do not certify matched hardware, numerical precision and batch size for all historical runs. They describe the observed resource footprint, with missing settings explicitly recorded, rather than a controlled hardware-matched speed comparison (Supplementary Tables S43, S44 and S49)."
    )
    edit(
        60,
        visible(paragraphs[60]),
        new60,
        "Synchronize S43 numbers, S49/common-task ranks and resource-comparison scope",
    )
    edit(
        72,
        "They provide a broadly accessible foundation for further evaluation and development.",
        "Their public training-data provenance is documented; model-weight release locations and licenses remain to be confirmed before availability can be claimed.",
        "Distinguish public training data from released weights",
    )
    edit(
        118,
        "Public resource model training and release",
        "Models trained on public genomic resources",
        "Avoid implying verified release",
    )
    edit(
        121,
        "Their model configurations and evaluation interfaces are described in Supplementary Methods S6.",
        "Their model configurations and evaluation interfaces are described in Supplementary Methods S6. The current model provenance table records unresolved release licenses; public training data alone does not establish availability of the model weights.",
        "Carry release qualification into Methods",
    )
    edit(
        124,
        "Each model retained its native tokenizer, context-length limit and pretrained output head.",
        "Each model retained its native tokenizer, context-length limit and pretrained output head. GPN-Star primary variant results use raw, uncalibrated native VEP LLRs with the prespecified task direction. A separate cLLR sensitivity analysis subtracts the checkpoint-specific neutral mean for the forward-reference central pentanucleotide and ALT allele before applying the same direction, and compares raw/cLLR on identical eligible variants. Calibration artifacts, checkpoint revisions, coverage and source hashes are recorded with the sensitivity results.",
        "Make raw primary and calibrated sensitivity protocols explicit",
    )
    edit(
        134,
        "A 919-output classification head was trained with per-label binary cross-entropy for three epochs.",
        "A 919-output classification head was trained with per-label binary cross-entropy under each recorded run configuration. The reference protocol used three epochs, whereas the currently selected hEvo-open-large S21_run_24 used one epoch and FP16 with actual pooler output; its declared configuration states mean-pooling. Historical S21_run_22 used three epochs and BF16 with mean pooling. These runs are retained separately and are not a matched training comparison.",
        "Document selected DeepSEA run and unresolved readout declaration",
    )
    edit(
        158,
        visible(paragraphs[158]),
        "For regional visualization, frozen model representations and separately trained window-scoring heads generated 512-bp-window score tracks at 64-bp intervals on hg38. The active Fig. 4e compares hEvo-large and GPN-Star-M. The heads were calibrated independently on validation negatives; the shared score cutoff 0.59 was selected on validation by maximizing the macro-average summit-hit F1 of the two models on a 0.30–0.90 grid at 0.01 steps. A common cutoff does not establish equal probability calibration. The 12 examples were selected from validation and test chromosomes using the hEvo/GPN peak-call and local-margin criteria recorded in the source manifest, while aggregate performance uses the complete shared test set. Caller parameters, chromosomes, head selection and source files are documented in Supplementary Methods S11.3 and the combined source table Fig4e_track_source_data.csv.",
        "Update current HAQER visualization and validation-only cutoff policy",
    )
    edit(
        173,
        "Pareto comparisons favoured higher predictive performance together with fewer parameters, lower latency or lower memory use; higher throughput was preferred for the throughput comparison.",
        "Fig. 5 Pareto comparisons use the lower-is-better mean within-task rank over 45 S49 units, together with fewer parameters, lower latency or lower memory use; higher throughput is preferred for the throughput comparison. The NT mean MCC is reported separately and is not the Pareto vertical axis.",
        "Correct Pareto performance axis",
    )
    edit(
        173,
        "Input length, batch size, numerical precision and device were retained with the measurement record.",
        f"S43/S44 contain {len(resource)} models across 18 NT tasks ({len(resource) * 18} model-task records), while S49 has {inventory['models']} models. Input length and available run settings are recorded in the resource companion manifest; absent device, precision or actual batch records are marked unknown. Resource contrasts are interpreted under recorded configurations rather than certified matched hardware.",
        "Expose actual resource roster and missing historical configuration metadata",
    )
    edit(
        176,
        "a genomic position with three candidate alternate alleles for the DDD within-site task",
        "a prespecified positive-variant candidate trio for the DDD within-site task (one positive and two other ALT candidates; distinct trios can share genomic coordinates)",
        "Match the retained DDD candidate IDs instead of conflating repeated positions",
    )
    edit(
        190,
        visible(paragraphs[190]),
        "Implementation and evaluation protocols are documented in the Methods and the dataset, model and evaluation provenance tables. The public training datasets are provided by their original projects. Final code-repository, model-weight and annotation-portal release addresses, release scope and model licenses have not yet been confirmed and must be supplied before publication. Controlled-cohort inputs remain subject to the access terms of their source providers.",
        "Replace placeholder URLs and unsupported availability assertions with an explicit pending-release statement",
    )
    edit(
        261,
        "CH) indices were calculated after PCA and t-SNE for ENCODE cCRE v4, GENCODE v45 and UCSC RefSeq curated annotations.",
        "CH) indices in the main panel were calculated after two-dimensional PCA for ENCODE cCRE v4, GENCODE v45 and UCSC RefSeq curated annotations, using 19 representative models. PCA and t-SNE results are reported separately in Supplementary Fig. S4 and Tables S11–S16.",
        "Align Fig2c legend with the main panel",
    )
    edit(
        263,
        "ClinVar (1), common-frequency GWAS fine-mapping (2), rare-variant GWAS fine-mapping (3), UK Biobank rare variants (4), DDD de novo variants (5) and MPRA (6)",
        "ClinVar (1), DDD de novo variants (2), common-frequency GWAS fine-mapping (3), rare-variant GWAS fine-mapping (4), UK Biobank rare variants (5) and MPRA (6)",
        "Match schematic benchmark IDs to the rendered Fig3a/b key",
    )
    edit(
        263,
        "i, Retrospective enrichment of subsequent pathogenic or likely pathogenic upgrades among the top 1% of variants initially classified as benign or likely benign, evaluated at annual ClinVar cut-offs from 2017 to 2025. j, Corresponding enrichment of subsequent downgrades among variants initially classified as pathogenic or likely pathogenic.",
        "i, Retrospective enrichment of subsequent downgrades among the top 1% of variants initially classified as pathogenic or likely pathogenic, evaluated at annual ClinVar cut-offs from 2017 to 2025. j, Corresponding enrichment of subsequent pathogenic or likely pathogenic upgrades among variants initially classified as benign or likely benign.",
        "Use the actual temporal backend order: i=downgrade, j=upgrade",
    )
    edit(
        265,
        visible(paragraphs[265]),
        "Fig. 4 | Ancestry classification, accelerated regions and allele preferences. a, Five-group classification of 1,500 1000 Genomes individuals (300 each in AFR, AMR, EAS, EUR and SAS) using fixed individual representations; MCC is shown for representative models. Unsupervised t-SNE separation is reported separately in Supplementary Fig. S8a and Table S33. b, Few-shot regression of ADMIXTURE K=5 ancestry proportions using 10–60 labelled individuals per population; the left plot shows mean Pearson r across budgets and the right plot shows the 60-per-population result with ±1 s.d. bars where reported. c,d, HAR (c) and HAQER (d) classification against five intergenic control sets; small points show replicate MCCs, large points their arithmetic mean, and ranges show observed minimum to maximum. e, hEvo-large and GPN-Star-M scores and peak calls at 12 selected HAQER intervals on a compressed hg38 axis, with known intervals, Ape divergence, ENCODE cCREs and RefSeq genes. Six validation and six test examples were selected for an hEvo summit inside the gold interval, no GPN summit call in the displayed context, local smoothed-score margin at least 0.15 and hEvo peak width at most 1.5 kb. A shared validation-selected cutoff of 0.59 is used with independent head calibration. The examples are illustrative; aggregate window and summit metrics use the complete shared test set. Nearby genes provide context and do not establish regulatory links. f, Geographic grouping and directional allele preference at 30 prespecified literature-supported loci, with 95% exact binomial intervals, alongside the one-dimensional CH index calculated from selected and other raw allele scores. Directional concordance and score separation do not directly estimate fitness effects or selection coefficients.",
        "Synchronize the caption to the current a–f Fig4 layout and displayed tracks",
    )
    # Preserve the a/b/c ablation clauses byte-for-byte, editing only e–h.
    edit(
        267,
        "e–h, Predictive performance versus computational cost across representative models. Mean MCC across 18 Nucleotide Transformer tasks is plotted against total predictor parameter count (backbone plus task-specific probe) (e), inference latency (f), throughput (g) and peak GPU memory (h). Dashed lines indicate Pareto frontiers favouring higher predictive performance together with fewer parameters, lower latency, higher throughput or lower memory use, respectively.",
        f"e–h, Cross-task mean rank versus recorded resource use. The vertical axis is the equal-weight mean within-task rank over available S49 task units (45 total; lower is better), with model coverage indicated as n/45 and missing scores not imputed. Horizontal axes show arithmetic means across 18 NT tasks for total predictor parameters (backbone plus probe) (e), extraction latency (f), nominal length-scaled throughput (g) and incremental peak GPU allocation in MiB (h). Dashed frontiers use complete rank and NT resource coverage. S49 contains {inventory['models']} models and S43 supplies resource measurements for {len(resource)}. NT accounts for 18/45 ranking units; the accompanying sensitivity table uses the {ncommon} tasks common to all models. Device, precision and batch settings are reported where recoverable; these contrasts are not certified hardware-matched comparisons.",
        "Update Pareto legend while preserving deferred ablation clauses",
    )
    # Keep the title/letter character styles separate from ordinary caption
    # prose; a whole-caption insertion would otherwise inherit title bolding.
    styled_patches = []
    for i, old, new, reason in patches:
        if i == 265:
            old_header, old_body = old.split(" a, ", 1)
            new_header, new_body = new.split(" a, ", 1)
            styled_patches.extend(
                [
                    (i, old_header, new_header, reason + " (title)"),
                    (i, old_body, new_body, reason + " (body)"),
                ]
            )
        elif i == 267:
            styled_patches.append(
                (i, old.removeprefix("e–h, "), new.removeprefix("e–h, "), reason)
            )
        else:
            styled_patches.append((i, old, new, reason))
    receipts = [
        revise(
            main,
            output / "Manuscript_to_XQY_revised_with_changes.docx",
            styled_patches,
            unchanged_paragraphs=(58, 166, 168),
        )
    ]
    # Keep the historical Evo2 track equations and protocol records explicitly
    # labelled as historical, preserving their fields/equations.
    current_track = (
        "The current Fig. 4e compares hEvo-large with GPN-Star-M using frozen representations and supervised window-scoring heads on hg38. Their tracks are stored together in Fig4e_track_source_data.csv. GPN-Star uses the official forward/reverse mean embedding over the central 512 bp of a 1,024-bp native MSA context; its head type and hard-negative variant were selected using validation window AUPRC, with train-only standardization. Windows are scanned at 64-bp steps, with chr1–12 for training, chr13–17 for validation and chr18–22 for testing. The primary displayed seed is 20260515; test summaries use seeds 20260515–20260517. "
        "Each head is calibrated independently on validation negatives. A shared score cutoff of 0.59 was selected on a 0.30–0.90 grid in 0.01 steps by maximizing macro-averaged validation summit-hit F1, breaking ties by macro recall and then fewer combined peaks. hEvo uses Gaussian sigma 128 bp, prominence 0.10, minimum distance 1,024 bp and minimum width 64 bp; GPN uses sigma 256 bp, prominence 0.15, distance 1,024 bp and width 256 bp. Both Gaussian kernels use truncate 8. "
        "The 12 displayed intervals comprise six validation and six test examples, selected for an hEvo summit inside the gold interval, no GPN summit within the displayed ±2 kb, local smoothed-score margin at least 0.15 and hEvo peak width at most 1.5 kb. HAQER0218 provides COL4A2-near context, without a regulatory-link claim. A peak hit is a called summit inside the original half-open HAQER interval. Independently calibrated heads need not have identical probability calibration. The source table reports the complete shared test set of 214 HAQERs and 1,216 labelled windows separately. The complete Fig. 4e track source data and annotations are included in the repository."
    )
    for source in sorted((root / "article").glob("*Supplementary_Methods*docx")):
        with zipfile.ZipFile(source) as z:
            r = ET.fromstring(z.read("word/document.xml"))
        ps = r.findall(".//w:p", NS)
        edits = []
        for i, p in enumerate(ps):
            t = visible(p)
            if t.startswith("Table S44 retained model-by-task records"):
                new = t.replace(
                    "latency, throughput and memory were summarized by the median of measured values",
                    "latency, throughput and memory in S43 were each summarized by the arithmetic mean across the 18 NT tasks",
                )
                new = new.replace(
                    "29 models × 18 tasks (522 combinations)",
                    f"{len(resource)} models × 18 tasks ({len(resource) * 18} combinations)",
                )
                new += f" The current Pareto vertical axis uses mean within-task rank over 45 S49 task units, with missing scores not imputed; it is distinct from NT mean MCC. S49 contains {inventory['models']} models, and the shared-task sensitivity comparison uses {ncommon} tasks. Resource provenance identifies recovered configurations and marks missing device, dtype or actual batch settings as unknown; hardware-matched performance is not asserted."
                edits.append(
                    (
                        i,
                        t,
                        new,
                        "Align resource means, model counts, rank axis and coverage",
                    )
                )
            if t.startswith("Regional score tracks were generated"):
                edits.append(
                    (
                        i,
                        t,
                        current_track,
                        "Document the current hEvo/GPN Fig4e workflow; retain old equations as historical",
                    )
                )
            if t.startswith("For HAQER scoring, frozen hEvo-large"):
                edits.append(
                    (
                        i,
                        "For HAQER scoring,",
                        "For the retained historical Evo2-based HAQER workflow,",
                        "Label historical head specifications",
                    )
                )
            if t.startswith("For HAQER inference, validation-negative"):
                edits.append(
                    (
                        i,
                        "For HAQER inference,",
                        "For inference in the retained historical Evo2-based workflow,",
                        "Label historical calibrations and peak caller settings",
                    )
                )
            if t.startswith("The HAQER panel shows a fixed set"):
                edits.append(
                    (
                        i,
                        "The HAQER panel",
                        "The historical Evo2-based HAQER example panel, not used in current Fig. 4e,",
                        "Remove historical examples from the active figure description",
                    )
                )
            if t.startswith("Training used AdamW") and "three epochs and seed 42" in t:
                edits.append(
                    (
                        i,
                        "three epochs and seed 42.",
                        "three epochs and seed 42 in the reference protocol. The selected hEvo-open-large S21_run_24 instead used one FP16 epoch and actual pooler output, whereas its declaration says mean-pooling; S21_run_22 used three BF16 epochs and mean pooling. Historical and selected runs are retained separately and are not matched training comparisons.",
                        "Document selected public-large DeepSEA run",
                    )
                )
            if (
                "GPN-Star" in t
                and "native" in t.lower()
                and ("VEP" in t or "1,024-bp MSA" in t or "1,024 bp MSA" in t)
            ):
                edits.append(
                    (
                        i,
                        t,
                        t
                        + " GPN-Star primary scores are raw, uncalibrated native VEP LLRs. Separate cLLR sensitivity uses the matching checkpoint's published forward-reference pentanucleotide/ALT neutral mean, preserving task directions and comparing identical eligible variants; calibration and coverage manifests accompany the results.",
                        "Clarify raw primary vs calibrated sensitivity",
                    )
                )
        receipts.append(
            revise(source, output / (source.stem + "_revised_with_changes.docx"), edits)
        )
    (output / "document_revision_receipts.json").write_text(
        json.dumps(receipts, ensure_ascii=False, indent=2)
    )
    print(
        json.dumps(
            {
                "documents": len(receipts),
                "tracked_text_edits": sum(len(r["changes"]) for r in receipts),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    run(args.repo_root.resolve(), args.output_dir.resolve())
