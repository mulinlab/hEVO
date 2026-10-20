"""Version-checked six-PC prediction and fixed-protocol training."""
import importlib.metadata
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from .classifier import Spec, SigmoidCalibration, ReferenceDistance, estimator, probabilities, assign
from .inputs import CLASSES, sha256, read_tsv, write_tsv

ASSETS = Path(__file__).resolve().parents[1] / "resources/final_6pc"


def load_model():
    protocol = json.loads((ASSETS / "protocol.json").read_text())
    for name, expected in protocol["files"].items():
        if sha256(ASSETS / name) != expected:
            raise ValueError("Final resource checksum mismatch: " + name)
    for package, version in protocol["software"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError(f"{package} must be {version}; use env/environment.yaml")
    with (ASSETS / "classifier.pkl").open("rb") as handle:
        artifact = pickle.load(handle)
    if (artifact["spec"].family, artifact["spec"].param, artifact["spec"].pcs) != ("knn", "15", 6):
        raise ValueError("Model is not the frozen six-PC kNN15")
    if artifact["threshold"]["probability"] != 0.8 or artifact["threshold"]["margin"] != 0.3:
        raise ValueError("Final confidence thresholds changed")
    if artifact["distance_spec"] != {"method": "population_envelope", "quantile": 0.999, "neighbors": 5}:
        raise ValueError("Final distance protocol changed")
    return artifact, protocol


def coordinates(path, ids):
    cols = [f"PC{i}" for i in range(1, 7)]
    frame = pd.read_csv(path, dtype={"IID": str}, usecols=["IID", *cols])
    if frame.IID.isna().any() or frame.IID.str.strip().eq("").any() or frame.IID.duplicated().any():
        raise ValueError("Blank/duplicate S1 sample IDs")
    if not np.isfinite(frame[cols].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite S1 PCs")
    frame = frame.set_index("IID")
    if len(ids) != len(set(ids)) or not set(ids) <= set(frame.index):
        raise ValueError("Duplicate target IDs or incomplete S1 coverage")
    return frame.loc[ids, cols].to_numpy(float)


def targets(path):
    frame = pd.read_csv(path, dtype={"IID": str, "Group": str})
    if len(frame) != 63809 or frame.IID.isna().any() or frame.IID.duplicated().any():
        raise ValueError("Expected exactly 63,809 unique target IDs")
    if int(frame.Group.eq("EAS").sum()) != 3188:
        raise ValueError("Internal Chinese cohort does not have 3,188 samples")
    if frame.Group.isna().any() or set(frame.Group) != {"AFR", "EAS", "HGDP", "1kgenome", "UKB", "UK10K", "TopMed", "SG10K"}:
        raise ValueError("Invalid source cohorts")
    return frame


def predict(c, dry_run=False):
    artifact, protocol = load_model()
    target = targets(c["inputs"]["targets"])
    x = coordinates(c["inputs"]["s1"], target.IID.tolist())
    if dry_run:
        return {"samples": len(target), "version": protocol["version"], "pcs": 6}
    dest = Path(c["output_dir"])
    if dest.exists():
        raise FileExistsError("Use a new output directory")
    p = artifact["calibrator"].transform(probabilities(artifact["model"], x))
    raw = np.asarray(CLASSES)[p.argmax(axis=1)]
    distance, cutoff = artifact["distance"].query(x, raw)
    result = assign(p, distance, cutoff, 0.8, 0.3)
    output = []
    for i, row in enumerate(target.to_dict("records")):
        manual = row["Group"] == "EAS"
        record = dict(sample=row["IID"], cohort=row["Group"],
                      assigned_label="EAS" if manual else str(result["assigned"][i]),
                      label_basis="user_confirmed_cohort" if manual else "S1_6PC_round2",
                      assignment_reason="user_confirmed_cohort" if manual else str(result["reason"][i]),
                      effective_rejected_flag=int(not manual and not result["accepted"][i]),
                      model_assigned_label=str(result["assigned"][i]),
                      raw_predicted_label=str(raw[i]),
                      max_probability="" if manual else float(result["maxp"][i]),
                      model_max_probability=float(result["maxp"][i]),
                      probability_margin="" if manual else float(result["margin"][i]),
                      model_probability_margin=float(result["margin"][i]),
                      model_low_confidence_flag=int(result["low_confidence"][i]),
                      model_outside_reference_flag=int(result["outside_reference"][i]),
                      distance=float(distance[i]), distance_cutoff=float(cutoff[i]),
                      input_qc_status="not_provided_by_S1", model_version=protocol["version"])
        record.update({"P_" + label: float(p[i, j]) for j, label in enumerate(CLASSES)})
        output.append(record)
    dest.mkdir(parents=True)
    write_tsv(dest / "sample_ancestry.tsv", output)
    fields = ["sample", "assigned_label", "label_basis", "assignment_reason", "model_assigned_label"]
    write_tsv(dest / "sample_popu.stats_63809.tsv", [{k: r[k] for k in fields} for r in output])
    return {"samples": len(output), "output_dir": str(dest), "version": protocol["version"]}


def check(c):
    _, protocol = load_model()
    target = targets(c["inputs"]["targets"])
    coords = coordinates(c["inputs"]["s1"], target.IID.tolist())
    result = {"samples": len(coords), "unique_ids": len(set(target.IID)), "version": protocol["version"]}
    path = Path(c["output_dir"]) / "sample_ancestry.tsv"
    if path.exists():
        out = pd.read_csv(path, sep="\t", dtype={"sample": str})
        if out["sample"].tolist() != target.IID.tolist():
            raise ValueError("Output order differs from target")
        if not out.loc[out.cohort.eq("EAS"), "assigned_label"].eq("EAS").all():
            raise ValueError("Chinese cohort annotation mismatch")
        if "Ancestry" in target and not out.assigned_label.eq(target.Ancestry).all():
            raise ValueError("Output differs from current S2 annotations")
        result["output_matches_current_table"] = True
    return result


def train(c, dry_run=False):
    """Refit only the final protocol using its frozen population-grouped folds."""
    load_model()
    rows = read_tsv(ASSETS / "reference.tsv")
    ids = [row["sample"] for row in rows]
    x = coordinates(c["inputs"]["s1"], ids)
    y = np.asarray([r["class"] for r in rows])
    folds = {r["sample"]: int(r["fold"]) for r in read_tsv(ASSETS / "calibration_folds.tsv")}
    if set(folds) != set(ids):
        raise ValueError("Calibration fold IDs differ from references")
    if dry_run:
        return {"reference_samples": len(ids), "folds": len(set(folds.values())), "pcs": 6}
    dest = Path(c["training_output_dir"])
    if dest.exists():
        raise FileExistsError("Use a new training output directory")
    oof = np.full((len(ids), len(CLASSES)), np.nan)
    fold_ids = np.array([folds[s] for s in ids])
    for fold in sorted(set(fold_ids)):
        test = fold_ids == fold
        for key in ["population", "kinship_group"]:
            if {r[key] for r, keep in zip(rows, test) if keep} & {r[key] for r, keep in zip(rows, test) if not keep}:
                raise ValueError("Grouped calibration split leakage: " + key)
        model = estimator().fit(x[~test], y[~test])
        oof[test] = probabilities(model, x[test])
    artifact = dict(spec=Spec("knn", "15", 6), model=estimator().fit(x, y),
                    calibrator=SigmoidCalibration().fit(oof, y),
                    distance=ReferenceDistance().fit(x, y, 0.999, [r["population"] for r in rows]),
                    reference_ids=ids, threshold={"probability": 0.8, "margin": 0.3},
                    distance_spec={"method": "population_envelope", "quantile": 0.999, "neighbors": 5})
    dest.mkdir(parents=True)
    with (dest / "classifier.pkl").open("wb") as handle:
        pickle.dump(artifact, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return {"training_output_dir": str(dest), "reference_samples": len(ids), "published_model_replaced": False}
