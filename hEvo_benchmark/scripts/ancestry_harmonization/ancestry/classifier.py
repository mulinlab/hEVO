"""Frozen S1-six-PC kNN classifier and sigmoid calibration."""
from dataclasses import dataclass
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .inputs import CLASSES, FOUR, FIVE


def estimator():
    return make_pipeline(StandardScaler(), KNeighborsClassifier(
        n_neighbors=15, weights="distance", n_jobs=1))


class ReferenceDistance:
    """Fixed population envelope; serialized class name stays compatible."""
    def __init__(self, method="population_envelope", neighbors=5):
        if method != "population_envelope" or neighbors != 5:
            raise ValueError("Only the final population envelope is supported")
        self.method, self.neighbors = method, neighbors

    def fit(self, x, y, quantile=0.999, populations=None):
        if quantile != 0.999 or populations is None or len(populations) != len(y):
            raise ValueError("Final envelope requires population labels and quantile 0.999")
        self.scale = StandardScaler().fit(x)
        z = self.scale.transform(x)
        self.models, self.cutoffs, self.population_calibration = {}, {}, {}
        for label in CLASSES:
            sub = z[np.asarray(y) == label]
            if len(sub) < 3:
                raise ValueError("Insufficient reference samples: " + label)
            nn = NearestNeighbors(n_neighbors=min(5, len(sub) - 1) + 1, n_jobs=1).fit(sub)
            dd, _ = nn.kneighbors(sub)
            self.models[label] = nn
            self.cutoffs[label] = max(float(np.quantile(dd[:, 1], quantile)), 1e-12)
            pop = np.asarray(populations)[np.asarray(y) == label]
            groups = np.unique(pop)
            used = label in FOUR and len(groups) >= 3
            if used:
                held = []
                for group in groups:
                    keep = pop != group
                    model = NearestNeighbors(n_neighbors=1, n_jobs=1).fit(sub[keep])
                    held.extend(model.kneighbors(sub[~keep])[0][:, 0])
                self.cutoffs[label] = max(self.cutoffs[label], float(np.quantile(held, quantile)))
            self.population_calibration[label] = {"populations": len(groups), "used": used}
        return self

    def query(self, x, pred):
        z = self.scale.transform(x)
        dist, cut = np.empty(len(x)), np.empty(len(x))
        for label in CLASSES:
            idx = np.flatnonzero(np.asarray(pred) == label)
            if len(idx):
                dist[idx] = self.models[label].kneighbors(z[idx], n_neighbors=1)[0][:, 0]
                cut[idx] = self.cutoffs[label]
        return dist, cut

@dataclass(frozen=True)
class Spec:
    family: str
    param: str
    pcs: int

    @property
    def key(self):
        return f"{self.family}_{self.param}_pc{self.pcs}"

def probabilities(model, x):
    p = model.predict_proba(x)
    out = np.zeros((len(x), len(CLASSES)))
    for j, label in enumerate(model.classes_):
        out[:, CLASSES.index(label)] = p[:, j]
    return out

class SigmoidCalibration:
    def fit(self, p, y):
        self.models = []
        y = np.asarray(y)
        for i, label in enumerate(CLASSES):
            target = (y == label).astype(int)
            if len(np.unique(target)) != 2:
                raise ValueError(f"Cannot calibrate unrepresented class {label}")
            m = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            m.fit(self.logit(p[:, i])[:, None], target)
            self.models.append(m)
        return self

    @staticmethod
    def logit(p):
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def transform(self, p):
        out = np.column_stack(
            [
                m.predict_proba(self.logit(p[:, i])[:, None])[:, 1]
                for i, m in enumerate(self.models)
            ]
        )
        return out / out.sum(axis=1, keepdims=True)

def assign(p, dist, cut, threshold, margin, quality=None):
    p = np.asarray(p)
    if p.ndim != 2 or p.shape[1] != len(CLASSES) or not np.isfinite(p).all():
        raise ValueError("Invalid probability matrix")
    if not np.allclose(p.sum(axis=1), 1, atol=1e-6):
        raise ValueError("Probabilities do not sum to one")
    if (
        not np.isfinite(dist).all()
        or not np.isfinite(cut).all()
        or np.any(np.asarray(cut) <= 0)
    ):
        raise ValueError("Invalid reference distance")
    raw = np.asarray(CLASSES)[p.argmax(axis=1)]
    ordered = np.sort(p, axis=1)
    maxp = ordered[:, -1]
    gap = ordered[:, -1] - ordered[:, -2]
    low = (maxp < threshold) | (gap < margin)
    outside = np.asarray(dist) > np.asarray(cut)
    bad = np.zeros(len(raw), bool) if quality is None else np.asarray(quality, bool)
    accepted = ~(low | outside | bad)
    assigned = np.where(accepted & np.isin(raw, FOUR), raw, "Others")
    reason = np.where(
        bad,
        "input_qc_failed",
        np.where(
            outside,
            "outside_reference",
            np.where(
                low,
                "low_confidence",
                np.where(
                    np.isin(raw, FOUR), "accepted", np.char.add("reference_", raw)
                ),
            ),
        ),
    )
    return {
        "raw": raw,
        "assigned": assigned,
        "reason": reason,
        "maxp": maxp,
        "margin": gap,
        "low_confidence": low,
        "outside_reference": outside,
        "input_qc_failed": bad,
        "accepted": accepted,
    }

def metrics(y, result):
    y = np.asarray(y)
    assigned = result["assigned"]
    out = {}
    for label in FOUR:
        tp = int(np.sum((y == label) & (assigned == label)))
        n_pred = int(np.sum(assigned == label))
        n_true = int(np.sum(y == label))
        out[label] = {
            "tp": tp,
            "predicted": n_pred,
            "support": n_true,
            "precision": tp / n_pred if n_pred else 0.0,
            "accepted_correct_fraction": tp / n_true if n_true else 0.0,
        }
    return out
