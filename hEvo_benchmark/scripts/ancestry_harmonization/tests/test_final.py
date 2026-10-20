"""Integrity and rejection checks for the single maintained annotation protocol."""
from pathlib import Path
import sys
import json
import shutil
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ancestry import final
from ancestry.classifier import probabilities, assign


def test_model_is_final():
    a, p = final.load_model()
    assert p['pcs'] == a['spec'].pcs == 6
    assert len(a['reference_ids']) == 2791
    assert 'sensitivity_models' not in a
    assert a['model'].named_steps['kneighborsclassifier'].n_neighbors == 15


def test_extreme_coordinate_rejected():
    a, _ = final.load_model()
    x = np.full((1, 6), 1e6)
    prob = a['calibrator'].transform(probabilities(a['model'], x))
    pred = np.array(final.CLASSES)[prob.argmax(axis=1)]
    distance, cutoff = a['distance'].query(x, pred)
    r = assign(prob, distance, cutoff, .8, .3)
    assert r['assigned'][0] == 'Others'
    assert r['outside_reference'][0]


def test_coordinate_identity_and_validity(tmp_path):
    f = tmp_path / 'pcs.csv'
    data = pd.DataFrame({'IID': ['a', 'b'], **{f'PC{i}': [float(i), -float(i)] for i in range(1, 7)}})
    data.to_csv(f, index=False)
    x = final.coordinates(f, ['b', 'a'])
    assert x[0, 0] == -1
    with pytest.raises(ValueError): final.coordinates(f, ['missing'])
    with pytest.raises(ValueError): final.coordinates(f, ['a', 'a'])
    data.loc[1, 'IID'] = 'a';data.to_csv(f, index=False)
    with pytest.raises(ValueError): final.coordinates(f, ['a'])
    data.loc[1, 'IID'] = 'b';data.loc[1, 'PC2'] = np.inf;data.to_csv(f, index=False)
    with pytest.raises(ValueError): final.coordinates(f, ['a'])


def test_resource_change_stops_loading(tmp_path, monkeypatch):
    source = final.ASSETS
    shutil.copytree(source, tmp_path / 'assets')
    (tmp_path / 'assets/reference.tsv').write_text('changed')
    monkeypatch.setattr(final, 'ASSETS', tmp_path / 'assets')
    with pytest.raises(ValueError, match='checksum'): final.load_model()
