"""Écriture d'un modèle à arbres sans mémo de l'état des arbres (`agritech.serialization`).

Ces tests verrouillent, avec la version de joblib du projet, le comportement qui évite le pic mémoire au
chargement du modèle `/recommend` : le fichier se relit avec `joblib.load`, le modèle relu est identique,
et l'état des arbres ne reste pas dans le mémo du lecteur pickle. `NumpyPickler` et `NumpyUnpickler` sont
des classes internes de joblib : si une montée de version change leur fonctionnement, ces tests échouent.
"""

from __future__ import annotations

import lzma
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from joblib.numpy_pickle import NumpyUnpickler
from sklearn.ensemble import ExtraTreesRegressor

from agritech.modeling import experiment_pipeline
from agritech.recommend_config import RECOMMEND_CROPS, RECOMMEND_FINAL_PARAMS
from agritech.serialization import dump_without_tree_state_memo

N_TREES = RECOMMEND_FINAL_PARAMS["n_estimators"]


@pytest.fixture(scope="module")
def forest_pipeline() -> tuple:
    """One-hot + ExtraTrees avec les réglages finaux de `/recommend`, appris sur 300 lignes synthétiques."""
    rng = np.random.default_rng(0)
    rows = 300
    X = pd.DataFrame(
        {
            "crop": rng.choice(RECOMMEND_CROPS, rows),
            "rain_mm": rng.uniform(100, 3000, rows),
            "temp_hist": rng.uniform(5, 30, rows),
        }
    )
    y = rng.uniform(1, 40, rows)
    pipeline = experiment_pipeline(ExtraTreesRegressor(**RECOMMEND_FINAL_PARAMS), ["crop"], ["rain_mm", "temp_hist"])
    return pipeline.fit(X, y), X


@pytest.fixture
def written(forest_pipeline, tmp_path: Path) -> Path:
    path = tmp_path / "model.joblib"
    dump_without_tree_state_memo(forest_pipeline[0], path)
    return path


def _tree_states(pipeline) -> list[dict]:
    return [estimator.tree_.__getstate__() for estimator in pipeline[-1].estimators_]


def _tree_states_left_in_memo(path: Path) -> int:
    """Relit `path` avec le lecteur de joblib et compte les états d'arbres encore dans son mémo à la fin."""
    with lzma.open(path) as file:
        unpickler = NumpyUnpickler(str(path), file, ensure_native_byte_order=False)
        unpickler.load()
    return sum(isinstance(value, dict) and "nodes" in value for value in unpickler.memo.values())


def test_written_model_reloads_with_joblib_load_and_gives_the_same_predictions(forest_pipeline, written: Path):
    pipeline, X = forest_pipeline

    reloaded = joblib.load(written)

    assert np.array_equal(reloaded.predict(X), pipeline.predict(X))
    assert reloaded[-1].get_params() == pipeline[-1].get_params()


def test_written_trees_are_identical_bit_for_bit(forest_pipeline, written: Path):
    """Chaque champ des nœuds et chaque valeur des feuilles, au bit près.

    Les 7 octets de remplissage de la structure `Node` de scikit-learn ne sont pas comparés : ils ne sont
    jamais lus et leur contenu dépend de la mémoire réutilisée pendant le chargement.
    """
    before, after = _tree_states(forest_pipeline[0]), _tree_states(joblib.load(written))

    assert len(after) == len(before) == N_TREES
    for state_before, state_after in zip(before, after):
        assert (state_after["node_count"], state_after["max_depth"]) == (
            state_before["node_count"],
            state_before["max_depth"],
        )
        for field in state_before["nodes"].dtype.names:
            assert state_after["nodes"][field].tobytes() == state_before["nodes"][field].tobytes()
        assert state_after["values"].tobytes() == state_before["values"].tobytes()


def test_tree_states_are_not_kept_in_the_pickle_memo(forest_pipeline, written: Path, tmp_path: Path):
    """`joblib.dump` garde un état par arbre dans le mémo jusqu'à la fin de la lecture ; la fonction, aucun."""
    standard = tmp_path / "standard.joblib"
    joblib.dump(forest_pipeline[0], standard, compress=("lzma", 3))

    assert _tree_states_left_in_memo(standard) == N_TREES
    assert _tree_states_left_in_memo(written) == 0


def test_compression_is_the_lzma_of_joblib_dump(forest_pipeline, written: Path, tmp_path: Path):
    """Même en-tête lzma (format « alone », niveau 3) que `joblib.dump(..., compress=("lzma", 3))`."""
    standard = tmp_path / "standard.joblib"
    joblib.dump(forest_pipeline[0], standard, compress=("lzma", 3))

    assert written.read_bytes()[:13] == standard.read_bytes()[:13]
