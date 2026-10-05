"""Écriture d'un modèle à arbres pour un chargement économe en mémoire.

Le modèle `/recommend` est une forêt de 150 arbres (ExtraTrees). Quand `joblib.load` relit un fichier
écrit par `joblib.dump`, chaque arbre est reconstruit à partir d'un dictionnaire d'état (`nodes`,
`values`), que scikit-learn recopie dans ses propres tableaux. Mais le lecteur pickle garde chacun de ces
dictionnaires dans son « mémo » jusqu'à la fin du fichier : les tableaux des arbres existent alors deux
fois en mémoire, et le pic au démarrage de l'API dépasse largement la mémoire utilisée ensuite.

`dump_without_tree_state_memo` écrit le même objet, avec la même compression que
`joblib.dump(model, path, compress=("lzma", 3))`, mais sans mettre ces dictionnaires dans le mémo :
chacun est libéré dès que son arbre est reconstruit. Le fichier se relit avec le `joblib.load` habituel,
sans code spécifique dans l'API, et le modèle relu est le même (mêmes arbres, mêmes prédictions).

Dépendance à une API interne de joblib : `joblib.numpy_pickle.NumpyPickler` ne fait pas partie de l'API
publique documentée de joblib. Elle sert uniquement ici, à l'écriture des artefacts par le script de
reconstruction, jamais au chargement par l'API. `tests/agritech/test_serialization.py` vérifie ce
comportement avec la version de joblib du projet : une montée de version qui le casserait fait échouer
ces tests.
"""

from __future__ import annotations

import lzma
from pathlib import Path
from typing import Any

from joblib.numpy_pickle import NumpyPickler  # API interne de joblib, voir la docstring du module

# Même compression que `joblib.dump(..., compress=("lzma", 3))` : format lzma « alone », niveau 3.
LZMA_FORMAT = lzma.FORMAT_ALONE
LZMA_PRESET = 3


def dump_without_tree_state_memo(model: Any, path: Path) -> None:
    """Écrit `model` dans `path` comme `joblib.dump(..., compress=("lzma", 3))`, sans mémo des arbres.

    Le fichier se relit avec `joblib.load`. Seule différence avec `joblib.dump` : l'état des arbres
    scikit-learn n'est pas gardé en mémoire pendant toute la lecture (voir la docstring du module).
    """
    with lzma.open(path, "wb", format=LZMA_FORMAT, preset=LZMA_PRESET) as file:
        _TreeStateNoMemoPickler(file).dump(model)


class _TreeStateNoMemoPickler(NumpyPickler):
    """Pickler de joblib qui ne met pas l'état des arbres scikit-learn dans le mémo.

    Le mémo permet d'écrire une simple référence quand un même objet apparaît plusieurs fois. L'état d'un
    arbre (le dictionnaire de `Tree.__getstate__`) est recréé à chaque sérialisation et n'apparaît qu'une
    fois : le laisser hors du mémo ne change pas l'objet relu.
    """

    def memoize(self, obj: Any) -> None:
        if _is_tree_state(obj):
            return
        super().memoize(obj)


def _is_tree_state(obj: Any) -> bool:
    """Vrai pour l'état d'un arbre scikit-learn : un dictionnaire avec les clés `nodes` et `values`."""
    return isinstance(obj, dict) and "nodes" in obj and "values" in obj
