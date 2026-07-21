"""Validación cruzada temporal robusta para datasets desbalanceados.

`SafeTimeSeriesSplit` envuelve a `sklearn.model_selection.TimeSeriesSplit`
y fusiona folds consecutivos cuando el fold de validación no contiene
suficiente representación de la clase minoritaria. Esto evita métricas
NaN o degeneradas durante la búsqueda Bayesiana de hiperparámetros.

Uso típico:
    from src.cv import SafeTimeSeriesSplit
    cv = SafeTimeSeriesSplit(n_splits=5, min_minority_samples=2)
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=...)

Justificación metodológica:
    El bloqueo temporal se apoya principalmente en Cerqueira, Torgo & Mozetič
    (2020, "Evaluating Time Series Forecasting Models", Machine Learning), que
    muestra empíricamente la robustez del holdout secuencial en regímenes de
    muestra pequeña. Se cita defensivamente a Bergmeir & Benítez (2012, "On the
    use of cross-validation for time series predictor evaluation", Information
    Sciences): ese trabajo argumenta que el K-fold estándar PUEDE ser válido —e
    incluso preferible— en series estacionarias autoregresivas; aquí adoptamos
    el split temporal estricto como postura CONSERVADORA anti-leakage dado el
    horizonte de decisión t=0, NO como prohibición universal de K-fold.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.metrics import make_scorer, recall_score
from sklearn.model_selection import TimeSeriesSplit


class SafeTimeSeriesSplit:
    """`TimeSeriesSplit` con densidad mínima garantizada para la clase minoritaria.

    Cuando un fold de validación no contiene al menos `min_minority_samples`
    muestras de la clase positiva, el fold se acumula (`pending_val`) y se
    fusiona con el siguiente. El fold final emite un `UserWarning` cuando
    cierra con menos de la cuota.
    """

    def __init__(self, n_splits: int = 5, min_minority_samples: int = 2):
        self.n_splits = n_splits
        self.min_minority_samples = min_minority_samples
        self.minority_class = 1  # Clase de interés en T1: Crítico

    def split(self, X, y, groups=None):
        y_arr = np.asarray(y)
        base = TimeSeriesSplit(n_splits=self.n_splits)
        folds_emitted = 0
        pending_val = np.array([], dtype=int)
        train_idx = None  # Inicialización para el fold residual

        for train_idx, val_idx in base.split(X):
            combined_val = np.concatenate([pending_val, val_idx]).astype(int)
            minority_count = int((y_arr[combined_val] == self.minority_class).sum())

            if minority_count >= self.min_minority_samples:
                yield train_idx, combined_val
                pending_val = np.array([], dtype=int)
                folds_emitted += 1
            else:
                pending_val = combined_val

        # Residuo final: si quedan índices sin emitir, los liberamos con warning
        if len(pending_val) > 0 and folds_emitted < self.n_splits and train_idx is not None:
            n_min = int((y_arr[pending_val] == self.minority_class).sum())
            warnings.warn(
                f'Fold final con solo {n_min} muestras de la clase minoritaria. '
                f'La significancia estadística de este fold es baja.'
            )
            yield train_idx, pending_val

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits


# Scorer legacy: Recall puro de la clase 1 (Crítico).
# Conservado para compatibilidad con notebooks históricos; los optimizadores
# nuevos usan `scorer_operativo` definido en `src.eval`.
critical_recall = make_scorer(recall_score, pos_label=1, zero_division=0)
