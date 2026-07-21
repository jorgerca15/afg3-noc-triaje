"""Calibración isotónica via OOF temporal — helper compartido.

Originalmente embebido en notebooks/modelo_v4.ipynb (celda 3) y v5. Extraído a
src/ para que los scripts del Avance Final V2 (T12-T14) puedan persistir las
predicciones OOF del training set, necesarias para el barrido del peso α del
ensamble HÍBRIDO.

Compatibilidad: la firma original `(estimator, X_tr, y_tr, X_te, n_splits=5)`
sigue funcionando idéntica a la del notebook. El nuevo flag `return_oof_train`
añade un cuarto elemento a la tupla retornada con la predicción OOF del train.
"""

from __future__ import annotations

import numpy as np
from sklearn.base import clone
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.cv import SafeTimeSeriesSplit


def calibrar_isotonica(
    estimator,
    X_train_,
    y_train_,
    X_test_,
    n_splits: int = 5,
    return_oof_train: bool = False,
):
    """Entrena `estimator` y calibra una capa isotónica vía OOF temporal.

    Args:
        estimator: clasificador sklearn-compatible con `predict_proba`.
        X_train_: matriz de features de train (DataFrame, array denso o csr).
        y_train_: target binario de train (Series o array).
        X_test_: matriz de features de test, mismo schema que train.
        n_splits: folds de SafeTimeSeriesSplit (default 5).
        return_oof_train: si True, además devuelve la predicción OOF del train
            (tamaño len(y_train_), con NaN en posiciones del primer fold que el
            partidor no validó). Necesario para barrido α del ensamble HÍBRIDO.

    Returns:
        Si `return_oof_train=False`: (proba_test_raw, proba_test_cal, estimator_fitted)
        Si `return_oof_train=True`:  (proba_test_raw, proba_test_cal, estimator_fitted, proba_oof_train)
    """
    cv = SafeTimeSeriesSplit(n_splits=n_splits, min_minority_samples=2)
    proba_oof = np.full(len(y_train_), np.nan, dtype=float)

    for tr_idx, va_idx in cv.split(X_train_, y_train_):
        clf_clone = clone(estimator)
        if hasattr(X_train_, "iloc"):
            X_tr = X_train_.iloc[tr_idx]
            X_va = X_train_.iloc[va_idx]
        else:
            X_tr = X_train_[tr_idx]
            X_va = X_train_[va_idx]
        y_tr = y_train_.iloc[tr_idx] if hasattr(y_train_, "iloc") else y_train_[tr_idx]
        clf_clone.fit(X_tr, y_tr)
        proba_oof[va_idx] = clf_clone.predict_proba(X_va)[:, 1]

    mask_valid = ~np.isnan(proba_oof)
    if mask_valid.sum() < 30:
        # Fallback: pocos OOF válidos → calibración trivial (raw == cal)
        estimator.fit(X_train_, y_train_)
        proba_raw = estimator.predict_proba(X_test_)[:, 1]
        if return_oof_train:
            return proba_raw, proba_raw, estimator, proba_oof
        return proba_raw, proba_raw, estimator

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    y_arr = y_train_.values if hasattr(y_train_, "values") else np.asarray(y_train_)
    iso.fit(proba_oof[mask_valid], y_arr[mask_valid])

    estimator.fit(X_train_, y_train_)
    proba_test_raw = estimator.predict_proba(X_test_)[:, 1]
    proba_test_cal = iso.transform(proba_test_raw)

    if return_oof_train:
        # Devolvemos también la versión OOF calibrada del train para el barrido α
        proba_oof_cal = np.full_like(proba_oof, np.nan)
        proba_oof_cal[mask_valid] = iso.transform(proba_oof[mask_valid])
        return proba_test_raw, proba_test_cal, estimator, proba_oof_cal

    return proba_test_raw, proba_test_cal, estimator


def calibrar_platt(
    estimator,
    X_train_,
    y_train_,
    X_test_,
    n_splits: int = 5,
    return_oof_train: bool = False,
):
    """Calibración de Platt (logística 1D) vía OOF temporal — sibling paramétrico.

    Misma mecánica y contrato de retorno que `calibrar_isotonica`, pero la capa
    de calibración es una `LogisticRegression` unidimensional (escalamiento de
    Platt) en vez de isotónica. Motivación (revisión AFG III): con ~18-20
    positivos totales la isotónica no-paramétrica sobreajusta en mesetas/saltos
    abruptos y su mordida real es el threshold de piso de precisión. Platt impone
    suavidad monótona y es más robusta en régimen de positivos escasos. Se usa
    como **eje de sensibilidad** (correr cada config bajo isotónica Y Platt para
    verificar que el veredicto sea robusto al método de calibración).

    Cita: Platt 1999 "Probabilistic Outputs for Support Vector Machines";
    Niculescu-Mizil & Caruana 2005 (comparación isotónica vs sigmoide).

    Returns:
        Si `return_oof_train=False`: (proba_test_raw, proba_test_cal, estimator_fitted)
        Si `return_oof_train=True`:  (proba_test_raw, proba_test_cal, estimator_fitted, proba_oof_train)
    """
    cv = SafeTimeSeriesSplit(n_splits=n_splits, min_minority_samples=2)
    proba_oof = np.full(len(y_train_), np.nan, dtype=float)

    for tr_idx, va_idx in cv.split(X_train_, y_train_):
        clf_clone = clone(estimator)
        if hasattr(X_train_, "iloc"):
            X_tr = X_train_.iloc[tr_idx]
            X_va = X_train_.iloc[va_idx]
        else:
            X_tr = X_train_[tr_idx]
            X_va = X_train_[va_idx]
        y_tr = y_train_.iloc[tr_idx] if hasattr(y_train_, "iloc") else y_train_[tr_idx]
        clf_clone.fit(X_tr, y_tr)
        proba_oof[va_idx] = clf_clone.predict_proba(X_va)[:, 1]

    mask_valid = ~np.isnan(proba_oof)
    y_arr = y_train_.values if hasattr(y_train_, "values") else np.asarray(y_train_)
    # Platt requiere ambas clases presentes en el OOF para ajustar la sigmoide.
    suficiente = mask_valid.sum() >= 30 and len(np.unique(y_arr[mask_valid])) == 2
    if not suficiente:
        estimator.fit(X_train_, y_train_)
        proba_raw = estimator.predict_proba(X_test_)[:, 1]
        if return_oof_train:
            return proba_raw, proba_raw, estimator, proba_oof
        return proba_raw, proba_raw, estimator

    platt = LogisticRegression(solver="lbfgs", max_iter=2000)
    platt.fit(proba_oof[mask_valid].reshape(-1, 1), y_arr[mask_valid])

    estimator.fit(X_train_, y_train_)
    proba_test_raw = estimator.predict_proba(X_test_)[:, 1]
    proba_test_cal = platt.predict_proba(proba_test_raw.reshape(-1, 1))[:, 1]

    if return_oof_train:
        proba_oof_cal = np.full_like(proba_oof, np.nan)
        proba_oof_cal[mask_valid] = platt.predict_proba(
            proba_oof[mask_valid].reshape(-1, 1)
        )[:, 1]
        return proba_test_raw, proba_test_cal, estimator, proba_oof_cal

    return proba_test_raw, proba_test_cal, estimator


def generar_oof_multiclase(
    estimator,
    X_train_,
    y_train_,
    n_classes: int,
    n_splits: int = 5,
):
    """OOF predictions multiclase via SafeTimeSeriesSplit (sin calibración).

    Para clasificadores multiclase (T2 del proyecto NOC). No aplica isotónica
    porque T2 se evalúa en F1 macro + argmax post-predict, no en métrica
    probabilística — la calibración no afecta el resultado.

    Args:
        estimator: clasificador sklearn-compatible con `predict_proba`.
        X_train_: matriz densa (post-ColumnTransformer) o DataFrame.
        y_train_: target multiclase (1D, valores en {0, ..., n_classes-1}).
        n_classes: número de clases del estimator.
        n_splits: folds de SafeTimeSeriesSplit (default 5).

    Returns:
        proba_oof: (N, n_classes) con NaN en posiciones del primer fold no validado.
    """
    cv = SafeTimeSeriesSplit(n_splits=n_splits, min_minority_samples=2)
    proba_oof = np.full((len(y_train_), n_classes), np.nan, dtype=float)
    for tr_idx, va_idx in cv.split(X_train_, y_train_):
        clf_clone = clone(estimator)
        if hasattr(X_train_, "iloc"):
            X_tr = X_train_.iloc[tr_idx]
            X_va = X_train_.iloc[va_idx]
        else:
            X_tr = X_train_[tr_idx]
            X_va = X_train_[va_idx]
        y_tr = y_train_.iloc[tr_idx] if hasattr(y_train_, "iloc") else y_train_[tr_idx]
        clf_clone.fit(X_tr, y_tr)
        proba_oof[va_idx] = clf_clone.predict_proba(X_va)
    return proba_oof
