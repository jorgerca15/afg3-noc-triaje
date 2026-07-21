"""Producto Final NOC — evaluacion combinada T1+T2 y metricas del pipeline vivo.

Canon vigente del proyecto (pivote semantico). Agrupa lo que consumen los
motores `afg3_*` vigentes y los notebooks `modelo.ipynb` / `validacion.ipynb`:

- Scorer operativo dinamico para Optuna: `scorer_operativo` (maximiza Recall
  sujeto a un piso de Precision recorriendo la curva PR completa).
- Metricas de ranking en ventanas semanales: `recall_at_k_global`,
  `ndcg_at_k_global`, `metricas_por_ventana_7d`.
- Reporte binario consolidado T1: `evaluar_binario`.
- Calibracion cuantitativa (ECE + Brier iso/Platt): `expected_calibration_error`,
  `metricas_calibracion`.
- Producto Final sobre la matriz de prioridad ordinal T1+T2:
  `compute_prioridad_ordinal`, `compute_score_combinado`,
  `evaluar_producto_final`, `paired_bootstrap_ic_producto_final`.

Metodologia documentada en la bitácora de decisiones del proyecto (2026-04-19).
`src/eval.py` re-exporta estos nombres como shim de compatibilidad.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    make_scorer,
    ndcg_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def _recall_at_min_precision(y_true, y_proba, min_precision: float = 0.15) -> float:
    """Recall máximo obtenible manteniendo Precision >= `min_precision`.

    Recorre la curva Precision-Recall y selecciona el mayor Recall entre
    los umbrales que cumplen el piso operativo. Si ningún umbral logra el
    piso, retorna 0 — el modelo es inútil operativamente.
    """
    precisions, recalls, _ = precision_recall_curve(y_true, y_proba)
    valid = np.where(precisions >= min_precision)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(recalls[valid]))

# Scorer compatible con `cross_val_score` y Optuna.
# `response_method='predict_proba'` reemplaza al deprecado `needs_proba=True`
# desde sklearn 1.6 en adelante.
scorer_operativo = make_scorer(
    _recall_at_min_precision,
    response_method='predict_proba',
    min_precision=0.15,
)

def recall_at_k_global(y_true, y_score, k: int = 5, seed: int = 42) -> float:
    """Recall@k global con desempate aleatorio reproducible."""
    y_true = np.asarray(y_true)

    np.random.seed(seed)
    noise = np.random.uniform(0, 1e-12, size=len(y_score))
    y_score = np.asarray(y_score) + noise

    n_pos = int(y_true.sum())
    if n_pos == 0:
        return np.nan

    k_eff = min(k, len(y_true))
    order = np.argsort(-y_score)
    top_k = y_true[order[:k_eff]]

    return float(top_k.sum() / n_pos)

def ndcg_at_k_global(y_true, y_score, k: int = 5, seed: int = 42) -> float:
    """NDCG@k global con desempate aleatorio reproducible."""
    y_true = np.asarray(y_true).reshape(1, -1)

    np.random.seed(seed)
    noise = np.random.uniform(0, 1e-12, size=y_score.shape)
    y_score = (np.asarray(y_score) + noise).reshape(1, -1)

    if y_true.sum() == 0:
        return np.nan

    return float(ndcg_score(y_true, y_score, k=k))

def metricas_por_ventana_7d(dates, y_true, y_score, k: int = 5,
                              dias_ventana: int = 7) -> dict:
    """Recall@k y NDCG@k agregados por ventanas de `dias_ventana` días.

    Args:
        dias_ventana: tamaño de la ventana en días (default 7, retrocompatible).
            Las keys de retorno mantienen el sufijo `_7d` solo cuando dias_ventana=7,
            sino usan `_{dias_ventana}d`.
    """
    dates = pd.to_datetime(pd.Series(dates).values)
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    orden = np.argsort(dates)
    dates = dates[orden]
    y_true = y_true[orden]
    y_score = y_score[orden]

    sufijo = f"{dias_ventana}d"
    if len(dates) == 0:
        return {f'recall@{k}_{sufijo}': np.nan, f'ndcg@{k}_{sufijo}': np.nan, 'n_ventanas': 0}

    t0 = dates.min()
    t_end = dates.max()
    delta = pd.Timedelta(days=dias_ventana)

    recalls, ndcgs = [], []
    t = t0

    while t <= t_end:
        mask = (dates >= t) & (dates < t + delta)
        if mask.sum() >= k and y_true[mask].sum() >= 1:
            recalls.append(recall_at_k_global(y_true[mask], y_score[mask], k))
            ndcgs.append(ndcg_at_k_global(y_true[mask], y_score[mask], k))
        t = t + delta

    return {
        f'recall@{k}_{sufijo}': float(np.nanmean(recalls)) if recalls else 0.0,
        f'ndcg@{k}_{sufijo}': float(np.nanmean(ndcgs)) if ndcgs else 0.0,
        'n_ventanas': len(recalls),
    }

def evaluar_binario(nombre, y_true, y_score, y_pred, dates_test) -> dict:
    """Reporte T1 aislado: clasificación binaria headline + ranking diagnóstico.

    Las keys headline miden desempeño de clasificación del modelo T1 por sí solo
    (AUC-PR, AUC-ROC, Brier, Precision/Recall/F2 al threshold calibrado, F1
    macro/weighted, matriz de confusión). Son las métricas prescritas por el
    curso para binario desbalanceado (IR ~5.76:1).

    Las keys con sufijo `_diagnostico` son sanity checks del poder de ranking
    del modelo T1 aislado. **NO son la métrica contractual de ranking del
    producto** — ésa vive en `evaluar_producto_final` sobre la matriz ordinal
    T1+T2 (ver README §Producto Final).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    n_neg, n_pos = np.bincount(y_true, minlength=2)
    ir = n_neg / n_pos if n_pos > 0 else 0

    # --- Headline: clasificación T1 pura ---
    auc_roc = float(roc_auc_score(y_true, y_score)) if n_pos > 0 and n_neg > 0 else np.nan
    auc_pr = float(average_precision_score(y_true, y_score)) if n_pos > 0 else np.nan
    brier = float(brier_score_loss(y_true, y_score))

    metricas = {
        'modelo': nombre,
        'IR_real': round(ir, 2),
        'auc_pr': round(auc_pr, 4) if not np.isnan(auc_pr) else np.nan,
        'auc_roc': round(auc_roc, 4) if not np.isnan(auc_roc) else np.nan,
        'brier': round(brier, 4),
        'precision_thr': round(float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        'recall_thr': round(float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)), 4),
        'f1_macro': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'f1_weighted': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'f2': round(float(fbeta_score(y_true, y_pred, beta=2, pos_label=1, zero_division=0)), 4),
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
    }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    metricas.update({'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'TP': int(tp)})

    # --- Diagnóstico: ranking T1 aislado (NO es la métrica producto) ---
    vent = metricas_por_ventana_7d(dates_test, y_true, y_score, k=5)
    metricas.update({
        'recall@5_global_diagnostico': recall_at_k_global(y_true, y_score, k=5),
        'ndcg@5_global_diagnostico': ndcg_at_k_global(y_true, y_score, k=5),
        'recall@5_7d_diagnostico': vent['recall@5_7d'],
        'ndcg@5_7d_diagnostico': vent['ndcg@5_7d'],
        'n_ventanas_validas_diagnostico': vent['n_ventanas'],
    })

    return metricas

def expected_calibration_error(y_true, y_score, n_bins: int = 10) -> float:
    """ECE con binning uniforme de confianza (Naeini et al. 2015; Guo et al. 2017).

    ECE = Σ_b (|B_b|/N) · |acc(B_b) − conf(B_b)|, sobre `n_bins` bins de [0,1].
    Cuantifica la calibración por modelo (complementa las curvas de confiabilidad
    visuales). Nota: en el canon AFG III la calibración de los encoders de texto
    es comparable o mejor que la de RF; el ECE se reporta por modelo sin asumir
    de antemano cuál calibra mejor.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)
    n = len(y_true)
    if n == 0:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_score, bins[1:-1]), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        conf = float(y_score[m].mean())
        acc = float(y_true[m].mean())
        ece += (m.sum() / n) * abs(acc - conf)
    return float(ece)

def metricas_calibracion(estimator, X_train_, y_train_, X_test_, y_test_,
                         n_splits: int = 5, n_bins: int = 10) -> dict:
    """C3: ECE y Brier sobre test para {raw, isotónica, Platt} vía OOF temporal.

    Corre la calibración isotónica y la de Platt (ambas con OOF SafeTimeSeriesSplit)
    y reporta ECE+Brier de la proba cruda y de cada calibración. Un modelo cuyo
    veredicto solo se sostiene bajo un método de calibración se declara frágil.
    """
    from sklearn.base import clone
    from src.calibracion import calibrar_isotonica, calibrar_platt

    raw, cal_iso, _ = calibrar_isotonica(clone(estimator), X_train_, y_train_, X_test_, n_splits=n_splits)
    _, cal_platt, _ = calibrar_platt(clone(estimator), X_train_, y_train_, X_test_, n_splits=n_splits)
    y = np.asarray(y_test_)
    out = {}
    for nombre, p in [("raw", raw), ("isotonica", cal_iso), ("platt", cal_platt)]:
        out[nombre] = {
            "ece": round(expected_calibration_error(y, p, n_bins), 4),
            "brier": round(float(brier_score_loss(y, p)), 4),
        }
    return out

# Agrupación de keys para render tabular en notebook.
HEADLINE_KEYS_T1 = [
    'modelo', 'IR_real',
    'auc_pr', 'auc_roc', 'brier',
    'precision_thr', 'recall_thr', 'f2',
    'f1_macro', 'f1_weighted', 'accuracy',
    'TN', 'FP', 'FN', 'TP',
]

DIAGNOSTICO_KEYS_T1 = [
    'modelo',
    'recall@5_global_diagnostico', 'ndcg@5_global_diagnostico',
    'recall@5_7d_diagnostico', 'ndcg@5_7d_diagnostico',
    'n_ventanas_validas_diagnostico',
]

def compute_prioridad_ordinal(y_t1, y_t2, idx_emergencia: int = 2) -> np.ndarray:
    """Matriz de prioridad operativa NOC (0-3) — ver README §Producto Final.

        Nivel 3: Crítico AND Emergencia          — top-prioridad absoluta
        Nivel 2: Crítico AND T2 != Emergencia    — severo pero no urgente
        Nivel 1: Leve    AND Emergencia          — urgente pero no severo
        Nivel 0: resto                           — rutina

    Se usa tanto para construir la relevancia graduada real (con `y_true`)
    como para derivar el `score_operativo` predicho (con `y_pred`). Mantener
    una sola función evita divergencia entre el notebook (`df_test` con
    columnas materializadas) y la evaluación (`evaluar_producto_final`).

    Referencia: Järvelin & Kekäläinen 2002 "Cumulated Gain-Based Evaluation
    of IR Techniques" (ACM TOIS 20(4)). La matriz devuelve niveles lineales
    {0,1,2,3}; la transformación geométrica de ganancia `2^r - 1` que define
    el DCG se aplica downstream en `_ndcg_graduado`. El score de ordenamiento
    (ver `compute_score_combinado`) usa niveles lineales porque sólo importa
    el orden (toda transformación monótona preserva el ranking) y porque la
    geometría del impacto pertenece a la métrica del NDCG, no al score.
    """
    y_t1 = np.asarray(y_t1).astype(int)
    y_t2 = np.asarray(y_t2).astype(int)
    assert len(y_t1) == len(y_t2), (
        f'Longitudes desalineadas en compute_prioridad_ordinal: '
        f'{len(y_t1)} vs {len(y_t2)}'
    )
    is_crit = (y_t1 == 1)
    is_eme = (y_t2 == idx_emergencia)
    return (is_crit & is_eme) * 3 + (is_crit & ~is_eme) * 2 + (~is_crit & is_eme) * 1

def _score_ordinal_labels(y_true_t1, y_true_t2, pred_t1, pred_t2, idx_emergencia):
    """Construye los tres vectores de la matriz de prioridad ordinal.

    Returns:
        y_true_estricto : {0,1} — 1 sólo en Nivel 3 real.
        y_true_graduado : {0,1,2,3} — relevancia ordinal real.
        score_operativo : {0,1,2,3} — relevancia ordinal predicha.
    """
    y_true_graduado = compute_prioridad_ordinal(y_true_t1, y_true_t2, idx_emergencia)
    y_true_estricto = (y_true_graduado == 3).astype(int)
    score_operativo = compute_prioridad_ordinal(pred_t1, pred_t2, idx_emergencia).astype(float)

    return y_true_estricto, y_true_graduado, score_operativo

def compute_score_combinado(
    pred_t1,
    pred_t2,
    proba_t1,
    idx_emergencia: int = 2,
    epsilon_tiebreak: float = 0.1,
    noise_scale: float = 1e-9,
    seed: int = 42,
) -> np.ndarray:
    """Fórmula canónica del score de ordenamiento del Producto Final.

    Notación del informe (idéntica aquí): `Score = Nivel + α·P(Crítico) + ε`,
    con `Nivel = compute_prioridad_ordinal(pred_t1, pred_t2) ∈ {0,1,2,3}`,
    `α = epsilon_tiebreak = 0.1` (hiperparámetro fijo de REGLA DE NEGOCIO, peso
    de desempate intra-nivel) y `ε = ruido ~ U(0, 1e-9)` (desempate residual).
    El `α` de desempate es DISTINTO del `α*` del blend del ensamble (Grupo C).

    NATURALEZA METODOLÓGICA: esto es una **heurística de ordenamiento post-hoc /
    fusión lineal de scores basada en reglas de negocio**, NO un Learning-to-Rank
    pointwise (que entrenaría un modelo optimizando una pérdida de relevancia por
    ítem). Se eligió frente al paradigma LTR formal (Liu 2009, "Learning to Rank
    for Information Retrieval", Found. Trends IR) por interpretabilidad directa y
    para evitar leakage de grupo; la transición a listwise/pairwise (LambdaMART)
    es trabajo futuro no ejecutado.

    Única fuente de verdad de la fórmula. Tanto `evaluar_producto_final` (que
    la usa internamente) como las celdas de materialización en
    `notebooks/modelo_v4.ipynb` §7 deben llamar a este helper para garantizar
    que la columna `score_ordenamiento_<config>` y el ranking interno de la
    función produzcan el MISMO orden — test de no-regresión E3 lo verifica.

    Decisión metodológica vinculante (bitácora 2026-05-18, refuerzo R1):
    el score permanece lineal `{0,1,2,3} + 0.1·proba_t1`. NO se geometriza
    a `{0,1,3,7} + 0.1·proba_t1` porque (i) toda transformación monótona
    creciente preserva el ranking — el orden producido por argsort es
    idéntico bit-a-bit; (ii) geometrizar rompería el test de no-regresión
    E3 sobre los manifests SHA256 ya cerrados. La geometría del impacto
    (Järvelin & Kekäläinen 2002) vive aguas abajo en `_ndcg_graduado` vía
    `2^r - 1`, que es donde matemáticamente corresponde.

    Args:
        pred_t1: (n,) {0,1}.
        pred_t2: (n,) {0,1,2,3} codificación CATS_T2.
        proba_t1: (n,) [0,1] — P(Crítico) calibrada OOF.
        idx_emergencia: índice de 'Emergencia' en CATS_T2 (default 2).
        epsilon_tiebreak: peso del desempate intra-nivel (default 0.1).
        noise_scale: amplitud del ruido residual (default 1e-9).
        seed: semilla del RNG para reproducibilidad (default 42).

    Returns:
        score: (n,) float64.
    """
    s_op = compute_prioridad_ordinal(pred_t1, pred_t2, idx_emergencia).astype(float)
    rng = np.random.default_rng(seed)
    noise = rng.uniform(0, noise_scale, size=len(s_op))
    return s_op + epsilon_tiebreak * np.asarray(proba_t1, dtype=float) + noise

def _ndcg_graduado(y_grad_window, order, k):
    """NDCG@k con relevancia graduada {0,1,2,3} y ganancia geométrica.

    DCG normalizado por el ideal de los k mejores de la ventana.
    Retorna NaN si la ventana no tiene ninguna relevancia > 0 (IDCG=0).

    Implementa la formulación geométrica de Järvelin & Kekäläinen 2002
    "Cumulated Gain-Based Evaluation of IR Techniques" (ACM TOIS 20(4)):
    `DCG = Σ (2^rel_i - 1) / log2(i + 1)`. Esto garantiza que el aporte
    de los niveles ordinales crece exponencialmente con la relevancia
    (1, 3, 7 para niveles 1, 2, 3 respectivamente) — alineado con la
    realidad operativa del NOC donde la penalización por SLA no es
    lineal sino exponencial.
    """
    k_eff = min(k, len(y_grad_window))
    if k_eff == 0:
        return np.nan

    rel_top = y_grad_window[order[:k_eff]].astype(float)
    rel_ideal = np.sort(y_grad_window)[::-1][:k_eff].astype(float)

    discounts = 1.0 / np.log2(np.arange(2, k_eff + 2))
    dcg = float(np.sum((2 ** rel_top - 1) * discounts))
    idcg = float(np.sum((2 ** rel_ideal - 1) * discounts))

    if idcg == 0.0:
        return np.nan
    return dcg / idcg

def evaluar_producto_final(
    nombre,
    y_true_t1,
    y_true_t2,
    pred_t1,
    pred_t2,
    proba_t1,
    dates_test,
    k: int = 5,
    idx_emergencia: int = 2,
    epsilon_tiebreak: float = 0.1,
    seed: int = 42,
    dias_ventana: int = 7,
) -> dict:
    """Evalúa el ranking combinado T1+T2 en ventanas de 7 días.

    Matriz de prioridad ordinal (niveles 0-3):
        Nivel 3: Crítico AND Emergencia
        Nivel 2: Crítico AND T2 != Emergencia
        Nivel 1: Leve    AND Emergencia
        Nivel 0: resto

    Ordenamiento:
        score_combinado = score_operativo + epsilon_tiebreak * proba_t1
        Como `epsilon_tiebreak * proba_t1 ∈ [0, 0.1]` < 1.0 (salto entre niveles),
        la jerarquía ordinal se preserva y el desempate intra-nivel usa la
        confianza calibrada OOF del modelo T1.

    Métricas:
        - `recall@k_pf_7d_descriptivo`: promedio sobre ventanas con ≥ 1
          positivo estricto (Nivel 3). Varianza alta por baja cardinalidad
          estructural — reportado con fines descriptivos.
        - `ndcg@k_pf_7d_PRIMARIA`: promedio sobre ventanas con IDCG > 0.
          Métrica primaria del producto final — aprovecha los 4 niveles de
          relevancia y tiene soporte amplio.

    Args:
        nombre: identificador de la configuración evaluada.
        y_true_t1: (n,) {0,1}.
        y_true_t2: (n,) {0,1,2,3} con codificación estándar de CATS_T2.
        pred_t1: (n,) {0,1} post-threshold OOF.
        pred_t2: (n,) {0,1,2,3}.
        proba_t1: (n,) [0,1] — P(Crítico) calibrada OOF.
        dates_test: (n,) datetime-like.
        k: tamaño del top a evaluar (default 5).
        idx_emergencia: índice de 'Emergencia' en CATS_T2 (default 2).
        epsilon_tiebreak: magnitud del desempate intra-nivel (default 0.1).

    Returns:
        dict con métricas y contadores estructurales del test set.
    """
    arrays = [y_true_t1, y_true_t2, pred_t1, pred_t2, proba_t1, dates_test]
    n = len(arrays[0])
    for a in arrays[1:]:
        assert len(a) == n, (
            f'Longitudes desalineadas en evaluar_producto_final: '
            f'{[len(x) for x in arrays]}'
        )

    y_true_t1 = np.asarray(y_true_t1)
    y_true_t2 = np.asarray(y_true_t2)
    pred_t1 = np.asarray(pred_t1)
    pred_t2 = np.asarray(pred_t2)
    proba_t1 = np.asarray(proba_t1, dtype=float)
    dates_arr = pd.to_datetime(np.asarray(dates_test))

    y_est, y_grad, _ = _score_ordinal_labels(
        y_true_t1, y_true_t2, pred_t1, pred_t2, idx_emergencia,
    )
    # Desempate jerárquico (ver `compute_score_combinado`):
    #   (1) nivel ordinal — salto entre niveles = 1.0 (domina)
    #   (2) proba_t1 calibrada — domina cuando los niveles empatan
    #   (3) ruido 1e-9 — último desempate cuando tanto nivel como proba son iguales
    #       (crítico para Dummy, que tiene proba=0 uniforme y produciría orden
    #       cronológico espurio con `kind='stable'`).
    score_combinado = compute_score_combinado(
        pred_t1, pred_t2, proba_t1,
        idx_emergencia=idx_emergencia,
        epsilon_tiebreak=epsilon_tiebreak,
        seed=seed,
    )

    df_eval = pd.DataFrame({
        'date': dates_arr,
        'y_est': y_est,
        'y_grad': y_grad,
        'score': score_combinado,
    }).sort_values('date').reset_index(drop=True)

    sufijo_pf = f"pf_{dias_ventana}d"
    if len(df_eval) == 0:
        return {
            'modelo': nombre,
            f'recall@{k}_{sufijo_pf}_descriptivo': 0.0,
            f'ndcg@{k}_{sufijo_pf}_PRIMARIA': 0.0,
            'n_ventanas': 0,
            'n_ventanas_con_positivos_estrictos': 0,
            'n_positivos_estrictos_total': 0,
            'n_nivel_2': 0,
            'n_nivel_1': 0,
            'n_tickets_por_ventana_promedio': 0.0,
            # Keys nuevas defensa Zero-Density (Anexo PASO 20):
            'n_ventanas_totales_periodo': 0,
            'n_ventanas_validas_ndcg': 0,
            'cobertura_efectiva': 0.0,
        }

    t0 = df_eval['date'].min()
    t_end = df_eval['date'].max()
    delta = pd.Timedelta(days=dias_ventana)

    recalls, ndcgs, tamanos = [], [], []
    n_ventanas_con_pos = 0
    ventanas_totales_periodo = 0  # incluye las vacías

    cursor = t0
    while cursor <= t_end:
        ventanas_totales_periodo += 1
        w = df_eval[(df_eval['date'] >= cursor) & (df_eval['date'] < cursor + delta)]
        if len(w) == 0:
            cursor = cursor + delta
            continue

        tamanos.append(len(w))
        order = np.argsort(-w['score'].values, kind='stable')
        top = order[:k]

        y_est_w = w['y_est'].values
        y_grad_w = w['y_grad'].values
        total_pos = int(y_est_w.sum())

        if total_pos > 0:
            recalls.append(y_est_w[top].sum() / total_pos)
            n_ventanas_con_pos += 1

        ndcg_w = _ndcg_graduado(y_grad_w, order, k)
        if not np.isnan(ndcg_w):
            ndcgs.append(ndcg_w)

        cursor = cursor + delta

    return {
        'modelo': nombre,
        f'recall@{k}_{sufijo_pf}_descriptivo': float(np.mean(recalls)) if recalls else 0.0,
        f'ndcg@{k}_{sufijo_pf}_PRIMARIA': float(np.mean(ndcgs)) if ndcgs else 0.0,
        'n_ventanas': len(tamanos),
        'n_ventanas_con_positivos_estrictos': n_ventanas_con_pos,
        'n_positivos_estrictos_total': int(y_est.sum()),
        'n_nivel_2': int((y_grad == 2).sum()),
        'n_nivel_1': int((y_grad == 1).sum()),
        'n_tickets_por_ventana_promedio': float(np.mean(tamanos)) if tamanos else 0.0,
        # Keys nuevas defensa Zero-Density (Anexo PASO 20):
        'n_ventanas_totales_periodo': ventanas_totales_periodo,
        'n_ventanas_validas_ndcg': len(ndcgs),
        'cobertura_efectiva': (len(ndcgs) / ventanas_totales_periodo) if ventanas_totales_periodo > 0 else 0.0,
    }

def paired_bootstrap_ic_producto_final(
    nombre_a,
    nombre_b,
    y_true_t1,
    y_true_t2,
    pred_t1_a,
    pred_t2_a,
    proba_t1_a,
    pred_t1_b,
    pred_t2_b,
    proba_t1_b,
    dates_test,
    k: int = 5,
    idx_emergencia: int = 2,
    n_boot: int = 1000,
    seed: int = 42,
    dias_ventana: int = 7,
    estratificar_densidad_critica: bool = False,
    return_replicas: bool = False,
) -> dict:
    """Paired bootstrap a nivel ventana: evalúa dos modelos A y B con los
    MISMOS índices de ventana sampleados, controla la varianza compartida y
    reporta IC95 absolutos + IC95 del Δ (B-A) + p-value empírico.

    Soporta el caso de uso T15 del Avance Final V2 (HÍBRIDO vs tabular):
    si IC95 de Δ incluye 0, se documenta "empate estadístico".

    Args:
        estratificar_densidad_critica: si True, estratifica el sorteo de bloques
            por "ventana con ≥1 crítico T1" vs "ventana sin críticos", manteniendo
            FIJA la cantidad de ventanas de cada estrato en cada réplica (revisión
            AFG III). Default False reproduce bit-a-bit el block bootstrap previo.
            Caveat: estabiliza el ruido Monte-Carlo del p-valor (ninguna réplica
            sortea 0 ventanas-críticas), NO crea potencia ni cambia que el IC95
            siga incluyendo 0 a 12 críticos; condiciona en la proporción observada
            de ventanas-críticas (estimando distinto). Se reporta junto a la
            variante clásica para transparencia.

    Returns:
        dict con ndcg_a/b mean+ic, delta mean+ic, p_value (proporción Δ ≤ 0),
        n_boot, n_ventanas, estratificado (bool).
    """
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(np.asarray(dates_test))
    y_t1 = np.asarray(y_true_t1)
    y_t2 = np.asarray(y_true_t2)
    p_t1_a = np.asarray(pred_t1_a)
    p_t2_a = np.asarray(pred_t2_a)
    pr_t1_a = np.asarray(proba_t1_a)
    p_t1_b = np.asarray(pred_t1_b)
    p_t2_b = np.asarray(pred_t2_b)
    pr_t1_b = np.asarray(proba_t1_b)

    if len(dates) == 0:
        return {
            'modelo_a': nombre_a, 'modelo_b': nombre_b,
            'n_boot': 0, 'n_ventanas': 0,
        }

    cursor = dates.min().normalize()
    delta_t = pd.Timedelta(days=dias_ventana)
    ventanas_idx = []
    while cursor <= dates.max():
        mask = (dates >= cursor) & (dates < cursor + delta_t)
        if mask.any():
            ventanas_idx.append(np.where(mask)[0])
        cursor = cursor + delta_t

    n_v = len(ventanas_idx)
    if n_v == 0:
        return {
            'modelo_a': nombre_a, 'modelo_b': nombre_b,
            'n_boot': 0, 'n_ventanas': 0,
        }

    # Estratos por densidad crítica: posiciones de ventana con ≥1 crítico T1 vs sin.
    # El bloque (ventanas_idx[i]) se mantiene SIEMPRE íntegro — nunca se sub-muestrean
    # filas dentro de la ventana (preserva la competencia temporal del NDCG).
    if estratificar_densidad_critica:
        pos_crit = [i for i in range(n_v) if y_t1[ventanas_idx[i]].sum() > 0]
        pos_vacia = [i for i in range(n_v) if y_t1[ventanas_idx[i]].sum() == 0]
        pos_crit_arr = np.asarray(pos_crit, dtype=int)
        pos_vacia_arr = np.asarray(pos_vacia, dtype=int)

    ndcg_a, ndcg_b, deltas = [], [], []
    primary_key = f'ndcg@{k}_pf_{dias_ventana}d_PRIMARIA'
    for _ in range(n_boot):
        if estratificar_densidad_critica:
            partes = []
            if len(pos_crit_arr) > 0:
                partes.append(pos_crit_arr[rng.integers(0, len(pos_crit_arr), size=len(pos_crit_arr))])
            if len(pos_vacia_arr) > 0:
                partes.append(pos_vacia_arr[rng.integers(0, len(pos_vacia_arr), size=len(pos_vacia_arr))])
            sel = np.concatenate(partes)
        else:
            sel = rng.integers(0, n_v, size=n_v)
        idx = np.concatenate([ventanas_idx[i] for i in sel])
        res_a = evaluar_producto_final(
            nombre=nombre_a, y_true_t1=y_t1[idx], y_true_t2=y_t2[idx],
            pred_t1=p_t1_a[idx], pred_t2=p_t2_a[idx],
            proba_t1=pr_t1_a[idx], dates_test=dates[idx],
            k=k, idx_emergencia=idx_emergencia, dias_ventana=dias_ventana,
        )
        res_b = evaluar_producto_final(
            nombre=nombre_b, y_true_t1=y_t1[idx], y_true_t2=y_t2[idx],
            pred_t1=p_t1_b[idx], pred_t2=p_t2_b[idx],
            proba_t1=pr_t1_b[idx], dates_test=dates[idx],
            k=k, idx_emergencia=idx_emergencia, dias_ventana=dias_ventana,
        )
        va = res_a[primary_key]
        vb = res_b[primary_key]
        ndcg_a.append(va)
        ndcg_b.append(vb)
        deltas.append(vb - va)

    arr_a = np.asarray(ndcg_a)
    arr_b = np.asarray(ndcg_b)
    arr_d = np.asarray(deltas)
    # p-value empírico unilateral: ¿qué proporción de réplicas B ≤ A?
    p_value = float((arr_d <= 0).mean())
    resultado = {
        'modelo_a': nombre_a,
        'modelo_b': nombre_b,
        'ndcg_a_mean': float(arr_a.mean()),
        'ndcg_a_ic_low': float(np.percentile(arr_a, 2.5)),
        'ndcg_a_ic_high': float(np.percentile(arr_a, 97.5)),
        'ndcg_b_mean': float(arr_b.mean()),
        'ndcg_b_ic_low': float(np.percentile(arr_b, 2.5)),
        'ndcg_b_ic_high': float(np.percentile(arr_b, 97.5)),
        'delta_mean': float(arr_d.mean()),
        'delta_ic_low': float(np.percentile(arr_d, 2.5)),
        'delta_ic_high': float(np.percentile(arr_d, 97.5)),
        'delta_ic_incluye_0': bool(np.percentile(arr_d, 2.5) <= 0 <= np.percentile(arr_d, 97.5)),
        'p_value_unilateral_B_mejor_A': p_value,
        'n_boot': n_boot,
        'n_ventanas': n_v,
        'estratificado': bool(estratificar_densidad_critica),
    }
    if return_replicas:
        # Réplicas crudas para KDE (Gráfico B): solo cuando se piden explícitamente,
        # de modo que el contrato por defecto queda bit-idéntico (protege artefactos
        # hasheados y llamadores previos que comparan keys del dict).
        resultado['ndcg_a_replicas'] = arr_a.tolist()
        resultado['ndcg_b_replicas'] = arr_b.tolist()
        resultado['delta_replicas'] = arr_d.tolist()
    return resultado
