"""Analitica SUPERSEDED — costo asimetrico 50:1, cross-arm, MDES y ejes RAMA-B.

Bloque de trabajo historico de las iteraciones RAMA-B / PASO 19-21 (torneo
multimodelo, decision por costo 50:1, cadena T1->T2, sweeps de ventana, estres
adversarial). El pivote semantico (canon vigente) NO usa estas funciones en su
pipeline de resultados; se conservan para reproducir los scripts `avance_final_t*`
y los motores de reanclaje/multimodelo ya hasheados.

`src/eval.py` re-exporta estos nombres como shim de compatibilidad. Las metricas
del PRODUCTO Final vivo (matriz ordinal T1+T2) viven en `src/eval_producto.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from .eval_producto import (
    compute_prioridad_ordinal,
    compute_score_combinado,
    evaluar_producto_final,
)


def evaluar_multiclase(nombre, y_true, y_pred, labels=None, target_names=None) -> dict:
    """Reporte T2 aislado: accuracy + f1 macro/weighted + per-clase + matriz.

    Para T2 (4 categorías de tipo de trabajo) no se evalúa ranking — no hay
    un positivo único que permita Recall@k. El ranking T2 vive sólo dentro
    del producto final combinado (T1+T2) vía `evaluar_producto_final`.

    Args:
        nombre: identificador de la configuración.
        y_true, y_pred: arrays 1D con códigos {0,1,2,3}.
        labels: lista de códigos ordenados para la matriz (default [0,1,2,3]).
        target_names: nombres de clase alineados a `labels`.

    Returns:
        dict con accuracy, f1 macro/weighted, listas per-clase y matriz.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if labels is None:
        labels = sorted(set(np.concatenate([y_true, y_pred]).tolist()))
    if target_names is None:
        target_names = [str(l) for l in labels]

    report = classification_report(
        y_true, y_pred,
        labels=labels, target_names=target_names,
        output_dict=True, zero_division=0,
    )

    precision_por_clase = [round(float(report[n]['precision']), 4) for n in target_names]
    recall_por_clase = [round(float(report[n]['recall']), 4) for n in target_names]
    f1_por_clase = [round(float(report[n]['f1-score']), 4) for n in target_names]
    soporte_por_clase = [int(report[n]['support']) for n in target_names]

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        'modelo': nombre,
        'accuracy': round(float(accuracy_score(y_true, y_pred)), 4),
        'f1_macro': round(float(f1_score(y_true, y_pred, average='macro', zero_division=0)), 4),
        'f1_weighted': round(float(f1_score(y_true, y_pred, average='weighted', zero_division=0)), 4),
        'labels': list(labels),
        'target_names': list(target_names),
        'precision_por_clase': precision_por_clase,
        'recall_por_clase': recall_por_clase,
        'f1_por_clase': f1_por_clase,
        'soporte_por_clase': soporte_por_clase,
        'matriz_confusion': cm.tolist(),
    }

def evaluar_por_subgrupo(
    nombre,
    y_true_t1,
    y_true_t2,
    pred_t1,
    pred_t2,
    proba_t1,
    dates_test,
    subgrupo_array,
    k: int = 5,
    idx_emergencia: int = 2,
    min_n_por_grupo: int = 20,
) -> "list[dict]":
    """Evalúa el producto final segmentado por subgrupo (turno, tipo de red, etc.).

    Para cada nivel único en `subgrupo_array`, aplica `evaluar_producto_final`
    al slice correspondiente. Subgrupos con menos de `min_n_por_grupo` tickets
    se incluyen pero se marcan con flag `n_insuficiente=True`.

    Args:
        subgrupo_array: (n,) array-like con la categoría de cada ticket.
        min_n_por_grupo: umbral debajo del cual el subgrupo se reporta como
            de potencia insuficiente.

    Returns:
        Lista de dicts (uno por nivel del subgrupo) con métricas + nombre del
        subgrupo + n del slice + flag de potencia.
    """
    sg = np.asarray(subgrupo_array)
    y_t1 = np.asarray(y_true_t1)
    y_t2 = np.asarray(y_true_t2)
    p_t1 = np.asarray(pred_t1)
    p_t2 = np.asarray(pred_t2)
    pr_t1 = np.asarray(proba_t1)
    dates = pd.to_datetime(np.asarray(dates_test))

    filas = []
    for nivel in pd.Series(sg).dropna().unique():
        mask = (sg == nivel)
        n_g = int(mask.sum())
        if n_g == 0:
            continue
        try:
            res = evaluar_producto_final(
                nombre=f'{nombre} · {nivel}',
                y_true_t1=y_t1[mask],
                y_true_t2=y_t2[mask],
                pred_t1=p_t1[mask],
                pred_t2=p_t2[mask],
                proba_t1=pr_t1[mask],
                dates_test=dates[mask],
                k=k,
                idx_emergencia=idx_emergencia,
            )
        except Exception as exc:  # pragma: no cover — ventanas degeneradas
            res = {
                'modelo': f'{nombre} · {nivel}',
                f'recall@{k}_pf_7d_descriptivo': float('nan'),
                f'ndcg@{k}_pf_7d_PRIMARIA': float('nan'),
                'error': str(exc),
            }
        res['subgrupo'] = str(nivel)
        res['n_subgrupo'] = n_g
        res['n_insuficiente'] = bool(n_g < min_n_por_grupo)
        filas.append(res)
    return filas

def bootstrap_ic_producto_final(
    nombre,
    y_true_t1,
    y_true_t2,
    pred_t1,
    pred_t2,
    proba_t1,
    dates_test,
    k: int = 5,
    idx_emergencia: int = 2,
    n_boot: int = 1000,
    seed: int = 42,
    dias_ventana: int = 7,
) -> dict:
    """Bootstrap a nivel ventana del NDCG@5_pf_7d.

    Particiona el test en ventanas de 7 días, muestrea ventanas con reemplazo
    `n_boot` veces, y reporta media + IC95.

    Returns:
        dict con `ndcg_mean`, `ndcg_ic_low`, `ndcg_ic_high`, `n_boot`, `n_ventanas`.
    """
    rng = np.random.default_rng(seed)
    dates = pd.to_datetime(np.asarray(dates_test))
    y_t1 = np.asarray(y_true_t1)
    y_t2 = np.asarray(y_true_t2)
    p_t1 = np.asarray(pred_t1)
    p_t2 = np.asarray(pred_t2)
    pr_t1 = np.asarray(proba_t1)

    # Indexar ventanas de 7 días
    if len(dates) == 0:
        return {'modelo': nombre, 'ndcg_mean': float('nan'), 'ndcg_ic_low': float('nan'),
                'ndcg_ic_high': float('nan'), 'n_boot': 0, 'n_ventanas': 0}
    cursor = dates.min().normalize()
    delta = pd.Timedelta(days=dias_ventana)
    ventanas_idx = []
    while cursor <= dates.max():
        mask = (dates >= cursor) & (dates < cursor + delta)
        if mask.any():
            ventanas_idx.append(np.where(mask)[0])
        cursor = cursor + delta

    if len(ventanas_idx) == 0:
        return {'modelo': nombre, 'ndcg_mean': float('nan'), 'ndcg_ic_low': float('nan'),
                'ndcg_ic_high': float('nan'), 'n_boot': 0, 'n_ventanas': 0}

    ndcgs_b = []
    n_v = len(ventanas_idx)
    for _ in range(n_boot):
        sel = rng.integers(0, n_v, size=n_v)
        idx_concat = np.concatenate([ventanas_idx[i] for i in sel])
        # Re-evaluar producto final sobre el sub-test
        res = evaluar_producto_final(
            nombre=nombre,
            y_true_t1=y_t1[idx_concat],
            y_true_t2=y_t2[idx_concat],
            pred_t1=p_t1[idx_concat],
            pred_t2=p_t2[idx_concat],
            proba_t1=pr_t1[idx_concat],
            dates_test=dates[idx_concat],
            k=k,
            dias_ventana=dias_ventana,
            idx_emergencia=idx_emergencia,
        )
        ndcgs_b.append(res[f'ndcg@{k}_pf_{dias_ventana}d_PRIMARIA'])

    arr = np.asarray(ndcgs_b)
    return {
        'modelo': nombre,
        'ndcg_mean': float(arr.mean()),
        'ndcg_ic_low': float(np.percentile(arr, 2.5)),
        'ndcg_ic_high': float(np.percentile(arr, 97.5)),
        'n_boot': n_boot,
        'n_ventanas': n_v,
    }

def filtrar_cola_larga(
    df_test: pd.DataFrame,
    df_train: pd.DataFrame,
    columna: str,
    umbral: float = 0.05,
) -> dict:
    """Identifica tickets de test en categorías raras (frec < umbral) en train.

    Devuelve un dict con:
        - 'categorias_raras': set de valores con frecuencia < umbral en train
        - 'frecuencias_train': dict valor → frecuencia
        - 'mask_test': boolean array sobre df_test, True donde la categoría es rara
        - 'n_rare_test': cantidad de tickets test en cola larga
        - 'n_train': total train
        - 'umbral': el umbral aplicado
    """
    if columna not in df_train.columns:
        raise KeyError(f"columna {columna!r} no está en df_train")
    if columna not in df_test.columns:
        raise KeyError(f"columna {columna!r} no está en df_test")
    n_train = len(df_train)
    freq_train = df_train[columna].value_counts(dropna=False) / n_train
    raras = set(freq_train[freq_train < umbral].index)
    mask = df_test[columna].isin(raras).values
    return {
        'columna': columna,
        'umbral': umbral,
        'categorias_raras': sorted([str(c) for c in raras]),
        'n_categorias_raras': int(len(raras)),
        'frecuencias_train': {str(k): float(v) for k, v in freq_train.to_dict().items()},
        'mask_test': mask,
        'n_rare_test': int(mask.sum()),
        'n_train': int(n_train),
    }

def _rbo_extrapolated(list_a, list_b, p: float = 0.9) -> float:
    """Rank-Biased Overlap extrapolado (Webber, Moffat & Zobel 2010).

    RBO compara dos rankings posiblemente no conjuntos ponderando más las
    posiciones altas. El parámetro `p` controla la decadencia: con p=0.9 el
    peso acumulado del top-10 es ~86%; con p=0.8 el peso del top-5 es ~67%.

    Implementación:
        RBO_ext = (1-p) * Σ_{d=1..k} (X_d / d) * p^(d-1) + (X_k / k) * p^k

    donde X_d = |A[:d] ∩ B[:d]| (overlap acumulado a profundidad d) y k es
    la longitud común de las listas. El término (X_k/k)·p^k extrapola el
    overlap más allá del horizonte observable (asume estabilidad del overlap
    acumulado), evitando el sesgo conservador de RBO_min.

    Args:
        list_a, list_b: secuencias ordenadas (de mayor a menor relevancia).
        p: parámetro de decadencia ∈ (0, 1). Típico: 0.9 para top-10.

    Returns:
        RBO ∈ [0, 1]. 1.0 = rankings idénticos; 0.0 = sin overlap.
    """
    if not 0 < p < 1:
        raise ValueError(f'p debe estar en (0, 1), recibido {p}')
    a = list(list_a)
    b = list(list_b)
    k = min(len(a), len(b))
    if k == 0:
        return float('nan')

    seen_a, seen_b = set(), set()
    overlap = 0
    sum_term = 0.0
    for d in range(1, k + 1):
        seen_a.add(a[d - 1])
        seen_b.add(b[d - 1])
        # Recalcular overlap acumulado a profundidad d
        overlap = len(seen_a & seen_b)
        sum_term += (overlap / d) * (p ** (d - 1))

    rbo_min = (1 - p) * sum_term
    extrapolation = (overlap / k) * (p ** k)
    return float(rbo_min + extrapolation)

def bootstrap_estabilidad_ranking(
    nombre,
    y_true_t1,
    y_true_t2,
    pred_t1,
    pred_t2,
    proba_t1,
    dates_test,
    k_top: int = 20,
    idx_emergencia: int = 2,
    epsilon_tiebreak: float = 0.1,
    n_boot: int = 1000,
    seed: int = 42,
    rbo_p_values=(0.8, 0.9),
    jaccard_k_values=(5, 10, 20),
) -> dict:
    """Estabilidad del top-K bajo bootstrap a nivel ticket — multi-métrica.

    NOTA METODOLÓGICA: `compute_score_combinado` es determinista sobre los
    tickets, por lo que Kendall τ entre tickets compartidos en dos réplicas
    bootstrap es ≡ 1 (mantienen orden por el mismo score). Reportamos en su
    lugar dos familias de métricas con propiedades complementarias:

      - **Jaccard@K** (K ∈ {5, 10, 20}): mide overlap de composición del top-K
        entre réplicas consecutivas. Simple, intuitivo. Ciego al orden interno
        (un ticket en pos 1 cuenta igual que en pos K).

      - **RBO@p** (Rank-Biased Overlap, Webber et al. 2010, con p ∈ {0.8, 0.9}):
        métrica de oro para estabilidad de top-K en IR. Pondera descendentemente
        (top de la lista pesa más), maneja listas no conjuntas, valores en [0,1].
        Para nuestro caso (triaje NOC) es la métrica más alineada con el costo
        operativo real de la diferencia "primero vs décimo".

    Args:
        k_top: profundidad máxima del ranking que se guarda por réplica (default 20,
            suficiente para Jaccard@20 y RBO sobre top-20).
        rbo_p_values: tupla de p para reportar varios RBO en paralelo.
        jaccard_k_values: tupla de K para reportar varios Jaccard en paralelo.

    Reporta:
      - `jaccard_at5/10/20` (mean): Jaccard@K entre réplicas consecutivas.
      - `rbo_p08`, `rbo_p09` (mean): RBO con p=0.8 (énfasis top-5) y p=0.9 (top-10).
      - `n_tickets_inestables`: tickets en ≥ 50% de los top-K con `pos_std > 1.0`.

    Returns:
        dict con todas las métricas + n_boot.
    """
    rng = np.random.default_rng(seed)
    n = len(pred_t1)

    def _nan_result():
        out = {'modelo': nombre, 'n_tickets_inestables': 0, 'n_boot': 0}
        for k in jaccard_k_values:
            out[f'jaccard_at{k}'] = float('nan')
        for p in rbo_p_values:
            out[f'rbo_p{int(p * 10):02d}'] = float('nan')
        return out

    if n == 0:
        return _nan_result()

    p_t1 = np.asarray(pred_t1)
    p_t2 = np.asarray(pred_t2)
    pr_t1 = np.asarray(proba_t1)
    score = compute_score_combinado(
        p_t1, p_t2, pr_t1,
        idx_emergencia=idx_emergencia,
        epsilon_tiebreak=epsilon_tiebreak,
        seed=seed,
    )

    # Guardamos las primeras k_top posiciones del ranking de cada réplica
    # (lista ordenada de índices originales).
    rankings_ordered = []
    posiciones_por_ticket = {}
    k_obs = min(k_top, n)
    for b in range(n_boot):
        sel = rng.integers(0, n, size=n)
        score_b = score[sel]
        order = np.argsort(-score_b)[:k_obs]
        idx_top_orig = [int(i) for i in sel[order]]
        rankings_ordered.append(idx_top_orig)
        for pos, idx_orig in enumerate(idx_top_orig):
            posiciones_por_ticket.setdefault(idx_orig, []).append(pos)

    # Jaccard@K para cada K solicitado, entre réplicas consecutivas.
    jaccard_acumulado = {k: [] for k in jaccard_k_values}
    rbo_acumulado = {p: [] for p in rbo_p_values}

    for b in range(1, n_boot):
        a_full = rankings_ordered[b - 1]
        c_full = rankings_ordered[b]
        for k in jaccard_k_values:
            a_set = set(a_full[:k])
            c_set = set(c_full[:k])
            union = a_set | c_set
            if union:
                jaccard_acumulado[k].append(len(a_set & c_set) / len(union))
        for p in rbo_p_values:
            rbo_acumulado[p].append(_rbo_extrapolated(a_full, c_full, p=p))

    n_inestables = 0
    for idx, posiciones in posiciones_por_ticket.items():
        if len(posiciones) >= 0.5 * n_boot:
            if np.std(posiciones) > 1.0:
                n_inestables += 1

    out = {'modelo': nombre, 'n_boot': n_boot,
           'n_tickets_inestables': int(n_inestables)}
    for k in jaccard_k_values:
        arr = np.asarray(jaccard_acumulado[k]) if jaccard_acumulado[k] else np.array([np.nan])
        out[f'jaccard_at{k}'] = float(np.nanmean(arr))
    for p in rbo_p_values:
        arr = np.asarray(rbo_acumulado[p]) if rbo_acumulado[p] else np.array([np.nan])
        out[f'rbo_p{int(p * 10):02d}'] = float(np.nanmean(arr))
    return out

def ranking_puro(proba, seed: int = 42) -> np.ndarray:
    """Ordenamiento descendente reproducible con desempate aleatorio uniforme.

    Devuelve los índices (0-indexados) de mayor a menor probabilidad. El
    ruido `U(0, 1e-9)` rompe empates de forma determinista bajo la misma
    semilla. Canonicaliza el helper que vivía duplicado en los scripts
    `scripts/avance_final_t21_s1_ambiguedad.py` y
    `scripts/avance_final_t21_s3v2_ruptura.py` (sus copias inline se
    mantienen porque sus outputs ya están hasheados en `manifest_t21.json`).

    Cita: Voorhees 2001 (TREC) — convención estándar para ranking
    determinista con tiebreak reproducible.
    """
    proba = np.asarray(proba, dtype=float)
    rng = np.random.default_rng(seed)
    ruido = rng.uniform(0, 1e-9, size=len(proba))
    score_aux = proba + ruido
    return np.argsort(score_aux)[::-1]

def mean_reciprocal_rank_nivel3(y_true_t1, y_true_t2, proba_t1,
                                 score_combinado=None, seed: int = 42) -> dict:
    """MRR dual: T1 puro vs Crítico binario + Sistema combinado vs Nivel 3.

    Resuelve el sesgo de evaluación detectado en el plan revisión 3: si se
    evaluara `proba_t1` directamente contra Nivel 3 (Crítico ∩ Emergencia),
    se penalizaría injustamente al modelo T1 por no captar la dimensión T2
    (Emergencia) que matemáticamente es invisible al extractor binario.

    - `mrr_t1_puro` mide al EXTRACTOR: ordena por `proba_t1` y busca la
      posición del primer Crítico real. Mide capacidad de detección binaria.
    - `mrr_sistema` mide al PRODUCTO FINAL: ordena por `score_combinado`
      (matriz ordinal + desempate ε·proba) y busca la posición del primer
      Nivel 3 real. Mide el impacto operativo del ranking al operador NOC.

    Cita: Voorhees 1999 "The TREC-8 Question Answering Track Report".

    Args:
        y_true_t1: (n,) {0,1} target Crítico.
        y_true_t2: (n,) {0,1,2,3} target T2 codificado.
        proba_t1: (n,) [0,1] probabilidad calibrada de Crítico.
        score_combinado: (n,) score del sistema; si es None, mrr_sistema=NaN.
        seed: reproducibilidad del desempate.

    Returns:
        dict con `mrr_t1_puro` y `mrr_sistema`. NaN si no hay positivos.
    """
    y_true_t1 = np.asarray(y_true_t1)
    y_true_t2 = np.asarray(y_true_t2)
    proba_t1 = np.asarray(proba_t1, dtype=float)
    is_nivel3 = (y_true_t1 == 1) & (y_true_t2 == 2)

    # T1 puro contra Crítico binario
    if not np.any(y_true_t1 == 1):
        mrr_t1 = float('nan')
    else:
        idx_t1 = ranking_puro(proba_t1, seed=seed)
        pos_crit = np.where(y_true_t1[idx_t1] == 1)[0]
        mrr_t1 = 1.0 / (pos_crit[0] + 1) if len(pos_crit) > 0 else 0.0

    # Sistema combinado contra Nivel 3
    if score_combinado is None or not np.any(is_nivel3):
        mrr_sys = float('nan')
    else:
        sc = np.asarray(score_combinado, dtype=float)
        idx_sys = ranking_puro(sc, seed=seed)
        pos_n3 = np.where(is_nivel3[idx_sys])[0]
        mrr_sys = 1.0 / (pos_n3[0] + 1) if len(pos_n3) > 0 else 0.0

    return {'mrr_t1_puro': float(mrr_t1) if not np.isnan(mrr_t1) else float('nan'),
            'mrr_sistema': float(mrr_sys) if not np.isnan(mrr_sys) else float('nan')}

def population_stability_index(distrib_a, distrib_b, bins: int = 10) -> dict:
    """PSI con fixed-width binning sobre [0,1] y suavizado Laplace 1e-6.

    Reemplaza el binning por cuantiles que colapsa con distribuciones
    calibradas isotónicamente (donde las probabilidades se acumulan en
    valores específicos cercanos a la prevalencia, generando intervalos
    de ancho cero). Los 10 contenedores fijos de ancho 0.1 garantizan
    estabilidad numérica y permiten interpretación directa con los
    umbrales clásicos de la literatura.

    Interpretación canónica (Siddiqi 2006):
        PSI < 0.10           : distribución estable.
        0.10 ≤ PSI < 0.25    : cambio moderado, requiere monitoreo.
        PSI ≥ 0.25           : cambio significativo, requiere acción.

    Suavizado Laplace 1e-6 evita `log(0)`. Sesgo residual: el PSI queda
    ligeramente sub-estimado (más conservador) cuando hay bins vacíos
    reales; preferible al colapso numérico.

    Cita: Yurdakul 2018 "Statistical Properties of Population Stability
    Index"; Siddiqi 2006 "Credit Risk Scorecards" (capítulo de
    monitoreo).
    """
    distrib_a = np.asarray(distrib_a, dtype=float)
    distrib_b = np.asarray(distrib_b, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    freq_a, _ = np.histogram(distrib_a, bins=bin_edges)
    freq_b, _ = np.histogram(distrib_b, bins=bin_edges)

    n_a = max(len(distrib_a), 1)
    n_b = max(len(distrib_b), 1)
    prop_a = freq_a / n_a
    prop_b = freq_b / n_b
    prop_a = np.where(prop_a == 0, 1e-6, prop_a)
    prop_b = np.where(prop_b == 0, 1e-6, prop_b)

    aportes = (prop_a - prop_b) * np.log(prop_a / prop_b)
    psi_val = float(np.sum(aportes))

    if psi_val < 0.10:
        interpretacion = 'estable'
    elif psi_val < 0.25:
        interpretacion = 'cambio_moderado'
    else:
        interpretacion = 'cambio_significativo'

    return {
        'psi': psi_val,
        'interpretacion': interpretacion,
        'bin_edges': bin_edges.tolist(),
        'freq_a': freq_a.tolist(),
        'freq_b': freq_b.tolist(),
        'aportes_por_bin': aportes.tolist(),
    }

def bayes_optimal_threshold(prevalencia: float, ratio_cfn_cfp: float) -> float:
    """Umbral de decisión óptimo bajo función de pérdida asimétrica.

    Fórmula: `L* = 1 / (1 + (C_FN/C_FP) · (1 - π) / π)`, donde π es la
    prevalencia de la clase positiva y C_FN/C_FP es la razón entre el
    costo de un falso negativo y el de un falso positivo. Reemplaza al
    threshold operativo `Recall|Precision≥0.15` cuando éste colapsa en
    régimen de baja prevalencia.

    Cita: Elkan 2001 "The Foundations of Cost-Sensitive Learning"
    (Proc. IJCAI), Sec. 2 ("Making optimal decisions").
    """
    if prevalencia <= 0.0 or prevalencia >= 1.0:
        return 0.5
    cost_ratio = float(ratio_cfn_cfp)
    return 1.0 / (1.0 + cost_ratio * ((1.0 - prevalencia) / prevalencia))

def mean_average_precision(y_true, proba, seed: int = 42) -> float:
    """Average Precision (AP) sobre ordenamiento reproducible.

    Soporta uso dual:
        - MAP puro: `y_true = Crítico T1`, `proba = proba_t1`.
        - MAP sistema: `y_true = Nivel 3`, `proba = score_combinado`.

    AP integra Precisión sobre todos los puntos de Recall del ranking
    (Recommender Systems Handbook 2022, capítulo de evaluación). Devuelve
    NaN si no hay positivos.

    Cita: Burges et al 2005 "Learning to Rank using Gradient Descent";
    Voorhees 2001 (TREC) para formalización IR estándar.
    """
    y_true = np.asarray(y_true)
    proba = np.asarray(proba, dtype=float)
    if not np.any(y_true == 1):
        return float('nan')
    idx_orden = ranking_puro(proba, seed=seed)
    return float(average_precision_score(y_true[idx_orden], proba[idx_orden]))

def precision_at_k(y_true, proba, k: int, seed: int = 42) -> float:
    """Precision@K sobre ranking reproducible.

    Complementa a `recall_at_k_global` y soporta uso dual eligiendo
    apropiadamente `y_true` y `proba`:
        - puro: `y_true = Crítico T1`, `proba = proba_t1`.
        - sistema: `y_true = Nivel 3`, `proba = score_combinado`.

    Cita: Recommender Systems Handbook 2022 (Ricci et al.), Cap. 8
    ("Evaluating Recommender Systems").
    """
    proba = np.asarray(proba, dtype=float)
    y_true = np.asarray(y_true)
    if len(proba) == 0 or k <= 0:
        return float('nan')
    idx_orden = ranking_puro(proba, seed=seed)[:k]
    return float(np.mean(y_true[idx_orden] == 1))

def intra_list_similarity(top_k_indices, categorias_array) -> float:
    """Intra-List Similarity (ILS) con descuento por rango estilo NDCG.

    Mide la concentración del top-K en una sola Categoría. Similitud
    pairwise = 1 si dos tickets comparten Categoría, 0 si no. El
    descuento `1 / (log2(i+2) · log2(j+2))` evita el colapso a Δ=0 en
    ventanas con `n ≤ K` donde ambos modelos llevan los mismos items
    al top (la versión clásica set-based de Ziegler 2005 daría el
    mismo ILS para los dos órdenes — la rank-biased distingue el
    orden interno).

    Cita: Ziegler et al 2005 "Improving Recommendation Lists Through
    Topic Diversification" (WWW 2005); adaptación con descuento NDCG
    propia del proyecto (justificada en bitácora PASO 21 adendo).

    Args:
        top_k_indices: lista o array de índices en el orden del ranking.
        categorias_array: vector global de Categorías por índice.

    Returns:
        ILS rank-biased ∈ [0, 1]. Mayor = más concentrado en una sola
        Categoría (menos diverso).
    """
    top = np.asarray(top_k_indices, dtype=int)
    k = len(top)
    if k <= 1:
        return 0.0
    cats = np.asarray(categorias_array)[top]
    sim_acumulada = 0.0
    peso_acumulado = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            descuento = 1.0 / (np.log2(i + 2) * np.log2(j + 2))
            sim = 1.0 if cats[i] == cats[j] else 0.0
            sim_acumulada += sim * descuento
            peso_acumulado += descuento
    return float(sim_acumulada / peso_acumulado) if peso_acumulado > 0 else 0.0

def novedad_vargas_castells(top_k_indices, popularidad_dict) -> float:
    """Novedad rank-biased de Vargas-Castells con descuento NDCG.

    Cada item aporta su self-information `-log2(p_cat)` donde p_cat es
    la frecuencia de su Categoría en train. El descuento `1/log2(rank+2)`
    pondera más los primeros puestos, alineado con el costo operativo
    del NOC (un item raro en posición 1 vale más que en posición 5).

    Cita: Vargas & Castells 2011 "Rank and Relevance in Novelty and
    Diversity Metrics for Recommender Systems" (RecSys 2011); descuento
    NDCG-style propio del proyecto (bitácora PASO 21 adendo) para
    soportar comparación en ventanas con baja densidad.

    Args:
        top_k_indices: lista o array de índices en el orden del ranking.
        popularidad_dict: dict {idx_ticket -> frecuencia categoría en train}.
            Valores faltantes se asumen muy raros (1e-6).

    Returns:
        Novedad rank-biased ≥ 0. Mayor = más rescate de cola larga.
    """
    top = np.asarray(top_k_indices, dtype=int)
    k = len(top)
    if k == 0:
        return 0.0
    novedad_acumulada = 0.0
    peso_acumulado = 0.0
    for rank, idx in enumerate(top):
        p = popularidad_dict.get(int(idx), 1e-6)
        self_info = -np.log2(max(p, 1e-6))
        descuento = 1.0 / np.log2(rank + 2)
        novedad_acumulada += self_info * descuento
        peso_acumulado += descuento
    return float(novedad_acumulada / peso_acumulado) if peso_acumulado > 0 else 0.0

def costo_asimetrico(y_true, y_pred, ratio_cfn_cfp: float = 50) -> dict:
    """Pérdida económica asimétrica del NOC: `costo = ratio·FN + FP`.

    FUENTE ÚNICA de la métrica de costo (canon AFG III = 50:1). Un falso negativo
    (un crítico no priorizado) cuesta `ratio_cfn_cfp` veces más que una falsa
    alarma. Cita: Elkan 2001 "The Foundations of Cost-Sensitive Learning" (IJCAI).

    El SLA humano de 5 min es el presupuesto del ingeniero; el costo aquí pondera
    el error de triaje, no la latencia (esa es el piso de 2 s, ver benchmark).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    return {
        "FN": fn,
        "FP": fp,
        "costo": round(float(ratio_cfn_cfp) * fn + fp, 2),
        "ratio_cfn_cfp": float(ratio_cfn_cfp),
    }

def barrido_costo_umbral(y_true, proba, ratios=(50, 10), n_umbral: int = 101) -> pd.DataFrame:
    """Sweep de umbral × ratio para la curva de costo operativo (Gráfico C).

    Para cada umbral en `linspace(0,1,n_umbral)` y cada ratio en `ratios`, computa
    FN, FP y el costo asimétrico (`costo = ratio·FN + FP`). Long-form para graficar
    el *envelope* de costo. Reusa `costo_asimetrico` (H1) por punto.

    Returns:
        DataFrame con columnas {umbral, ratio, FN, FP, costo}.
    """
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    umbrales = np.linspace(0.0, 1.0, n_umbral)
    filas = []
    for u in umbrales:
        pred = (proba >= u).astype(int)
        for r in ratios:
            c = costo_asimetrico(y_true, pred, ratio_cfn_cfp=r)
            filas.append({"umbral": float(u), "ratio": float(r),
                          "FN": c["FN"], "FP": c["FP"], "costo": c["costo"]})
    return pd.DataFrame(filas)

def fdr_at_k(y_true, proba, k: int = 5, seed: int = 42) -> float:
    """Tasa de Descubrimiento Falso en la cabeza de la cola: `FDR@k = 1 - P@k`.

    Mide la fracción de los `k` tickets en la cima de la cola de prioridad que NO
    son verdaderos positivos. Reusa `precision_at_k`. Insumo del gatillo MLOps
    (FDR@5 > 30%, auditoría retrospectiva) y de la fatiga por alertas del operador.
    """
    p = precision_at_k(y_true, proba, k=k, seed=seed)
    if np.isnan(p):
        return float("nan")
    return float(1.0 - p)

def indice_reduccion_esfuerzo(
    y_true_t1, y_true_t2,
    pred_t1_a, pred_t2_a, proba_t1_a,
    pred_t1_b, pred_t2_b, proba_t1_b,
    seg_por_ticket: float = 45.0,
    idx_emergencia: int = 2,
    seed: int = 42,
) -> dict:
    """Índice de reducción de esfuerzo HITL: cuántas posiciones sube B los Nivel-3
    reales frente a A, traducido a horas de ingeniería ahorradas.

    Para cada ticket Nivel-3 real (Crítico AND Emergencia), su rango (posición en
    la cola global) bajo el `compute_score_combinado` de A y de B. `posiciones_subidas`
    = Σ max(0, rank_A − rank_B) (B lo sube si lo ubica más arriba = menor índice).
    `horas_ahorradas = posiciones_subidas · seg_por_ticket / 3600` (45 s por ticket
    mal ubicado que un operador abre y descarta).

    CAVEAT: estimación puntual; el lift de un retador semántico sobre RF NO es
    significativo (su IC incluye 0), así que el ahorro se reporta con esa salvedad,
    no como ahorro garantizado.
    """
    y_grad = compute_prioridad_ordinal(y_true_t1, y_true_t2, idx_emergencia)
    nivel3 = np.where(y_grad == 3)[0]
    n3 = int(len(nivel3))
    if n3 == 0:
        return {"posiciones_subidas": 0, "horas_ahorradas": 0.0, "n_nivel3": 0}

    score_a = compute_score_combinado(pred_t1_a, pred_t2_a, proba_t1_a,
                                      idx_emergencia=idx_emergencia, seed=seed)
    score_b = compute_score_combinado(pred_t1_b, pred_t2_b, proba_t1_b,
                                      idx_emergencia=idx_emergencia, seed=seed)
    # rank[i] = posición (0 = cima) del ticket i en el orden descendente del score.
    rank_a = np.empty(len(score_a), dtype=int)
    rank_a[np.argsort(-score_a)] = np.arange(len(score_a))
    rank_b = np.empty(len(score_b), dtype=int)
    rank_b[np.argsort(-score_b)] = np.arange(len(score_b))

    subidas = int(np.maximum(0, rank_a[nivel3] - rank_b[nivel3]).sum())
    horas = float(subidas * seg_por_ticket / 3600.0)
    return {"posiciones_subidas": subidas, "horas_ahorradas": round(horas, 3), "n_nivel3": n3}

def bootstrap_costo_pareado(
    y_true, proba_a, proba_b, umbral_a, umbral_b,
    ratio_cfn_cfp: float = 50, n_boot: int = 1000, seed: int = 42,
) -> dict:
    """Bootstrap pareado del costo asimétrico (Δcosto B−A) con IC95 + p.

    Re-muestreo de TICKETS con reemplazo. Los **umbrales se fijan A PRIORI** (se
    pasan como argumento, derivados del bloque de validación temporal) y NUNCA se
    re-sintonizan dentro del loop — re-tunear contaminaría y daría un IC95
    optimista. Coherente con "medir, no simular" (igual que el P99 y el Δ-AUC-PR).

    Returns:
        dict con costo puntual A/B, Δcosto mean+IC95, p (proporción Δ ≥ 0 = B más
        caro), delta_ic_incluye_0.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    pred_a = (np.asarray(proba_a, dtype=float) >= umbral_a).astype(int)
    pred_b = (np.asarray(proba_b, dtype=float) >= umbral_b).astype(int)
    n = len(y)
    costo_a_full = costo_asimetrico(y, pred_a, ratio_cfn_cfp)["costo"]
    costo_b_full = costo_asimetrico(y, pred_b, ratio_cfn_cfp)["costo"]

    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ca = ratio_cfn_cfp * ((y[idx] == 1) & (pred_a[idx] == 0)).sum() + ((y[idx] == 0) & (pred_a[idx] == 1)).sum()
        cb = ratio_cfn_cfp * ((y[idx] == 1) & (pred_b[idx] == 0)).sum() + ((y[idx] == 0) & (pred_b[idx] == 1)).sum()
        deltas[b] = cb - ca
    ic_low = float(np.percentile(deltas, 2.5))
    ic_high = float(np.percentile(deltas, 97.5))
    return {
        "costo_a": costo_a_full,
        "costo_b": costo_b_full,
        "delta_costo_mean": float(deltas.mean()),
        "delta_ic_low": ic_low,
        "delta_ic_high": ic_high,
        "delta_ic_incluye_0": bool(ic_low <= 0 <= ic_high),
        "p_value_unilateral_B_mas_caro": float((deltas >= 0).mean()),
        "ratio_cfn_cfp": float(ratio_cfn_cfp),
        "n_boot": n_boot,
    }

def bootstrap_delta_aucpr_pareado(
    y_true, proba_a, proba_b, n_boot: int = 1000, seed: int = 42,
) -> dict:
    """LA MEDICIÓN DECISIVA — Δ-AUC-PR(B−A) con IC95 + p unilateral.

    Bootstrap pareado a nivel TICKET (mismos índices para A y B en cada réplica,
    controla la varianza compartida). Decide la rama A/B/C del veredicto: el torneo
    canónico solo midió la significancia del Δ-NDCG, NUNCA la del Δ-AUC-PR, así que
    este número no existía. Si IC95 excluye 0 por el lado +, el retador demostró
    superioridad en discriminación.

    Returns:
        dict con auc_pr_a/b puntual, delta mean+IC95, p (proporción Δ ≤ 0),
        delta_ic_incluye_0.
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    pa = np.asarray(proba_a, dtype=float)
    pb = np.asarray(proba_b, dtype=float)
    n = len(y)
    auc_a_full = float(average_precision_score(y, pa)) if y.sum() > 0 else float("nan")
    auc_b_full = float(average_precision_score(y, pb)) if y.sum() > 0 else float("nan")

    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yi = y[idx]
        if yi.sum() == 0 or yi.sum() == len(yi):
            continue  # réplica degenerada (sin positivos o sin negativos): AP indefinida
        a = average_precision_score(yi, pa[idx])
        b = average_precision_score(yi, pb[idx])
        deltas.append(b - a)
    arr = np.asarray(deltas, dtype=float)
    if len(arr) == 0:
        return {"auc_pr_a": auc_a_full, "auc_pr_b": auc_b_full, "n_validas": 0}
    ic_low = float(np.percentile(arr, 2.5))
    ic_high = float(np.percentile(arr, 97.5))
    return {
        "auc_pr_a": round(auc_a_full, 4),
        "auc_pr_b": round(auc_b_full, 4),
        "delta_mean": float(arr.mean()),
        "delta_ic_low": ic_low,
        "delta_ic_high": ic_high,
        "delta_ic_incluye_0": bool(ic_low <= 0 <= ic_high),
        "delta_significativo_positivo": bool(ic_low > 0),
        "p_value_unilateral_B_mejor_A": float((arr <= 0).mean()),
        "n_validas": int(len(arr)),
        "n_boot": n_boot,
    }

def mdes_poder_simulacion(
    y_true, proba_base, umbral=None, metrica: str = "aucpr",
    efectos=None, proba_alt=None, ratio_cfn_cfp: float = 50,
    n_boot: int = 300, n_sim: int = 120, poder_objetivo: float = 0.80, seed: int = 42,
) -> dict:
    """Tamaño de Efecto Mínimo Detectable (MDES) / poder POR SIMULACIÓN.

    NO hay cierre paramétrico para el Δ-AUC-PR con 12 positivos, así que el poder de
    un test bootstrap se estima por **doble bootstrap**: se inyecta un efecto CONOCIDO,
    se re-muestrea el dataset (bootstrap externo = nuevo dato bajo el mismo DGP) y se
    mide con qué frecuencia el test interno declara la diferencia significativa. El
    MDES es el menor efecto con poder ≥ `poder_objetivo`.

    **`metrica='aucpr'` (HEADLINE, lo que de verdad importa):** el retador se sintetiza
    ESCALANDO la dirección REAL del retador semántico (`proba_alt`, p.ej. e5): retador_α =
    base + α·(alt − base), recortado a [0,1]. Esto PRESERVA la estructura de ruido real
    del Δ-AUC-PR a n=12 (a diferencia de un boost uniforme limpio, que la subestimaría).
    α=1 es el e5 real (Δ≈+0.083); se mide cuánto hay que AMPLIFICARLO para detectarlo. A
    n=12 críticos el poder en α=1 es BAJO (consistente con el IC95 observado que incluye 0):
    se necesita α≈2–3 (un Δ-AUC-PR ~2–3× el real) para 80% de poder. Si `proba_alt=None`,
    cae a un boost de críticos hacia 1 (`lift`), menos fiel y solo ilustrativo.

    **`metrica='costo'` (CONTRAPUNTO, bien-potenciado):** el costo está dominado por los
    falsos positivos (abundantes), así que su MDES es chico (poder alto). El CONTRASTE
    entre ambas curvas es el argumento honesto: *el costo se detecta, la detección de
    los 12 críticos NO* — por eso nada es significativo en discriminación aunque el costo
    sí lo sea. Responde a "¿por qué nada es significativo si su MDES de costo da poder 1?"
    """
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true).astype(int)
    p = np.asarray(proba_base, dtype=float)
    n = len(y)

    if metrica == "aucpr":
        crit = (y == 1)
        usar_real = proba_alt is not None
        alt = np.asarray(proba_alt, dtype=float) if usar_real else None
        # Escalas α (sobre la dirección real) o lifts (fallback). α=1 ≈ el retador real.
        escalas = efectos if efectos is not None else (
            (0.5, 1.0, 1.5, 2.0, 3.0) if usar_real else (0.05, 0.10, 0.20, 0.35, 0.55))
        curva = []
        for esc in escalas:
            if usar_real:
                p_ch = np.clip(p + float(esc) * (alt - p), 0.0, 1.0)  # amplifica el efecto REAL
            else:
                p_ch = p.copy()
                p_ch[crit] = p[crit] + float(esc) * (1.0 - p[crit])
            delta_true = float(average_precision_score(y, p_ch) - average_precision_score(y, p))
            det = 0
            for _ in range(n_sim):
                oidx = rng.integers(0, n, size=n)  # bootstrap externo = nuevo dataset bajo el DGP
                if y[oidx].sum() < 2 or y[oidx].sum() == len(oidx):
                    continue
                res = bootstrap_delta_aucpr_pareado(y[oidx], p[oidx], p_ch[oidx],
                                                    n_boot=n_boot, seed=int(rng.integers(1, 1_000_000_000)))
                if res.get("delta_significativo_positivo"):
                    det += 1
            curva.append({"escala": float(esc), "delta_aucpr": round(delta_true, 4),
                          "poder": round(det / n_sim, 3), "es_retador_real": bool(usar_real and abs(esc - 1.0) < 1e-9)})
        curva = sorted(curva, key=lambda r: r["delta_aucpr"])
        mdes = next((r["delta_aucpr"] for r in curva if r["poder"] >= poder_objetivo), None)
        return {
            "metrica": "aucpr",
            "modo": "escala_direccion_real" if usar_real else "lift_critico",
            "curva": curva,
            "mdes_aucpr": mdes,
            "poder_objetivo": poder_objetivo,
            "nota": ("MDES = menor Δ-AUC-PR detectable al poder objetivo (doble bootstrap sobre la "
                     "dirección REAL del retador, n=12 críticos). α=1 ≈ el retador real."),
        }

    # --- contrapunto: costo (bien-potenciado por FP abundantes) ---
    efectos_c = efectos if efectos is not None else (5, 20, 40, 80, 120)
    pred_base = (p >= (umbral if umbral is not None else 0.5)).astype(int)
    fp_idx = np.where((y == 0) & (pred_base == 1))[0]
    poder_por_efecto = {}
    for e in efectos_c:
        e_eff = min(int(e), len(fp_idx))
        if e_eff == 0:
            poder_por_efecto[int(e)] = 0.0
            continue
        det = 0
        for _ in range(n_sim):
            pred_ret = pred_base.copy()
            apagar = rng.choice(fp_idx, size=e_eff, replace=False)
            pred_ret[apagar] = 0
            res = _bootstrap_costo_pred_pareado(y, pred_base, pred_ret, ratio_cfn_cfp, n_boot=n_boot, rng=rng)
            if not res["delta_ic_incluye_0"]:
                det += 1
        poder_por_efecto[int(e)] = round(det / n_sim, 3)
    mdes = next((e for e in sorted(poder_por_efecto) if poder_por_efecto[e] >= poder_objetivo), None)
    return {
        "metrica": "costo",
        "ratio_cfn_cfp": float(ratio_cfn_cfp),
        "poder_por_efecto": poder_por_efecto,
        "mdes_costo": mdes,
        "poder_objetivo": poder_objetivo,
        "nota": "Costo bien-potenciado (FP abundantes): MDES chico. Contrapunto del AUC-PR.",
    }

def _bootstrap_costo_pred_pareado(y, pred_a, pred_b, ratio_cfn_cfp, n_boot, rng):
    """Núcleo bootstrap del costo sobre predicciones duras (para H8). Interno."""
    n = len(y)
    deltas = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ca = ratio_cfn_cfp * ((y[idx] == 1) & (pred_a[idx] == 0)).sum() + ((y[idx] == 0) & (pred_a[idx] == 1)).sum()
        cb = ratio_cfn_cfp * ((y[idx] == 1) & (pred_b[idx] == 0)).sum() + ((y[idx] == 0) & (pred_b[idx] == 1)).sum()
        deltas[b] = cb - ca
    lo = np.percentile(deltas, 2.5)
    hi = np.percentile(deltas, 97.5)
    return {"delta_ic_incluye_0": bool(lo <= 0 <= hi)}

def flip_ndcg_cross_arm(
    y_true_t1, y_true_t2,
    pred_t1_iso, pred_t1_platt, pred_t2,
    proba_t1_iso, proba_t1_platt, dates_test,
    umbral_ruido_ndcg=None, tau_hitl: float = 0.0924, jaccard_min: float = 0.80,
    k: int = 5, idx_emergencia: int = 2, dias_ventana: int = 7,
) -> dict:
    """Estabilidad del RANKING bajo cambio de calibrador (iso ↔ Platt) — Gráfico E.

    El gate NO es sobre discriminación (AUC-PR) sino sobre RANKING: la cola y el
    HITL consumen el ranking calibrado. Si el NDCG@k del producto final flipea al
    cambiar iso↔Platt, la cola es impredecible en producción (el mismo ticket cae
    a un lado u otro del umbral HITL según el calibrador).

    UMBRAL DE FLIP PRE-DECLARADO (no cualitativo, misma vara para todos):
        inestable = (flip > umbral_ruido_ndcg)  [criterio auto-referencial: se
                     mueve más que su propio ruido de medición, el ancho IC95 del
                     NDCG del modelo, pasado por el caller via bootstrap_ic_producto_final]
                    OR (jaccard_hitl < jaccard_min)  [criterio operativo: el set
                     auto-ruteado por HITL (proba ≥ τ) cambia > (1-jaccard_min) entre brazos].

    Returns:
        dict con ndcg_iso, ndcg_platt, flip, supera_ruido, jaccard_hitl,
        jaccard_bajo_umbral, inestable.
    """
    res_iso = evaluar_producto_final(
        nombre="iso", y_true_t1=y_true_t1, y_true_t2=y_true_t2,
        pred_t1=pred_t1_iso, pred_t2=pred_t2, proba_t1=proba_t1_iso,
        dates_test=dates_test, k=k, idx_emergencia=idx_emergencia, dias_ventana=dias_ventana,
    )
    res_platt = evaluar_producto_final(
        nombre="platt", y_true_t1=y_true_t1, y_true_t2=y_true_t2,
        pred_t1=pred_t1_platt, pred_t2=pred_t2, proba_t1=proba_t1_platt,
        dates_test=dates_test, k=k, idx_emergencia=idx_emergencia, dias_ventana=dias_ventana,
    )
    key = f"ndcg@{k}_pf_{dias_ventana}d_PRIMARIA"
    ndcg_iso = float(res_iso[key])
    ndcg_platt = float(res_platt[key])
    flip = abs(ndcg_iso - ndcg_platt)

    set_iso = set(np.where(np.asarray(proba_t1_iso, dtype=float) >= tau_hitl)[0].tolist())
    set_platt = set(np.where(np.asarray(proba_t1_platt, dtype=float) >= tau_hitl)[0].tolist())
    union = set_iso | set_platt
    jaccard = float(len(set_iso & set_platt) / len(union)) if union else 1.0

    supera_ruido = None if umbral_ruido_ndcg is None else bool(flip > umbral_ruido_ndcg)
    jaccard_bajo = bool(jaccard < jaccard_min)
    inestable = bool((supera_ruido is True) or jaccard_bajo)
    return {
        "ndcg_iso": round(ndcg_iso, 4),
        "ndcg_platt": round(ndcg_platt, 4),
        "flip": round(flip, 4),
        "umbral_ruido_ndcg": (None if umbral_ruido_ndcg is None else round(float(umbral_ruido_ndcg), 4)),
        "supera_ruido": supera_ruido,
        "jaccard_hitl": round(jaccard, 4),
        "jaccard_min": jaccard_min,
        "jaccard_bajo_umbral": jaccard_bajo,
        "inestable": inestable,
    }
