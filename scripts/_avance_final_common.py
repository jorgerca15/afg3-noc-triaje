"""Utilidades compartidas para los scripts del Avance Final AFG II.

Centraliza la reconstrucción del test set y los mapeos de targets para que todos
los scripts lean exactamente el mismo orden temporal sobre fechas correctas.

Parseo de fechas (CORREGIDO 2026-06-21): `format='mixed', dayfirst=False`. Los
datos de `Fecha aviso incidencia` están en **formato US (mes-primero)** —
verificado sobre el crudo: 354/588 filas tienen el 2º campo >12 (sólo legible
como día) y **cero** filas tienen el 1º campo >12 (no hay día-primero); la
convención es única. El parser previo (`dayfirst=True`) invertía mes/día en ~37%
de las filas y revolvía el orden del split temporal (test 12 vs 11 críticos).
CONSECUENCIA: las predicciones/embeddings `.npy` generadas con el parser viejo
quedan **invalidadas**; se regeneran en la tarea posterior de modelado sobre este
split correcto. Ningún número del modelado es válido hasta esa regeneración.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CSV = ROOT / "data" / "processed" / "noc-tickets_clean_v2.csv"
PRED = ROOT / "data" / "processed" / "predicciones"

MAPEO_TIPO_AGRUPADO = {
    "Correctivo": "Correctivo",
    "Emergencia": "Emergencia",
    "Inspección": "Preventivo",
    "Preventivo": "Preventivo",
    "Soporte": "Incidente",
    "Consumo de energía": "Incidente",
    "Incidencia": "Incidente",
    "Quiebre Técnico": "Incidente",
    "Homologacion": "Incidente",
}
T2_LABELS = ["Correctivo", "Preventivo", "Emergencia", "Incidente"]
T2_TO_IDX = {lbl: i for i, lbl in enumerate(T2_LABELS)}

NOMBRE_LEGIBLE = {
    "dummy": "N0 · Dummy",
    "reglas": "N0 · Reglas",
    "n1_xgb": "N1 · XGB tabular",
    "n2_xgb": "N2 · XGB multimodal",
    "beto_lr": "N3 · BETO + LR",
    "beto_xgb": "N3 · BETO + XGB",
    "beto_concat": "N3 · BETO + estructura + LR",
}


def cargar_split_temporal_canonico() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split temporal 60/40 sobre fechas correctas (US, dayfirst=False).

    Retorna (df_train, df_test) por corte posicional tras ordenar por fecha.
    El parseo US es el correcto (ver docstring del módulo); el split se re-deriva
    sobre el orden temporal real. Tras el sort se valida sanidad temporal (NaT,
    rango plausible y monotonía) para cerrar el resquicio del error silencioso
    (una fecha válida pero con mes/día invertido, que NaT no detectaría).
    """
    df_raw = pd.read_csv(CSV)
    df = df_raw.rename(
        columns={
            "Fecha aviso incidencia_final": "Fecha aviso incidencia",
            "Criticidad Incidencia_final": "Criticidad",
            "Tipo de trabajo_final": "Tipo de trabajo",
            "Descripcion_final": "Descripcion",
            "Cantidad de clientes afectados_final": "Cantidad_clientes_afectados",
            "Categoria de incidencia_final": "Categoría de incidencia",
            "Clientes afectados_final": "clientes_afectados_flag",
            "missing_Cantidad de clientes afectados_final": "clientes_faltante_flag",
            "Tipo de producto_final": "Tipo de producto",
            "Operador_final": "Operador",
            "Distrito_final": "Distrito",
            "tipo_cliente_final": "tipo_cliente",
        }
    )
    mask = (df["Cantidad_clientes_afectados"] == 591) & (df["Criticidad"] == "Leve")
    df.loc[mask, "Criticidad"] = "Crítico"

    df["Fecha aviso incidencia"] = pd.to_datetime(
        df["Fecha aviso incidencia"], format="mixed", dayfirst=False
    )

    df["target_criticidad"] = (
        df["Criticidad"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["crítico", "critico"])
    ).astype(int)
    df["tipo_trabajo_agrupado"] = df["Tipo de trabajo"].map(MAPEO_TIPO_AGRUPADO)
    if df["tipo_trabajo_agrupado"].isna().any():
        valores = df.loc[df["tipo_trabajo_agrupado"].isna(), "Tipo de trabajo"].unique()
        raise ValueError(f"Tipo de trabajo sin mapeo: {valores}")
    df["target_t2"] = df["tipo_trabajo_agrupado"].map(T2_TO_IDX).astype(int)

    df = df.sort_values("Fecha aviso incidencia").reset_index(drop=True)

    # --- Sanidad temporal tras el fix de parseo (cierra el error silencioso) ---
    # NaT no detecta una fecha "válida pero equivocada" (mes/día invertido sigue
    # siendo una fecha legal). Por eso, además de NaT==0, validamos rango plausible
    # (sin futuro imposible) y monotonía estricta del orden que define el split.
    # Nota: NaT==0 y la monotonía son guardas invariantes en el tiempo; el rango
    # (now+1d) es relativo a la fecha de corrida — caza un misparse que produzca un
    # max futuro (hoy lo hace: dayfirst=True daría max 2026-12-01 > now), pero su
    # poder decae si se corre después de esa fecha. Los tres juntos son la red.
    _f = df["Fecha aviso incidencia"]
    assert _f.isna().sum() == 0, "NaT tras parseo US (dayfirst=False): revisar formato de fecha"
    assert _f.is_monotonic_increasing, (
        "El split exige orden temporal: la serie no quedó monótona tras el sort"
    )
    _tope = pd.Timestamp.now() + pd.Timedelta(days=1)
    assert pd.Timestamp("2022-01-01") <= _f.min() and _f.max() <= _tope, (
        f"Rango temporal implausible [{_f.min()} … {_f.max()}]: revisar dayfirst/format"
    )

    n_train = int(len(df) * 0.60)
    return df.iloc[:n_train].reset_index(drop=True), df.iloc[n_train:].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Embargo temporal (AFG III · C2) — purga lado-TRAIN para cerrar target-leakage
# ---------------------------------------------------------------------------
# Hallazgo verificado (afg3_baselines_clasicos.construir_features): las features
# son intrínsecas por-ticket en t=0 (sin agregación/rezago/rolling/groupby), así
# que max_lookback = 0 y gap = max(ventana_eval, max_lookback) = ventana_eval = 7.
# El embargo NO cierra leakage por agregación (no existe), pero SÍ cierra leakage
# por maduración de etiqueta: un ticket de train abierto dentro de `gap` días
# antes del punto de split consolida su criticidad real en territorio de test.
# Se purga del lado TRAIN (preserva el test intacto y, con ello, sus 11 críticos
# observados y la potencia estadística; verificado sobre fechas correctas
# (dayfirst=False): train-side 352->345 (7 purgados), test 11 críticos).
MAX_LOOKBACK_DIAS = 0
VENTANA_EVAL_DIAS = 7
# 11 = conteo OBSERVADO de críticos en el test del split canónico sobre fechas
# CORRECTAS (dayfirst=False + corrección fila 591), NO un piso pre-registrado.
# Reconciliación de conteos (todos con NaT=0):
#   · total crudo (sin corrección 591) ............ 87
#   · total target (con flip 591) ................. 88  (15,0%)
#   · test con dayfirst=True+flip  (parser viejo) . 12  ← lo que usó el modelado buggy
#   · test con dayfirst=False+flip (CANÓNICO) ..... 11  ← este valor
#   · test con dayfirst=False, sin flip ........... 10
# El split se fija por densidad de ventana (target-ciego, recompute monótono en
# la auditoría de procedencia de parámetros): este número es RESULTADO del split.
CRITICOS_TEST_OBSERVADOS = 11


def cargar_split_temporal_embargo(embargo_dias: int | None = None,
                                  ventana_eval: int = VENTANA_EVAL_DIAS,
                                  max_lookback: int = MAX_LOOKBACK_DIAS):
    """Split 60/40 canónico + embargo temporal lado-TRAIN (C2).

    gap = embargo_dias si se pasa, si no max(ventana_eval, max_lookback).
    Purga del train los tickets abiertos dentro de `gap` días antes del punto de
    split (sus etiquetas madurarían en territorio de test). Test queda intacto.

    Retorna (df_train_embargado, df_test, diag) donde diag documenta el impacto.
    `criticos_test_observados` es el conteo observado en test (NO un piso): el split
    se fija por densidad de ventana (target-ciego), así que este número es resultado.
    """
    df_train, df_test = cargar_split_temporal_canonico()
    gap = int(embargo_dias) if embargo_dias is not None else max(ventana_eval, max_lookback)
    split_point = df_test["Fecha aviso incidencia"].min()
    corte = split_point - pd.Timedelta(days=gap)

    n_train_0 = len(df_train)
    crit_train_0 = int(df_train["target_criticidad"].sum())
    df_train_emb = df_train[df_train["Fecha aviso incidencia"] <= corte].reset_index(drop=True)
    crit_test = int(df_test["target_criticidad"].sum())

    # Validación de la regla del gap (no número mágico): gap >= max(eval, lookback).
    assert gap >= max(ventana_eval, max_lookback), (
        f"gap={gap} < max(ventana_eval={ventana_eval}, max_lookback={max_lookback})"
    )

    diag = {
        "gap_dias": gap,
        "ventana_eval": ventana_eval,
        "max_lookback": max_lookback,
        "split_point": str(split_point.date()),
        "corte_embargo": str(corte.date()),
        "n_train": [n_train_0, len(df_train_emb)],
        "criticos_train": [crit_train_0, int(df_train_emb["target_criticidad"].sum())],
        "n_test": len(df_test),
        "criticos_test": crit_test,
        "criticos_test_observados": CRITICOS_TEST_OBSERVADOS,
    }
    return df_train_emb, df_test, diag


def cargar_predicciones_config(config: str) -> dict:
    """Carga los 4 arrays .npy de una configuración del manifest."""
    return {
        "pred_t1": np.load(PRED / f"pred_t1_{config}.npy"),
        "pred_t2": np.load(PRED / f"pred_t2_{config}.npy"),
        "proba_t1_raw": np.load(PRED / f"proba_t1_raw_{config}.npy"),
        "proba_t1_cal": np.load(PRED / f"proba_t1_calibrada_{config}.npy"),
    }


def verificar_alineacion(df_test: pd.DataFrame, preds: dict, config: str) -> None:
    n = len(df_test)
    for clave, arr in preds.items():
        if len(arr) != n:
            raise AssertionError(
                f"Config {config} array {clave}: len={len(arr)} ≠ n_test={n}"
            )
