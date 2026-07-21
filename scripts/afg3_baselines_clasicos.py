"""AFG III · Baselines clásicos sobre T1 (RF, LogReg, SVM, NB) — techo estructural aislado.

Cierra el gap de baselines obligatorios de la literatura (Nishad/Fahmy citan
TF-IDF + SVM/LogReg; el .bib cita Random Forest `breiman2001randomforests`).
Comparación JUSTA y AISLADA de clasificadores clásicos sobre el mismo split y
feature set ESTRUCTURAL.

CANON VIGENTE (pivote semántico): el brazo estructural es inerte para T1 (RF
AUC-PR ~0,07, cuyo IC incluye el piso de prevalencia ~0,047); la señal que
discrimina la criticidad vive en la SEMÁNTICA de la descripción t=0 (BETO), no
en las features tabulares. Este script mide el techo estructural para dejarlo
documentado como negativo honesto, no como modelo productivo.

Feature set (firewall del target, CANON EDA v3 — ya aplicado en NUM_COLS/CAT_COLS):
- NUM_COLS = ['n_tokens', 'clientes_faltante_flag'] · CAT_COLS = ['Distrito'].
- EXCLUIDAS como proxies del target: `Categoría de incidencia` (V×Emergencia
  0,63) y `clientes_bin` (V×Emergencia 0,94, 105/107 Emergencia). RETIRADAS como
  confound temporal: `hora`, `mes`, `miss_tipo_producto`. `Distrito` se agrega
  (V×T1 2024+ ≈ 0,19; el EDA la recomendó y faltaba).

Metodología — apples-to-apples entre los clásicos:
- Split temporal con embargo lado-train (`cargar_split_temporal_embargo`).
- Objetivo Optuna = `scorer_operativo` (Recall sujeto a Precision >= 0,15, piso
  operativo del scorer dinámico), idéntico para todos los clásicos.
- CV temporal = `SafeTimeSeriesSplit` (nunca CV aleatorio; los clásicos no
  entienden series de tiempo, se ajustan con el corte de fecha exacto).
- Calibración isotónica temporal (`calibrar_isotonica`, OOF SafeTimeSeriesSplit).
- Producto final: T1 del baseline + T2 común → `evaluar_producto_final`, para
  aislar el efecto de T1.

NOTA HISTÓRICA (pre-pivote, trazabilidad): el bloque `__main__` conserva una
comparación de de-riesgo RF vs. el incumbente tabular XGB-N1 de la iteración
RAMA-B, con su integrity check numérico propio. Ese contraste documenta la línea
estructural superada; el veredicto vigente lo fija la escalera semántica, no esta
comparación tabular.

Presupuesto de tuneo: N_TRIALS (env AFG3_NTRIALS, default 40) para TODOS los
clásicos por igual. AFG3_SMOKE=1 fuerza 3 trials (smoke test).

Salidas:
- data/processed/predicciones/proba_t1_{m}_afg3_{raw,cal}.npy, pred_t1_{m}_afg3.npy
- outputs/afg3_baselines_clasicos.csv
- outputs/afg3_baselines_clasicos.json
- registro en outputs/manifest_afg3.json
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts._avance_final_common import cargar_split_temporal_embargo  # noqa: E402
from src.calibracion import calibrar_isotonica  # noqa: E402
from src.cv import SafeTimeSeriesSplit  # noqa: E402
from src.eval import (  # noqa: E402
    evaluar_binario,
    evaluar_producto_final,
    metricas_calibracion,
    paired_bootstrap_ic_producto_final,
    scorer_operativo,
)
from src.manifest import registrar_artefactos  # noqa: E402

PRED = ROOT / "data" / "processed" / "predicciones"
OUT_DIR = ROOT / "outputs"

RANDOM_STATE = 42
MIN_PRECISION = 0.15
N_TRIALS = 3 if os.environ.get("AFG3_SMOKE") == "1" else int(os.environ.get("AFG3_NTRIALS", "40"))

# --- Feature set REVISADO (auditoría EDA 2026-06-21) — STAGED: aplica al regenerar el modelado ---
# RETIRADAS por CONFOUND TEMPORAL (su "señal" es el régimen 2023; desaparece en 2024+):
#   · miss_tipo_producto — 0 en test, Δ AUC-PR de retirarla = -0,001
#   · hora — artefacto 'hora==0' (78% en 2023); la hora real (hora>0) tiene ρ 2024+ ≈ -0,04
#   · mes — proxy de 2023 (ρ 2024+ = -0,007)
# AGREGADA: Distrito — señal real que el EDA recomendó pero NUNCA se incluyó (V 2024+ = 0,25).
# EVALUAR en regen: clientes_faltante_flag (borderline; V cae 0,20→0,07 en 2024+).
# Fuentes: la auditoría de procedencia del EDA.
#
# ✅ FECHAS CORREGIDAS (2026-06-21): el loader ya parsea con dayfirst=False (US mes-primero);
#   el split correcto da 11 críticos en test (no 12). Las predicciones/embeddings .npy viejas
#   quedaron invalidadas → regenerar en ESTA fase posterior sobre el split correcto.
#
# FEATURE SET tras el FIREWALL DEL TARGET (EDA v3) — APLICADO, no diferido.
#   El criterio (no el plan) define el alcance: un proxy del target con V de ese tamaño SALE ahora,
#   o la regen arrastraría leakage y habría que rehacerla. Se aplican AMBOS proxies, por simetría:
#   RETIRADAS por FIREWALL (proxies del target, persisten en 2024+ → leakage estructural, no confound):
#     · `Categoría de incidencia` — V×Criticidad 0,51 / V×Emergencia 0,63 (train); persiste 2024+ 0,56.
#       Se refina post-diagnóstico (dominio, confirmado por el equipo). Infla el eje CRITICIDAD.
#     · `clientes_bin` — V×Emergencia 0,94 (train; 105/107 Emergencia con clientes=Sí); persiste 2024+ 0,84.
#       Deriva de `Clientes afectados_final`, casi una copia del eje EMERGENCIA del target. Prueba interna:
#       la corrección de la fila 591 (591 clientes ∧ Leve → Crítico) usó "clientes afectados" para fijar el
#       TARGET → es información del target, no feature. El timing t=0 es SECUNDARIO: aunque se conociera al
#       abrir el ticket, un proxy casi perfecto de medio target queda descalificado igual.
#   RETIRADAS por CONFOUND TEMPORAL (señal = régimen 2023, se desvanece en 2024+): miss_tipo_producto, hora, mes.
#   AGREGADA: Distrito (señal débil pero estable, V×T1 2024+ ≈ 0,19).
#   El sufijo `_final` NO es argumento (lo llevan todas las columnas): lo es la fuerza de proxy + el timing.
#   Set sobreviviente: deliberadamente MAGRO (2 estructurales débiles + el texto) — refuerza el titular
#   honesto del EDA: ninguna feature estructural t=0 discrimina con fuerza.
NUM_COLS = [
    "n_tokens",
    "clientes_faltante_flag",
]
CAT_COLS = ["Distrito"]


def construir_features(df: pd.DataFrame) -> pd.DataFrame:
    """Recompone las features derivadas (modelo_v4 §1), con clientes_bin CORRECTO.

    Nota: `hora` y `mes` se siguen derivando, pero YA NO están en NUM_COLS/CAT_COLS
    (retiradas como confounds temporales). Se conservan sólo para diagnóstico/EDA
    (la demo del confound en el EDA v3 y la equidad por turno), NO entran a la matriz
    de features del modelo.
    """
    df = df.copy()
    df["hora"] = df["Fecha aviso incidencia"].dt.hour
    df["mes"] = df["Fecha aviso incidencia"].dt.month
    df["n_tokens"] = df["Descripcion"].fillna("").astype(str).str.split().map(len)
    # Definición correcta: la columna toma valores 'sí'/'no' (no 'alto'/'medio').
    df["clientes_bin"] = (
        df["clientes_afectados_flag"].astype(str).str.lower().isin(["sí", "si", "true", "1"])
    ).astype(int)
    return df


def build_preprocessor() -> ColumnTransformer:
    """OHE + StandardScaler, idéntico al N1 XGB (dense output para NB/SVM)."""
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ]
    )


# ---------------------------------------------------------------------------
# Constructores de clasificador por modelo (mismo presupuesto Optuna)
# ---------------------------------------------------------------------------

def clf_rf(trial: optuna.Trial):
    return RandomForestClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 500),
        max_depth=trial.suggest_int("max_depth", 2, 16),
        min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 10),
        max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=1,
    )


def clf_logreg(trial: optuna.Trial):
    return LogisticRegression(
        C=trial.suggest_float("C", 1e-3, 1e2, log=True),
        class_weight="balanced",
        solver="lbfgs",
        max_iter=4000,
        random_state=RANDOM_STATE,
    )


def clf_svm(trial: optuna.Trial):
    kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
    return SVC(
        C=trial.suggest_float("C", 1e-2, 1e2, log=True),
        kernel=kernel,
        gamma=trial.suggest_categorical("gamma", ["scale", "auto"]),
        class_weight="balanced",
        probability=True,
        random_state=RANDOM_STATE,
    )


def clf_nb(trial: optuna.Trial):
    return GaussianNB(var_smoothing=trial.suggest_float("var_smoothing", 1e-11, 1e-3, log=True))


MODELOS = {
    "rf": ("Random Forest", clf_rf),
    "logreg": ("Regresión Logística", clf_logreg),
    "svm": ("SVM", clf_svm),
    "nb": ("Naive Bayes (Gaussian)", clf_nb),
}


def threshold_operativo(y_true, proba, min_precision=MIN_PRECISION):
    """Threshold que maximiza Recall sujeto a Precision >= piso (convención t20c)."""
    precs, recs, ths = precision_recall_curve(y_true, proba)
    valid = np.where(precs[:-1] >= min_precision)[0]
    if len(valid) == 0:
        return 0.5, "precision_floor_NO_alcanzado"
    idx = valid[np.argmax(recs[:-1][valid])]
    return float(ths[idx]), "precision_floor_alcanzado"


def optimizar_y_evaluar(clave, nombre, ctor, X_train, X_test, y_train, y_test,
                        dates_test, y_test_t2, pred_t2_comun, cv):
    """Optuna + calibración isotónica temporal + threshold + producto final."""
    def objective(trial):
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", ctor(trial))])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(pipe, X_train, y_train, cv=cv,
                                     scoring=scorer_operativo, n_jobs=1)
        return float(np.nanmean(scores))

    study = optuna.create_study(direction="maximize",
                                sampler=TPESampler(seed=RANDOM_STATE),
                                study_name=f"afg3_{clave}")
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)

    best_pipe = Pipeline([("prep", build_preprocessor()),
                          ("clf", ctor(optuna.trial.FixedTrial(study.best_params)))])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proba_raw, proba_cal, _fitted, oof_cal = calibrar_isotonica(
            best_pipe, X_train, y_train, X_test, n_splits=5, return_oof_train=True
        )

    mask = np.isfinite(oof_cal)
    thr, flag = threshold_operativo(np.asarray(y_train)[mask], oof_cal[mask])
    pred_test = (proba_cal >= thr).astype(int)

    m_bin = evaluar_binario(f"T1 · {nombre}", y_test, proba_cal, pred_test, dates_test)
    m_pf = evaluar_producto_final(
        nombre=f"{nombre} + T2 común", y_true_t1=y_test, y_true_t2=y_test_t2,
        pred_t1=pred_test, pred_t2=pred_t2_comun, proba_t1=proba_cal,
        dates_test=dates_test, k=5,
    )

    # C3 · ECE+Brier pre/post isotónica y Platt (sostén cuantitativo de robustez)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cal_c3 = metricas_calibracion(best_pipe, X_train, y_train, X_test, y_test)

    # Persistir predicciones
    np.save(PRED / f"proba_t1_{clave}_afg3_raw.npy", proba_raw)
    np.save(PRED / f"proba_t1_{clave}_afg3_cal.npy", proba_cal)
    np.save(PRED / f"pred_t1_{clave}_afg3.npy", pred_test)

    return {
        "clave": clave, "modelo": nombre,
        "best_params": study.best_params,
        "cv_score_optuna": round(float(study.best_value), 4),
        "threshold": round(thr, 4), "threshold_flag": flag,
        "n_positivos_pred": int(pred_test.sum()),
        "auc_pr": m_bin["auc_pr"], "auc_roc": m_bin["auc_roc"], "brier": m_bin["brier"],
        "f2": m_bin["f2"], "precision_thr": m_bin["precision_thr"], "recall_thr": m_bin["recall_thr"],
        "ndcg5_pf": round(m_pf["ndcg@5_pf_7d_PRIMARIA"], 4),
        "recall5_pf": round(m_pf["recall@5_pf_7d_descriptivo"], 4),
        "calibracion_c3": cal_c3,
        "_proba_cal": proba_cal, "_pred_t1": pred_test,
    }


def main():
    print(f"[AFG3-Fase1a] N_TRIALS={N_TRIALS} (smoke={os.environ.get('AFG3_SMOKE')=='1'})", flush=True)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    df_train, df_test, embargo_diag = cargar_split_temporal_embargo()
    df_train = construir_features(df_train)
    df_test = construir_features(df_test)
    print(f"[AFG3] EMBARGO C2: gap={embargo_diag['gap_dias']}d  n_train {embargo_diag['n_train'][0]}->{embargo_diag['n_train'][1]}  "
          f"test={embargo_diag['n_test']} criticos_test={embargo_diag['criticos_test']}  "
          f"contingencia={embargo_diag['contingencia_requerida']}", flush=True)
    assert len(df_test) == 236, f"R1 violado: test={len(df_test)}"
    assert not embargo_diag["contingencia_requerida"], (
        "Contingencia C2 requerida (criticos_test<12): activar ventana móvil ampliada"
    )

    for c in NUM_COLS + CAT_COLS:
        assert c in df_train.columns, f"Falta feature {c}"
    print(f"[AFG3] clientes_bin train dist: {df_train['clientes_bin'].value_counts().to_dict()}", flush=True)

    X_train = df_train[NUM_COLS + CAT_COLS].copy()
    X_test = df_test[NUM_COLS + CAT_COLS].copy()
    y_train = df_train["target_criticidad"].astype(int).to_numpy()
    y_test = df_test["target_criticidad"].astype(int).to_numpy()
    y_test_t2 = df_test["target_t2"].astype(int).to_numpy()
    dates_test = df_test["Fecha aviso incidencia"]

    # T2 común = baseline paso20 (mismo que la tabla consolidada iteración 2)
    pred_t2_comun = np.load(PRED / "pred_t2_baseline_paso20.npy")
    assert len(pred_t2_comun) == 236, f"T2 común len={len(pred_t2_comun)}"

    # --- Referencia N1 XGB v2 (cargado, NO reentrenado) → debe dar ~0.884 ---
    proba_n1 = np.load(PRED / "proba_t1_n1_xgb_v2_calibrada.npy")
    pred_n1 = np.load(PRED / "pred_t1_n1_xgb_v2.npy")
    m_n1 = evaluar_producto_final(
        nombre="N1 XGB (referencia)", y_true_t1=y_test, y_true_t2=y_test_t2,
        pred_t1=pred_n1, pred_t2=pred_t2_comun, proba_t1=proba_n1, dates_test=dates_test, k=5,
    )
    ndcg_n1 = round(m_n1["ndcg@5_pf_7d_PRIMARIA"], 4)
    integrity_ok = abs(ndcg_n1 - 0.884) < 0.02
    print(f"\n[AFG3] INTEGRITY CHECK · N1 XGB referencia NDCG@5 PF = {ndcg_n1} "
          f"(esperado ~0.884) → {'OK' if integrity_ok else 'WARN'}", flush=True)

    # --- Correr RF PRIMERO (de-risk), luego el resto ---
    cv = SafeTimeSeriesSplit(n_splits=5, min_minority_samples=2)
    orden = ["rf", "logreg", "svm", "nb"]
    resultados = {}
    for clave in orden:
        nombre, ctor = MODELOS[clave]
        print(f"\n[AFG3] === {nombre} ({clave}) — Optuna {N_TRIALS} trials ===", flush=True)
        r = optimizar_y_evaluar(clave, nombre, ctor, X_train, X_test, y_train, y_test,
                                dates_test, y_test_t2, pred_t2_comun, cv)
        resultados[clave] = r
        print(f"[AFG3] {nombre}: AUC-PR={r['auc_pr']}  F2={r['f2']}  "
              f"NDCG@5 PF={r['ndcg5_pf']}  (N1 ref={ndcg_n1})", flush=True)
        if clave == "rf":
            delta_punt = round(r["ndcg5_pf"] - ndcg_n1, 4)
            print(f"\n  >>> DE-RISK RF vs N1: ΔNDCG@5 PF = {delta_punt:+.4f} "
                  f"({'RF supera' if delta_punt > 0 else 'N1 se mantiene'} en puntual) <<<\n", flush=True)

    # --- IC95: paired bootstrap RF vs N1 (Item 3) ---
    rf = resultados["rf"]
    n_boot = 200 if os.environ.get("AFG3_SMOKE") == "1" else 1000
    pb = paired_bootstrap_ic_producto_final(
        nombre_a="N1 XGB", nombre_b="Random Forest",
        y_true_t1=y_test, y_true_t2=y_test_t2,
        pred_t1_a=pred_n1, pred_t2_a=pred_t2_comun, proba_t1_a=proba_n1,
        pred_t1_b=rf["_pred_t1"], pred_t2_b=pred_t2_comun, proba_t1_b=rf["_proba_cal"],
        dates_test=dates_test, k=5, n_boot=n_boot, seed=RANDOM_STATE,
    )
    print(f"\n[AFG3] PAIRED BOOTSTRAP RF vs N1 ({n_boot} reps): "
          f"Δ(RF−N1)={pb.get('delta_mean', float('nan')):+.4f} "
          f"IC95=[{pb.get('delta_ic_low', float('nan')):+.4f}, {pb.get('delta_ic_high', float('nan')):+.4f}] "
          f"p={pb.get('p_value_unilateral_B_mejor_A', float('nan')):.3f}", flush=True)

    # --- Persistir tabla + JSON ---
    filas = [{"clave": "n1_xgb_ref", "modelo": "N1 XGB (referencia, vigente)",
              "auc_pr": m_n1.get("auc_pr", None), "ndcg5_pf": ndcg_n1,
              "recall5_pf": round(m_n1["recall@5_pf_7d_descriptivo"], 4)}]
    for clave in orden:
        r = resultados[clave]
        filas.append({k: r[k] for k in ["clave", "modelo", "auc_pr", "auc_roc", "brier",
                                        "f2", "precision_thr", "recall_thr", "ndcg5_pf",
                                        "recall5_pf", "threshold", "cv_score_optuna"]})
    df_tabla = pd.DataFrame(filas)
    df_tabla.to_csv(OUT_DIR / "afg3_baselines_clasicos.csv", index=False)

    salida = {
        "fase": "AFG III · Fase 1a — baselines clásicos (de-risk tabular)",
        "n_test": 236, "criticos_test": int(y_test.sum()),
        "n_trials_optuna": N_TRIALS,
        "t2_comun": "pred_t2_baseline_paso20.npy",
        "embargo_c2": embargo_diag,
        "integrity_check": {"ndcg_n1_reproducido": ndcg_n1, "esperado": 0.884, "ok": integrity_ok},
        "nota_auditoria_clientes_bin": "definición correcta sí→1; t5_shap usa alto/medio (bug, ≡0)",
        "resultados": {c: {k: resultados[c][k] for k in
                           ["modelo", "auc_pr", "f2", "ndcg5_pf", "recall5_pf",
                            "threshold", "best_params", "cv_score_optuna", "calibracion_c3"]}
                       for c in orden},
        "de_risk_rf_vs_n1": {
            "ndcg_n1": ndcg_n1, "ndcg_rf": rf["ndcg5_pf"],
            "delta_puntual": round(rf["ndcg5_pf"] - ndcg_n1, 4),
            "paired_bootstrap": {k: (round(pb[k], 4) if isinstance(pb.get(k), float) else pb.get(k))
                                 for k in pb},
        },
    }
    (OUT_DIR / "afg3_baselines_clasicos.json").write_text(
        json.dumps(salida, indent=2, ensure_ascii=False))

    outs = [str(PRED / f"proba_t1_{c}_afg3_cal.npy") for c in orden]
    outs += [str(PRED / f"pred_t1_{c}_afg3.npy") for c in orden]
    outs += [str(OUT_DIR / "afg3_baselines_clasicos.csv"),
             str(OUT_DIR / "afg3_baselines_clasicos.json")]
    registrar_artefactos(paso="AFG3_T22a_baselines", inputs=[
        str(PRED / "pred_t2_baseline_paso20.npy"),
        str(PRED / "proba_t1_n1_xgb_v2_calibrada.npy"),
    ], outputs=outs, ruta_manifest=str(OUT_DIR / "manifest_afg3.json"))

    print("\n[AFG3] === TABLA FINAL ===", flush=True)
    print(df_tabla.to_string(index=False), flush=True)
    print(f"\n[AFG3] Salidas: outputs/afg3_baselines_clasicos.{{csv,json}} + manifest_afg3.json", flush=True)


if __name__ == "__main__":
    main()
