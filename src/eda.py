"""Helpers exhibibles del EDA v3 (paridad C1: lo explicado == lo ejecutado).

Reúsan la V de Cramér con de-sesgo de Bergsma del motor de auditoría
(`scripts/afg3_auditoria_eda.py::cramers_v`) y añaden un árbol de importancia
ligero. El cuaderno los exhibe con `inspect.getsource` para que el lector vea el
código real que produce cada número, no una paráfrasis.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.tree import DecisionTreeClassifier


def cramers_v(x, y, bias_correction: bool = True) -> dict:
    """V de Cramér con de-sesgo de Bergsma (2013) + diagnóstico de potencia.

    El NaN se trata como una categoría propia ('·NA·'): la faltancia es
    informativa (MNAR) y descartarla sesgaría la asociación. Devuelve un dict:

      - ``v``       fuerza de asociación en [0, 1] (de-sesgada).
      - ``p``       p-valor del χ² de independencia (¿la asociación es real?).
      - ``cochran`` fracción de celdas con frecuencia esperada < 5. Si > 0,2 la
                    V es inestable (muestra fina): leer con cautela.
      - ``n``       tamaño efectivo de la tabla.

    Bergsma corrige el sesgo al alza de la V cruda en muestras chicas y tablas
    grandes (cardinalidad alta infla la V): es imprescindible para comparar en un
    mismo eje features de distinta cardinalidad sin premiar a las más granulares.
    """
    x = pd.Series(x).astype(object).where(pd.notna(x), "·NA·")
    y = pd.Series(y).astype(object).where(pd.notna(y), "·NA·")
    ct = pd.crosstab(x, y)
    if min(ct.shape) < 2:
        return dict(v=np.nan, p=np.nan, cochran=np.nan, n=int(ct.values.sum()))
    chi2, p, _dof, exp = chi2_contingency(ct, correction=False)
    n = ct.values.sum()
    phi2 = chi2 / n
    r, k = ct.shape
    if bias_correction:
        phi2 = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
        r = r - (r - 1) ** 2 / (n - 1)
        k = k - (k - 1) ** 2 / (n - 1)
    v = float(np.sqrt(phi2 / max(1e-12, min(k - 1, r - 1))))
    cochran = float((exp < 5).mean())
    return dict(v=v, p=float(p), cochran=cochran, n=int(n))


def tree_importance(df, features, target: str = "target_criticidad",
                    max_depth: int = 5, seed: int = 42) -> pd.Series:
    """Importancia de un árbol de decisión balanceado, como rankeador rápido de señal.

    Las categóricas se codifican por enteros (``cat.codes``); las numéricas se
    usan tal cual. Devuelve una Serie ordenada de mayor a menor importancia.

    Advertencia metodológica: un árbol mide poder predictivo, NO causalidad ni
    validez temporal. No distingue una señal de un confound (una feature atada al
    período puede coronar el ranking sin generalizar). Por eso el EDA NUNCA le
    cree a la #1 sin cruzarla antes con la prueba temporal (¿la señal sobrevive a
    quitar el régimen 2023?). El árbol propone; la lente temporal dispone.
    """
    X = df[list(features)].copy()
    for c in features:
        if X[c].dtype == object or str(X[c].dtype) == "category":
            X[c] = X[c].astype("category").cat.codes
    dt = DecisionTreeClassifier(max_depth=max_depth, class_weight="balanced",
                                random_state=seed)
    dt.fit(X, df[target])
    return pd.Series(dt.feature_importances_, index=list(features)).sort_values(ascending=False)
