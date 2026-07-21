"""Utilidades reusables del pipeline de triaje NOC.

Módulos incluidos en este repositorio:
- cv: SafeTimeSeriesSplit y scorers de validación cruzada temporal.
- eda: métricas de asociación (V de Cramér) para el diagnóstico exploratorio.
- eval_producto / eval_legacy / eval: métricas de ranking (Recall@k, NDCG@k),
  matriz de prioridad ordinal y evaluación del producto combinado T1 + T2.
- calibracion: calibración isotónica temporal vía OOF (SafeTimeSeriesSplit).
- sanitizacion_pii: capa de gobernanza para saneamiento de PII del texto libre.
- manifest: registro y verificación criptográfica (SHA256) de artefactos.
"""

RANDOM_STATE = 42
