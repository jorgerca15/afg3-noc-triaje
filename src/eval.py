"""Shim de compatibilidad de `src.eval` (metricas del proyecto).

El modulo se dividio en dos para separar el canon-vivo de la analitica
superseded, sin romper ningun import existente:

- `src/eval_producto.py` : Producto Final T1+T2 + metricas del pipeline vivo.
- `src/eval_legacy.py`   : analitica RAMA-B / PASO 19-21 (costo 50:1, cross-arm, MDES).

Este archivo re-exporta AMBOS namespaces, de modo que `from src.eval import X`
sigue resolviendo cualquier nombre previo (publico o helper interno). Para codigo
nuevo, importar directamente del submodulo que corresponda.
"""

from __future__ import annotations

from .eval_producto import *  # noqa: F401,F403
from .eval_legacy import *  # noqa: F401,F403

# Helpers con prefijo `_` que se importan por nombre en motores/notebooks vivos
# y que `import *` no re-exporta por convencion.
from .eval_producto import (  # noqa: F401
    _ndcg_graduado,
    _recall_at_min_precision,
    _score_ordinal_labels,
)
from .eval_legacy import (  # noqa: F401
    _bootstrap_costo_pred_pareado,
    _rbo_extrapolated,
)
