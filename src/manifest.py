"""Manifest criptográfico de integridad para la cadena T20 (saneamiento PASO 20).

Implementa validación SHA256 de inputs/outputs entre pasos consecutivos para
detectar desincronizaciones silenciosas (bug F1 documentado en
la bitácora de decisiones del proyecto (2026-05-17)).

Convertimos la cadena T20 de "frágil por convenio" en "frágil por contrato":
si un input no coincide con el output declarado por el paso previo, el script
aborta con mensaje claro antes de propagar el estado inconsistente.

Patrón de uso por script:

```python
from src.manifest import verificar_integridad, registrar_artefactos

# Al inicio (saltar para el primer eslabón T20a):
verificar_integridad(paso_previo="T20a", inputs_actuales=[
    "data/processed/predicciones/oof_train_original_proba_t1_n1_xgb_v2.npy",
    "data/processed/predicciones/oof_train_original_proba_t1_hibrido_v2.npy",
])

# ... lógica del script ...

# Al final:
registrar_artefactos(
    paso="T20b",
    inputs=[...],
    outputs=[...],
)
```
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_DEFAULT_MANIFEST = "outputs/manifest_t20.json"


def calcular_sha256(ruta_archivo: str) -> str:
    """Hash SHA256 de un archivo, leído en bloques de 64 KB."""
    if not os.path.exists(ruta_archivo):
        raise FileNotFoundError(f"Archivo crítico no encontrado: {ruta_archivo}")
    sha256_hash = hashlib.sha256()
    with open(ruta_archivo, "rb") as f:
        for bloque in iter(lambda: f.read(65536), b""):
            sha256_hash.update(bloque)
    return sha256_hash.hexdigest()


def registrar_artefactos(
    paso: str,
    inputs: list,
    outputs: list,
    ruta_manifest: str = _DEFAULT_MANIFEST,
) -> None:
    """Registra hashes SHA256 de inputs y outputs de un paso en el manifest global.

    Idempotente: re-correr un paso sobrescribe su entrada en el JSON sin tocar
    las de otros pasos.

    Args:
        paso: identificador del paso, p. ej. "T20a", "T20c", "T15".
        inputs: lista de rutas (string o Path) a los archivos input.
        outputs: lista de rutas a los archivos output que este paso produjo.
        ruta_manifest: ruta al JSON global (default `outputs/manifest_t20.json`).
    """
    data = {}
    if os.path.exists(ruta_manifest):
        with open(ruta_manifest, "r", encoding="utf-8") as f:
            data = json.load(f)

    data[paso] = {
        "inputs": {str(f): calcular_sha256(str(f)) for f in inputs},
        "outputs": {str(f): calcular_sha256(str(f)) for f in outputs},
    }

    Path(ruta_manifest).parent.mkdir(parents=True, exist_ok=True)
    with open(ruta_manifest, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def verificar_integridad(
    inputs_actuales: list,
    paso_previo=None,
    ruta_manifest: str = _DEFAULT_MANIFEST,
) -> None:
    """Verifica integridad SHA256 de inputs contra el manifest.

    Modo flexible — un input puede provenir de cualquier paso de la cadena:
    - Si `paso_previo` es None (default): busca cada input en outputs de
      cualquier paso registrado. Permite que un script consuma archivos de
      múltiples pasos anteriores (p. ej. T20c usa OOF de T20a y pickles de T20b).
    - Si `paso_previo` es str: exige que cada input esté en outputs de ese
      paso específico (modo estricto).
    - Si `paso_previo` es list/tuple: busca solo en esos pasos.

    Args:
        inputs_actuales: lista de rutas a verificar (string o Path).
        paso_previo: opcional, restringe la búsqueda a un paso o lista de pasos.
        ruta_manifest: ruta al JSON global (default `outputs/manifest_t20.json`).

    Raises:
        RuntimeError: manifest ausente o paso_previo no registrado.
        ValueError: hash mismatch o input no declarado en ningún paso.
    """
    if not os.path.exists(ruta_manifest):
        raise RuntimeError(
            f"No existe el manifiesto global ({ruta_manifest}). "
            f"Ejecuta la cadena desde el inicio (T20a)."
        )
    with open(ruta_manifest, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determinar qué pasos consultar
    if paso_previo is None:
        pasos_a_consultar = list(data.keys())
    elif isinstance(paso_previo, str):
        if paso_previo not in data:
            raise RuntimeError(
                f"El paso previo {paso_previo} no ha sido registrado en el "
                f"manifiesto. Ejecuta {paso_previo} antes de este paso."
            )
        pasos_a_consultar = [paso_previo]
    else:  # list/tuple
        for p in paso_previo:
            if p not in data:
                raise RuntimeError(
                    f"El paso previo {p} no ha sido registrado. "
                    f"Ejecuta {p} antes de este paso."
                )
        pasos_a_consultar = list(paso_previo)

    for f_ in inputs_actuales:
        f_str = str(f_)
        hash_actual = calcular_sha256(f_str)
        encontrado_en = None
        for paso in pasos_a_consultar:
            registrado = data[paso]["outputs"].get(f_str)
            if registrado is None:
                continue
            if registrado != hash_actual:
                raise ValueError(
                    f"CRÍTICO: el archivo {f_str} sufrió modificaciones externas o "
                    f"desincronización vs su estado al cierre de {paso}. "
                    f"Re-ejecuta la cadena desde {paso}."
                )
            encontrado_en = paso
            break
        if encontrado_en is None:
            ctx = ("ningún paso registrado" if paso_previo is None
                   else f"los pasos {pasos_a_consultar}")
            raise ValueError(
                f"CRÍTICO: el archivo {f_str} no fue declarado como output de "
                f"{ctx}. Revisa que estés ejecutando la cadena en orden y que "
                f"este input sea realmente producido por la cadena T20 (los "
                f"insumos externos del PASO 19 / modelo_v4 NO se verifican)."
            )
