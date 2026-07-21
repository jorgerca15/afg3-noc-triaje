"""Sanitización PII del texto libre del ticket — capa de gobernanza (Fase 2 · AFG III).

ALCANCE (declarado, no ambiguo): esta capa es **gobernanza/defensa + filtro de
entrada para el despliegue futuro**. NO se retrofitea al pipeline canónico
congelado: los modelos canónicos (RF, BETO, e5) entrenaron sobre el texto
*pseudonimizado-por-columna* (`<CLIENT_XX>`, `DIST_XXX`, `TCK_XXXXXXXXXX`), NO
sobre texto text-scrubbed. Por tanto el argumento de "no partir el conteo TF-IDF"
aplica al pipeline de despliegue donde la sanitización está activa, no al canon.

POR QUÉ TOKENS CATEGÓRICOS Y NO HASH (decisión de ciberseguridad, AFG III):
un hash determinista de una IP o un nombre tiene entropía baja y es reversible en
segundos por tablas arcoíris; además, un hash recurrente asociado a fallas críticas
se vuelve una FEATURE que el modelo aprende (sesgo oculto por identidad). Se usa
substitución por **tokens categóricos estériles**:
    <IP_CRITICA_n> · <NODO_x> · <CLIENTE_CORP_n> · <TICKET_n> · <CREDENCIAL>
Consistencia INTRA-TICKET: la misma entidad dentro de un ticket mapea al MISMO
token (una IP repetida → un único <IP_CRITICA_1>, no _1 y _2 — relevante para no
partir el conteo de término en TF-IDF). Reset del contador POR-TICKET: el token
`<IP_CRITICA_1>` del ticket A es independiente del del ticket B → no hay un
token-identidad estable que el modelo pueda memorizar entre tickets.

PROMESA ACOTADA (ética honesta): el reset por-ticket NO "elimina" el sesgo, lo
TRANSFORMA — pasa de memorizar una IP a aprender que `<IP_CRITICA_n>` ≈ crítico,
más una señal categórica de tipo y conteo. Redacción honesta: "reduce el sesgo de
identidad estable, a costa de una señal categórica de tipo y conteo que documentamos".

Las CREDENCIALES se REDACTAN (token plano `<CREDENCIAL>`) y NO se guardan en el
mapeo: un secreto expuesto se rota, no se traza. Solo IP/cliente/nodo/ticket quedan
en el diccionario de mapeo, que vive cifrado y gitignored, y que el modelo nunca lee.
"""
from __future__ import annotations

import json
import os
import re
import stat
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUTA_MAPEO_DEFAULT = ROOT / "data" / "secure" / "pii_mapping.enc"

# --- Patrones regex (orden de prioridad: credencial > ip > pseudónimos > ticket) ---
# Credencial: clave/valor explícito → se redacta el match completo.
_RE_CREDENCIAL = re.compile(
    r"(?i)\b(?:password|passwd|pwd|contrase\w+a|clave|token|api[_-]?key|secret|bearer)\b\s*[:=]\s*\S+"
)
_RE_IPV4 = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d?\d)\b"
)
# Pseudónimos de columna ya presentes en el dataset (defensa en profundidad).
_RE_CLIENTE = re.compile(r"<CLIENT_[A-Z0-9_]+>")
_RE_NODO = re.compile(r"\bDIST_\d+\b")
_RE_TICKET = re.compile(r"\bTCK_[A-Z0-9]+\b")

# tipo → (regex, plantilla_token, traceable). 'traceable=False' = redactado, no va al mapeo.
_REGLAS = [
    ("CREDENCIAL", _RE_CREDENCIAL, "<CREDENCIAL>", False),
    ("IP_CRITICA", _RE_IPV4, "<IP_CRITICA_{n}>", True),
    ("CLIENTE_CORP", _RE_CLIENTE, "<CLIENTE_CORP_{n}>", True),
    ("NODO", _RE_NODO, "<NODO_{n}>", True),
    ("TICKET", _RE_TICKET, "<TICKET_{n}>", True),
]


def detectar_pii(texto: str) -> list[dict]:
    """Detecta entidades PII en el texto. Devuelve lista de {tipo, valor, inicio, fin}.

    Resuelve solapes quedándose con el match de mayor prioridad (orden de `_REGLAS`)
    y, a igual prioridad, el de mayor longitud.
    """
    txt = "" if texto is None else str(texto)
    candidatos = []
    for tipo, regex, _plantilla, _trace in _REGLAS:
        for m in regex.finditer(txt):
            candidatos.append({"tipo": tipo, "valor": m.group(0),
                               "inicio": m.start(), "fin": m.end()})
    # Resolver solapes: prioridad por orden en _REGLAS, luego longitud.
    prioridad = {t[0]: i for i, t in enumerate(_REGLAS)}
    candidatos.sort(key=lambda c: (c["inicio"], prioridad[c["tipo"]], -(c["fin"] - c["inicio"])))
    seleccion, ocupado_hasta = [], -1
    for c in candidatos:
        if c["inicio"] >= ocupado_hasta:
            seleccion.append(c)
            ocupado_hasta = c["fin"]
    return seleccion


def sanitizar_texto(texto: str) -> tuple[str, dict]:
    """Sustituye PII por tokens categóricos estériles (consistencia intra-ticket).

    Returns:
        (texto_esteril, mapeo) donde `mapeo` = {token: valor_original} SOLO para
        entidades trazables (IP/cliente/nodo/ticket). Las credenciales se redactan
        a `<CREDENCIAL>` y NO entran al mapeo (un secreto se rota, no se traza).
        El contador se reinicia en cada llamada (reset por-ticket).
    """
    txt = "" if texto is None else str(texto)
    detecciones = detectar_pii(txt)
    if not detecciones:
        return txt, {}

    plantillas = {t[0]: t[2] for t in _REGLAS}
    traceable = {t[0]: t[3] for t in _REGLAS}
    contador: dict[str, int] = {}
    valor_a_token: dict[tuple[str, str], str] = {}  # (tipo, valor) → token (consistencia intra-ticket)
    mapeo: dict[str, str] = {}

    # Reemplazar de derecha a izquierda para no invalidar offsets.
    detecciones_ord = sorted(detecciones, key=lambda d: d["inicio"], reverse=True)
    out = txt
    for d in detecciones_ord:
        tipo, valor = d["tipo"], d["valor"]
        clave = (tipo, valor)
        if clave in valor_a_token:
            token = valor_a_token[clave]
        else:
            if "{n}" in plantillas[tipo]:
                contador[tipo] = contador.get(tipo, 0) + 1
                token = plantillas[tipo].format(n=contador[tipo])
            else:
                token = plantillas[tipo]  # plano (credencial)
            valor_a_token[clave] = token
            if traceable[tipo]:
                mapeo[token] = valor
        out = out[:d["inicio"]] + token + out[d["fin"]:]
    return out, mapeo


def sanitizar_dataframe(textos, ids=None) -> tuple[list[str], dict]:
    """Aplica `sanitizar_texto` fila-a-fila (reset por-ticket).

    Returns:
        (textos_esteriles, mapeo_por_ticket) donde mapeo_por_ticket = {id: {token: valor}}.
        Si `ids` es None, usa el índice posicional como id.
    """
    textos = list(textos)
    ids = list(ids) if ids is not None else list(range(len(textos)))
    esteriles, mapeo_por_ticket = [], {}
    for tid, t in zip(ids, textos):
        s, m = sanitizar_texto(t)
        esteriles.append(s)
        if m:
            mapeo_por_ticket[str(tid)] = m
    return esteriles, mapeo_por_ticket


def persistir_mapeo_cifrado(mapeo_por_ticket: dict, ruta: Path = RUTA_MAPEO_DEFAULT) -> dict:
    """Persiste el diccionario de mapeo a un archivo CIFRADO y gitignored.

    El modelo NUNCA lee este archivo: existe solo para que un operador humano pueda
    des-anonimizar bajo control. Si `cryptography` está disponible se usa Fernet
    (AES-128 autenticado); si no, se escribe JSON con permisos 0o600 y un flag
    `cifrado=False` + advertencia (el archivo igual queda fuera de git y fuera del
    alcance del modelo; para cifrado-at-rest real instalar `cryptography`).

    Returns:
        dict con {ruta, cifrado(bool), n_tickets, ruta_clave(si aplica)}.
    """
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(mapeo_por_ticket, ensure_ascii=False).encode("utf-8")
    try:
        from cryptography.fernet import Fernet
        ruta_clave = ruta.with_suffix(".key")
        if ruta_clave.exists():
            clave = ruta_clave.read_bytes()
        else:
            clave = Fernet.generate_key()
            ruta_clave.write_bytes(clave)
            os.chmod(ruta_clave, stat.S_IRUSR | stat.S_IWUSR)  # 0o600
        token = Fernet(clave).encrypt(payload)
        ruta.write_bytes(token)
        os.chmod(ruta, stat.S_IRUSR | stat.S_IWUSR)
        return {"ruta": str(ruta), "cifrado": True, "n_tickets": len(mapeo_por_ticket),
                "ruta_clave": str(ruta_clave)}
    except Exception:
        ruta_json = ruta.with_suffix(".json")
        ruta_json.write_bytes(payload)
        os.chmod(ruta_json, stat.S_IRUSR | stat.S_IWUSR)
        return {"ruta": str(ruta_json), "cifrado": False, "n_tickets": len(mapeo_por_ticket),
                "advertencia": "cryptography no instalado → JSON 0o600 gitignored; "
                               "instalar `cryptography` para cifrado-at-rest real."}
