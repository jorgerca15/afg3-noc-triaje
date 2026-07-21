# Asistente de triaje de tickets NOC basado en modelos de lenguaje

Pipeline reproducible para clasificar y **priorizar tickets de un Centro de Operación
de Red (NOC)** en el momento de apertura (`t = 0`), combinando una señal de
**severidad** y una de **urgencia** en un ranking operativo.

La pregunta central del proyecto es empírica: **¿la información que permite anticipar
la criticidad de un incidente vive en las variables estructuradas del ticket
(distrito, conteos, banderas) o en el texto libre de la descripción?** Todo se evalúa
únicamente con lo disponible al abrir el ticket, sin usar campos que se completan
durante o después de la atención (cero fuga temporal).

## Qué resuelve

El NOC prioriza combinando dos ejes, ambos leídos del ticket en `t = 0`:

- **T1 — Severidad**: clasificación binaria `Crítico / Leve`.
- **T2 — Urgencia**: tipo de trabajo (`Correctivo`, `Preventivo`, `Emergencia`,
  `Incidente`).

Ambos ejes se fusionan en una **matriz de prioridad ordinal** de cuatro niveles y el
sistema entrega un **ranking Top-K** por ventanas de 7 días. La calidad del ranking se
mide con **NDCG@5** (métrica primaria, graduada sobre los cuatro niveles) y
**Recall@5** sobre los incidentes de máxima prioridad (descriptiva).

## Resultado principal

Sobre una partición **temporal pura 60/40** (con embargo del lado de entrenamiento
para cerrar la maduración de etiqueta) y un conjunto de test con 11 incidentes
críticos observados:

| Señal en `t = 0`                         | Discriminación (AUC-PR) |
|------------------------------------------|:-----------------------:|
| Piso de no-skill (prevalencia en test)   | 0,047                   |
| Estructural (Random Forest)              | 0,071                   |
| Texto — TF-IDF                           | 0,166                   |
| Texto — MiniLM multilingüe               | 0,196                   |
| Texto — e5 multilingüe                   | 0,251                   |
| Texto — BETO (BERT en español)           | **0,427**               |

La conclusión es robusta: **la señal estructural es prácticamente inerte** (su
intervalo de confianza incluye el piso de prevalencia), mientras que **la semántica de
la descripción sí discrimina la criticidad** por encima del azar.

Esta señal **no es un artefacto de fuga de información**. Una **prueba de permutación**
(200 reordenamientos aleatorios de las etiquetas) sitúa el AUC-PR observado muy por
encima de la distribución nula (**p = 0,005**), y un test de **anti-circularidad**
confirma que el modelo no está leyendo la etiqueta: de los 11 tickets a los que asigna
mayor confianza, solo **5 son críticos reales** (un modelo que copiara la respuesta
acertaría casi todos).

El **mismo patrón se repite en el eje de urgencia** (T2, tipo de trabajo), medido con
**F1 macro** sobre las cuatro clases:

| Señal en `t = 0`                          | Discriminación (F1 macro) |
|-------------------------------------------|:-------------------------:|
| Clasificador trivial (clase mayoritaria)  | 0,131                     |
| Estructural (Random Forest)               | 0,242                     |
| Texto — MiniLM multilingüe                | 0,402                     |
| Texto — BETO (BERT en español)            | 0,428                     |
| Texto — e5 multilingüe                    | 0,439                     |
| Texto — TF-IDF                            | 0,464                     |

También en la urgencia el texto casi **duplica** al nivel estructural, y la mejora es
sólida: el intervalo de confianza estructural (`[0,195, 0,292]`) no se solapa con el del
texto (`[0,420, 0,506]`). En términos operativos, la detección de **Emergencia** —la
clase que define el nivel máximo de prioridad— sube de un **recall de 0,567** con la vía
estructural a **0,733–0,833** con el texto. Dentro de la familia semántica los
codificadores son estadísticamente indistinguibles (sus intervalos se solapan), por lo
que no se afirma un mejor codificador de urgencia; en el producto se usa BETO en ambos
ejes por consistencia con la severidad. Que el texto gobierne **los dos ejes** es lo que
habilita el producto combinado.

El **producto full-semántico** (T1 y T2 leídos ambos del texto con BETO) alcanza:

- **NDCG@5 = 0,874**
- **Recall@5 (nivel máximo) = 1,0** — los 11/11 incidentes críticos quedan en el Top-5,
  sin falsos negativos.

**Nota de honestidad estadística.** El desempeño absoluto es alto, pero el régimen
muestral es pequeño (11 críticos en test). Por eso el proyecto declara explícitamente
sus límites de potencia: el efecto que aísla la contribución de la severidad no es
concluyente por tamaño de muestra, no por el modelo (se estima que confirmarlo exigiría del orden de **32 críticos** en
test, unas 2,9 veces los actuales). El cuaderno documenta este punto sin sobrevender el
resultado.

## El asistente de dos capas

El producto del proyecto es un asistente de dos capas:

- **Capa 1 — triaje automático (validado)**: codifica la descripción con BETO y produce
  la prioridad ordinal y el ranking Top-K. Es la capa que reporta y reproduce este
  repositorio.
- **Capa 2 — asistencia generativa (prototipo, no validado)**: sobre el ticket
  priorizado, un modelo de lenguaje local propone una hipótesis de causa raíz y una
  mitigación a partir de casos históricos similares. Se mantiene como prototipo bajo
  supervisión humana: en la evaluación ciega no alcanzó la utilidad mínima en el estrato
  de tickets críticos (**2,92 sobre 3,0**), por lo que no se despliega.

El operador decide siempre; el asistente **ordena y sugiere, no reemplaza el juicio
humano**.

## Cómo reproducir

Requisitos: Python 3.10+.

```bash
# 1. Crear entorno e instalar dependencias
python -m venv venv
source venv/bin/activate        # en Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. Abrir el cuaderno y ejecutarlo de principio a fin
jupyter notebook notebooks/pipeline_canonico.ipynb
```

De forma no interactiva:

```bash
jupyter nbconvert --to notebook --execute notebooks/pipeline_canonico.ipynb --inplace
```

El cuaderno es un **visor reproducible**: reconstruye en vivo el split temporal, las
features y la matriz de prioridad a partir del dataset incluido, y lee los artefactos
de resultados precomputados (`outputs/*.json`). No reentrena los modelos de embeddings.
Como autochequeo, recomputa el **SHA256** de cada artefacto canónico y lo compara
contra el manifest, de modo que cualquier divergencia bit a bit se detecta al ejecutar.

## Estructura del repositorio

```
afg3-noc-triaje/
├── notebooks/
│   └── pipeline_canonico.ipynb      # cuaderno principal (visor reproducible end-to-end)
├── src/                             # utilidades reusables (métricas, calibración, split, PII)
├── scripts/                         # loaders del split temporal y baselines estructurales
├── outputs/                         # artefactos de resultados (JSON) + manifest SHA256
├── data/processed/                  # dataset pseudonimizado + embeddings y predicciones
├── requirements.txt
└── README.md
```

## Sobre los datos

El dataset de tickets se distribuye **pseudonimizado**: los identificadores de cliente,
distrito, nodo y ticket se sustituyen por tokens categóricos anonimizados
(`<CLIENT_XX>`, `DIST_XXX`, `TCK_XXXXXXXXXX`) antes de versionarse. No contiene
información personal ni identificable. El repositorio incluye además una capa de
gobernanza de PII para el texto libre, pensada como filtro de entrada para un
despliegue futuro.
