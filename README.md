# Clasificación del español andino: comparación de ML, LLM y RAG morfosintáctico

Repositorio del experimento de tesis sobre **clasificación binaria de oraciones** según la presencia de fenómenos lingüísticos asociados al español andino peruano. Se comparan cuatro sistemas bajo prueba (*Systems Under Test*, SUT), con énfasis en la recuperación morfosintáctica de reglas lingüísticas frente a la recuperación semántica.

> **Alcance:** la evaluación se realizó con **900 oraciones sintéticas** y etiquetas operativas `andino` / `no_andino`. Estas categorías se refieren a los fenómenos definidos para el experimento; no permiten inferir el origen, identidad ni variedad lingüística completa de una persona.

## Sistemas comparados

| Sistema | Implementación | Procedimiento |
| --- | --- | --- |
| **SUT 1 — ML** | spaCy `es_core_news_md` + regresión logística | Promedio de embeddings de palabras (300 dimensiones) y validación cruzada estratificada de 5 folds. |
| **SUT 2 — LLM** | `meta-llama/Llama-3.1-8B-Instruct` | Clasificación mediante instrucciones, sin recuperación externa. |
| **SUT 3 — RAG semántico** | BGE-M3 + FAISS + Llama | Similitud coseno sobre las nueve reglas completas, recuperación de **0 o 1 regla**, umbral **0.4**. |
| **SUT 4 — RecMS** | Stanza + detectores heurísticos + Llama | Análisis morfosintáctico, búsqueda de la **primera regla coincidente** según una prioridad fija; recuperación de **0 o 1 regla**. |

Los SUT 2, 3 y 4 utilizan el mismo modelo generativo y la misma plantilla de instrucciones. Los SUT 3 y 4 utilizan el mismo documento de reglas. El modelo generativo se carga en 4 bits (NF4), con `max_new_tokens=256` y `do_sample=False`.

## Conjunto de datos y conocimiento lingüístico

- **Archivo:** `datos/ejemplos_dialectos_900.csv`.
- **Distribución:** 9 reglas (`R01`–`R09`) × 2 etiquetas × 50 oraciones = **900 casos**; 450 por clase.
- **Campos principales:** `case_id` (identificador), `regla_id` (regla del grupo), `test_case` (texto), `label` (etiqueta de referencia).
- **Reglas:** `datos/reglas_andino.txt`, con identificadores, descripciones y ejemplos.
- **Umbral de recuperación semántica:** `datos/umbral_rag.txt`, cuyo valor para los resultados publicados es **0.4**.

El campo `regla_id` permite agrupar y evaluar los casos: **no se entrega como entrada a los recuperadores ni al LLM**. En los ejemplos `no_andino`, identifica el grupo de contraste, no una regla que deba recuperarse.

## Estructura del repositorio

```text
tesis_experimento/
├── datos/
│   ├── ejemplos_dialectos_900.csv
│   ├── reglas_andino.txt
│   └── umbral_rag.txt
├── src/
│   ├── comun.py
│   ├── sut1_ml.py
│   ├── sut2_llm.py
│   ├── sut3_rag.py
│   ├── sut4_recms.py
│   └── evaluacion.py
├── notebooks/
│   ├── experimentos.ipynb
│   └── analisis_resultados.ipynb
├── resultados/
│   ├── sut1_ml_predicciones.csv
│   ├── sut2_llm_predicciones.csv
│   ├── sut3_rag_predicciones.csv
│   ├── sut4_recms_predicciones.csv
│   └── analisis_final/              # tablas, figuras y resúmenes derivados
├── requirements.txt
├── .gitignore
└── README.md
```

`src/` implementa los sistemas y las funciones compartidas. `notebooks/experimentos.ipynb` coordina las inferencias y guarda predicciones; `notebooks/analisis_resultados.ipynb` reproduce la evaluación a partir de los CSV, sin volver a consultar los modelos.

## Requisitos

- **Google Colab** y **Google Drive** para ejecutar los notebooks tal como están configurados.
- **GPU** (por ejemplo, T4) para nuevas inferencias con Llama; **CPU** es suficiente para analizar resultados existentes.
- Acceso aprobado en [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) al modelo `meta-llama/Llama-3.1-8B-Instruct` para volver a ejecutar los SUT generativos.
- Bibliotecas descritas en `requirements.txt`. El notebook de experimentos también instala sus propias dependencias y descarga `es_core_news_md` de spaCy; para Stanza descarga los recursos del español cuando los necesita.

No se incluyen pesos de modelos ni credenciales. Las versiones de varias dependencias no están fijadas de manera exacta; por lo tanto, la repetición de inferencias en otro entorno podría no producir salidas idénticas.

## Ejecución en Google Colab

### 1. Preparar los archivos en Drive

Descarga o copia el repositorio **completo** a la ruta:

```text
Mi unidad/tesis_experimento/
```

La ruta está configurada en ambos notebooks como:

```python
RAIZ = Path('/content/drive/MyDrive/tesis_experimento')
```

Si utilizas otra carpeta, cambia `RAIZ` en ambos notebooks antes de ejecutar las celdas. Comprueba que `datos/umbral_rag.txt` contenga `0.4` si deseas utilizar la configuración documentada.

### 2. Obtener o verificar las predicciones

1. Abre **`notebooks/experimentos.ipynb`** en Colab.
2. Ejecuta las celdas **de arriba abajo**.
3. Si los cuatro archivos `resultados/sut*_predicciones.csv` existen, están completos y superan la validación, el notebook **los reutiliza sin volver a ejecutar inferencias**. En este caso basta con un entorno CPU.
4. Si falta algún CSV, activa una GPU y, para los SUT basados en Llama, autoriza el modelo en Hugging Face. Añade el token como **secreto de Colab `HF_TOKEN`** (panel *Secretos*) y permite el acceso del notebook; **no lo escribas en el código ni lo publiques**.
5. Las inferencias nuevas de SUT 2–4 se guardan en bloques de **25 casos**, con respaldos parciales en Drive que permiten reanudar el proceso si se interrumpe la sesión.

Para forzar una ejecución nueva de algún sistema, modifica explícitamente la variable `REGENERAR` del notebook **después de conservar una copia de los resultados publicados**. No mezcles respaldos de ejecuciones con configuraciones diferentes.

> La validación de CSV existentes comprueba su estructura y correspondencia con el dataset, pero **no equivale a volver a ejecutar los modelos desde cero**.

### 3. Reproducir las métricas y figuras

1. Abre **`notebooks/analisis_resultados.ipynb`** en Colab.
2. Selecciona **CPU** y ejecuta **Entorno de ejecución → Ejecutar todas**.
3. El notebook verifica los **3600 registros de predicción** y produce las tablas y figuras dentro de `resultados/analisis_final/`.
4. El archivo `resultados/analisis_final/consolidado_resultados.md` resume los resultados; el ZIP generado por el notebook es solo un paquete de transferencia y no necesita versionarse en Git.

## Resultados principales

Resultados de **900 oraciones sintéticas** por sistema:

| Sistema | Accuracy | F1 macro | Aciertos |
| --- | ---: | ---: | ---: |
| **SUT 4 — RecMS** | **0.8978** | **0.8968** | **808** |
| SUT 1 — Regresión logística | 0.7644 | 0.7644 | 688 |
| SUT 2 — LLM nativo | 0.7067 | 0.6795 | 636 |
| SUT 3 — RAG semántico | 0.6756 | 0.6594 | 608 |

**Recuperación de reglas:** la métrica operacional «Hit Rate@1 con abstención» considera correcto recuperar la regla objetivo en ejemplos `andino` o **no recuperar ninguna** en ejemplos `no_andino`. Bajo esa definición, el SUT 3 alcanza **0.3756** y el SUT 4, **0.9400**. Esta métrica no es idéntica al Hit@1 de recuperación convencional; las tasas por clase también se reportan por separado.

**Comparación principal:** SUT 4 supera a SUT 3 en **0.2222 de Accuracy** (IC del 95 % por bootstrap pareado: **[0.1933, 0.2522]**) y **0.2374 de F1 macro** (IC del 95 %: **[0.2057, 0.2710]**). La prueba exacta de McNemar para aciertos pareados, con corrección de Holm entre los seis pares, obtuvo **p ajustado ≈ 4.90 × 10⁻⁴⁴**.

**Análisis de errores:** en las 50 oraciones andinas asociadas a R05 (uso reportativo de «dice»), SUT 4 recuperó correctamente R05 en **50/50**, pero clasificó correctamente solo **1/50**. Esto muestra que una recuperación correcta no garantiza una clasificación final correcta.

## Protocolo estadístico y consideraciones metodológicas

- **SUT 1:** validación cruzada estratificada de 5 folds (720 casos de entrenamiento y 180 de prueba por fold); cada caso tiene una predicción *out-of-fold*.
- **SUT 2–4:** inferencia sobre las mismas 900 oraciones, sin ajuste supervisado de los parámetros de Llama con ese conjunto de datos.
- **Clasificación:** Accuracy, Precision, Recall y F1 por clase, F1 macro, matrices de confusión y métricas desagregadas por regla.
- **Recuperación:** calidad de la regla recuperada y abstención, globalmente y por regla.
- **Contrastes:** McNemar exacto sobre resultados de clasificación pareados, con corrección de Holm para seis pares; los valores p **no contrastan directamente F1 macro**.
- **Incertidumbre:** bootstrap pareado estratificado por clase, con **5000 remuestreos** y semilla **42**, para diferencias de Accuracy y F1 macro.
- **Selección de umbral:** el umbral **0.4** del SUT 3 se adoptó después de observar resultados iniciales; su evaluación debe interpretarse como **exploratoria/post hoc**, no como una calibración confirmatoria totalmente independiente.
- **Alcance:** los resultados corresponden a las nueve reglas y a los ejemplos sintéticos evaluados. No demuestran generalización automática a usos espontáneos del español peruano. Las explicaciones se conservaron para análisis cualitativo; **no se reporta una puntuación automática de fidelidad**.

## Uso de los resultados

Los archivos `resultados/sut*_predicciones.csv` son las salidas registradas del experimento. Para revisar los valores publicados, ejecuta primero `notebooks/analisis_resultados.ipynb` sobre esos CSV. Para estudiar cambios en la arquitectura o los modelos, conserva los resultados actuales y registra la nueva configuración por separado.
