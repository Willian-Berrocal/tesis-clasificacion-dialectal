# Resultados del experimento (900 oraciones sintéticas)

## Configuración

- Clases: andino / no_andino (450 ejemplos por clase).
- Nueve reglas R01–R09, 50 casos por combinación de regla y clase.
- SUT 1: embeddings spaCy + regresión logística; cinco folds estratificados (predicciones fuera del fold de entrenamiento).
- SUT 2: Llama 3.1 8B Instruct sin reglas recuperadas.
- SUT 3: BGE-M3 + FAISS + Llama; recuperación top-1 y umbral coseno 0.4.
- SUT 4: Stanza + detectores heurísticos en orden de prioridad + Llama; recuperación top-1.
- **Nota de protocolo:** el umbral 0.4 de SUT 3 se escogió tras una evaluación inicial (análisis exploratorio).

## Clasificación global

| sistema                     |   accuracy |   precision_macro |   recall_macro |   f1_macro |   aciertos |   errores |
|:----------------------------|-----------:|------------------:|---------------:|-----------:|-----------:|----------:|
| SUT 1 - Regresión logística |     0.7644 |            0.7644 |         0.7644 |     0.7644 |        688 |       212 |
| SUT 2 - LLM nativo          |     0.7067 |            0.8126 |         0.7067 |     0.6795 |        636 |       264 |
| SUT 3 - RAG semántico       |     0.6756 |            0.7167 |         0.6756 |     0.6594 |        608 |       292 |
| SUT 4 - RecMS               |     0.8978 |            0.9136 |         0.8978 |     0.8968 |        808 |        92 |

## Validación cruzada de SUT 1

- Media del F1 macro de cinco folds: 0.7643; DE muestral: 0.0210.

## Recuperación con abstención

| sistema               |   hit_at_1_abstencion |   reglas_correctas_andino |   reglas_incorrectas_andino |   sin_regla_andino |   abstenciones_correctas_no_andino |   reglas_indebidas_no_andino |   recall_regla_andino |   abstencion_no_andino |
|:----------------------|----------------------:|--------------------------:|----------------------------:|-------------------:|-----------------------------------:|-----------------------------:|----------------------:|-----------------------:|
| SUT 3 - RAG semántico |                0.3756 |                       115 |                         143 |                192 |                                223 |                          227 |                0.2556 |                 0.4956 |
| SUT 4 - RecMS         |                0.9400 |                       403 |                          14 |                 33 |                                443 |                            7 |                0.8956 |                 0.9844 |

La definición operacional valora recuperar la regla objetivo en frases andinas y no recuperar regla en frases no andinas.

## Contrastes pareados

- McNemar exacto SUT 3 vs SUT 4, valor p ajustado de Holm: 4.9027e-44.
- Diferencia en accuracy SUT 4 − SUT 3: 0.2222; IC95 bootstrap [0.1933, 0.2522].
- Diferencia en F1 macro SUT 4 − SUT 3: 0.2374; IC95 bootstrap [0.2057, 0.2710].
- Comparaciones por pares: 6; significativas tras Holm: 5.

## Caso R05

En las 50 oraciones andinas asignadas a R05, SUT 4 recupera R05 en 50/50 casos, pero clasifica correctamente solo 1/50. La recuperación pertinente no garantiza una etiqueta correcta del LLM.

## Alcance

Resultados del conjunto sintético estudiado. No se presupone generalización a otros materiales dialectales ni evaluación automática de fidelidad de explicaciones.
