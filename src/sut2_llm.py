"""SUT 2: clasificación dialectal mediante Llama sin recuperación de reglas.

Importar desde el directorio raíz del proyecto:
    from src.comun import cargar_llama
    from src.sut2_llm import clasificar_llm, predecir_dataset

El generador se carga una sola vez y se comparte con SUT 3 y SUT 4.
"""

import pandas as pd

from src.comun import clasificar_con_llama


def clasificar_llm(texto, generador):
    """Predice 'andino' o 'no_andino' sin consultar reglas externas."""
    return clasificar_con_llama(texto, generador)


def predecir_dataset(df, generador):
    """Genera una fila por frase, con el mismo esquema básico del SUT 1."""
    salida = df[["case_id", "regla_id", "test_case", "label"]].rename(
        columns={"test_case": "texto", "label": "etiqueta_real"}
    ).copy()
    salida.insert(1, "sut", "sut2_llm")

    predicciones = [clasificar_llm(texto, generador) for texto in salida["texto"]]
    salida["prediccion"] = [r["prediccion"] for r in predicciones]
    salida["explicacion"] = [r["explicacion"] for r in predicciones]
    salida["respuesta_cruda"] = [r["respuesta_cruda"] for r in predicciones]
    return salida
