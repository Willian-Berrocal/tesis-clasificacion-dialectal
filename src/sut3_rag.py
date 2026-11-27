"""SUT 3: clasificación con Llama y recuperación semántica (BGE-M3 + FAISS).

Importar desde la raíz del proyecto:
    from src.comun import cargar_llama
    from src.sut3_rag import cargar_recuperador, clasificar_rag, predecir_dataset

Cada regla del repositorio es una unidad de recuperación. Se selecciona, como
máximo, una regla (k=1) cuando supera el umbral de similitud configurado.
"""

import numpy as np
import pandas as pd

from src.comun import RUTA_REGLAS, SIN_REGLA, cargar_reglas, clasificar_con_llama

MODELO_EMBEDDINGS = "BAAI/bge-m3"


def cargar_recuperador(umbral_similitud, ruta_reglas=RUTA_REGLAS, dispositivo="cpu"):
    """Construye una sola vez el índice FAISS sobre las nueve reglas completas.

    El umbral se aplica a similitud coseno, NO a una probabilidad de certeza.
    Debe fijarse con ejemplos independientes antes de evaluar las 900 frases.
    """
    import faiss
    from sentence_transformers import SentenceTransformer

    if not -1.0 <= umbral_similitud <= 1.0:
        raise ValueError("El umbral de similitud coseno debe estar entre -1 y 1.")

    reglas = cargar_reglas(ruta_reglas)
    ids = list(reglas)
    textos = [reglas[regla_id] for regla_id in ids]
    codificador = SentenceTransformer(MODELO_EMBEDDINGS, device=dispositivo)
    vectores = codificador.encode(
        textos,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectores = np.asarray(vectores, dtype=np.float32)
    indice = faiss.IndexFlatIP(vectores.shape[1])
    indice.add(vectores)

    return {
        "codificador": codificador,
        "indice": indice,
        "ids": ids,
        "reglas": reglas,
        "umbral_similitud": umbral_similitud,
    }


def recuperar_regla(texto, recuperador):
    """Devuelve 0 o 1 regla y el score de la mejor candidata.

    Si la similitud no alcanza el umbral, regla_id y texto serán None.
    En ese caso no se introduce ninguna regla al prompt del LLM.
    """
    vector = recuperador["codificador"].encode(
        [texto], convert_to_numpy=True, normalize_embeddings=True
    )
    vector = np.asarray(vector, dtype=np.float32)
    puntuaciones, posiciones = recuperador["indice"].search(vector, 1)
    score = float(puntuaciones[0, 0])
    posicion = int(posiciones[0, 0])

    if score < recuperador["umbral_similitud"]:
        return {"regla_id": None, "texto": None, "score": score}

    regla_id = recuperador["ids"][posicion]
    return {
        "regla_id": regla_id,
        "texto": recuperador["reglas"][regla_id],
        "score": score,
    }


def clasificar_rag(texto, generador, recuperador):
    """Clasifica la frase con el mismo Llama/prompt usado por los otros SUT."""
    recuperada = recuperar_regla(texto, recuperador)
    contexto = recuperada["texto"] if recuperada["regla_id"] else SIN_REGLA
    resultado = clasificar_con_llama(texto, generador, contexto=contexto)
    resultado["regla_recuperada"] = recuperada["regla_id"]
    resultado["score_recuperacion"] = recuperada["score"]
    return resultado


def predecir_dataset(df, generador, recuperador):
    """Predice las filas entregadas; permite ejecutar muestras pequeñas o el dataset completo."""
    salida = df[["case_id", "regla_id", "test_case", "label"]].rename(
        columns={"test_case": "texto", "label": "etiqueta_real"}
    ).copy()
    salida.insert(1, "sut", "sut3_rag")

    predicciones = [clasificar_rag(texto, generador, recuperador) for texto in salida["texto"]]
    salida["prediccion"] = [r["prediccion"] for r in predicciones]
    salida["regla_recuperada"] = [r["regla_recuperada"] for r in predicciones]
    salida["score_recuperacion"] = [r["score_recuperacion"] for r in predicciones]
    salida["explicacion"] = [r["explicacion"] for r in predicciones]
    salida["respuesta_cruda"] = [r["respuesta_cruda"] for r in predicciones]
    return salida
