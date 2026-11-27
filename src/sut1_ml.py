"""SUT 1: embeddings de palabras (spaCy) + regresión logística.

Usa validación cruzada de 5 folds estratificada por (regla_id, label).
Los campos regla_id y label nunca se incorporan a las características de entrada.

Ejecutar desde la raíz del proyecto:
    python src/sut1_ml.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import spacy
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


RAIZ = Path(__file__).resolve().parent.parent
DATOS = RAIZ / "datos" / "ejemplos_dialectos_900.csv"
SALIDA = RAIZ / "resultados" / "sut1_ml_predicciones.csv"
MODELO_EMBEDDINGS = "es_core_news_md"
SEMILLA = 42
N_FOLDS = 5
ETIQUETAS = ("andino", "no_andino")


def cargar_datos(ruta=DATOS):
    """Lee el dataset y valida las 900 frases del protocolo."""
    df = pd.read_csv(ruta, encoding="utf-8-sig")
    columnas = {"case_id", "regla_id", "test_case", "label"}
    if not columnas.issubset(df.columns):
        raise ValueError(f"Columnas obligatorias: {sorted(columnas)}")
    if df[list(columnas)].isna().any().any():
        raise ValueError("Hay valores ausentes en columnas obligatorias.")

    df = df.copy()
    df["test_case"] = df["test_case"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["regla_id"] = df["regla_id"].astype(str).str.strip()

    if len(df) != 900 or df["case_id"].nunique() != 900:
        raise ValueError("Se esperan 900 frases con case_id únicos.")
    if (df["test_case"] == "").any() or df["test_case"].duplicated().any():
        raise ValueError("Hay frases vacías o duplicadas.")
    if set(df["label"]) != set(ETIQUETAS):
        raise ValueError(f"Las etiquetas deben ser {ETIQUETAS}.")
    frecuencias = df.groupby(["regla_id", "label"]).size()
    if len(frecuencias) != 18 or not (frecuencias == 50).all():
        raise ValueError("Se esperan 50 frases por cada combinación de regla y clase.")
    return df


def crear_folds(df):
    """Genera (train_idx, test_idx) con 10 casos de cada estrato por fold."""
    estratos = df["regla_id"] + "|" + df["label"]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEMILLA)
    for indices_train, indices_test in cv.split(df["test_case"], estratos):
        yield indices_train, indices_test


def cargar_vectorizador():
    """Carga únicamente el tokenizador y los vectores estáticos de spaCy."""
    nlp = spacy.load(
        MODELO_EMBEDDINGS,
        exclude=["tok2vec", "morphologizer", "parser", "attribute_ruler", "lemmatizer", "ner"],
    )
    if nlp.vocab.vectors_length != 300:
        raise ValueError("Se esperaba un modelo spaCy de 300 dimensiones.")
    return nlp


def vectorizar_textos(nlp, frases):
    """Promedia embeddings por frase sin eliminar palabras funcionales."""
    if isinstance(frases, str):
        frases = [frases]
    frases = list(frases)
    dimension = nlp.vocab.vectors_length
    vectores = []
    for doc in nlp.pipe(frases, batch_size=64):
        validos = [
            token.vector
            for token in doc
            if not token.is_space
            and not token.is_punct
            and token.has_vector
            and token.vector_norm > 0
        ]
        emb = np.mean(np.stack(validos), axis=0) if validos else np.zeros(dimension)
        vectores.append(emb)
    return np.asarray(vectores, dtype=np.float32).reshape(-1, dimension)


def crear_modelo():
    """Escala dentro de cada fold y entrena regresión logística."""
    return Pipeline([
        ("escalador", StandardScaler()),
        ("clasificador", LogisticRegression(max_iter=2000, random_state=SEMILLA)),
    ])


def entrenar_modelo(df, nlp):
    """Entrena con todos los datos recibidos; útil para evaluación externa posterior."""
    modelo = crear_modelo()
    modelo.fit(vectorizar_textos(nlp, df["test_case"]), df["label"])
    return modelo


def predecir_frases(modelo, nlp, frases):
    """Permite probar nuevas frases sin alterar el código del clasificador."""
    X = vectorizar_textos(nlp, frases)
    return modelo.predict(X)


def ejecutar_cv(df, nlp):
    """Devuelve predicciones out-of-fold y F1 macro de cada fold."""
    X = vectorizar_textos(nlp, df["test_case"])
    y = df["label"].to_numpy()
    predicciones = np.empty(len(df), dtype=object)
    prob_andino = np.full(len(df), np.nan, dtype=float)
    folds = np.zeros(len(df), dtype=int)
    f1_folds = []

    for numero, (train_idx, test_idx) in enumerate(crear_folds(df), start=1):
        modelo = crear_modelo()
        modelo.fit(X[train_idx], y[train_idx])
        predicciones[test_idx] = modelo.predict(X[test_idx])
        indice_andino = list(modelo.classes_).index("andino")
        prob_andino[test_idx] = modelo.predict_proba(X[test_idx])[:, indice_andino]
        folds[test_idx] = numero
        f1 = f1_score(y[test_idx], predicciones[test_idx], average="macro", zero_division=0)
        f1_folds.append(f1)
        print(f"Fold {numero}: 720 train / 180 test | F1 macro: {f1:.4f}")

    salida = df[["case_id", "regla_id", "test_case", "label"]].rename(columns={
        "test_case": "texto", "label": "etiqueta_real",
    }).copy()
    salida.insert(1, "sut", "sut1_ml")
    salida["fold"] = folds
    salida["prediccion"] = predicciones
    salida["prob_andino"] = prob_andino
    print(f"F1 macro (media ± DE): {np.mean(f1_folds):.4f} ± {np.std(f1_folds, ddof=1):.4f}")
    return salida, f1_folds


def main():
    df = cargar_datos()
    nlp = cargar_vectorizador()
    resultados, _ = ejecutar_cv(df, nlp)
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    resultados.to_csv(SALIDA, index=False, encoding="utf-8-sig")
    print(f"Predicciones guardadas: {SALIDA} ({len(resultados)} filas)")


if __name__ == "__main__":
    main()
