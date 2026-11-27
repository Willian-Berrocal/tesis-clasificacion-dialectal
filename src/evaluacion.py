"""Evaluación común de los cuatro SUT.

Clasificación: accuracy, precision/recall/F1 por clase, F1 macro,
matriz de confusión y resultados por regla.
Recuperación: Hit Rate@1. En frases andinas, acierta solo si se recupera
la regla objetivo; en frases no andinas, acierta solo si no se recupera ninguna.
Comparación pareada: prueba exacta de McNemar.

Las funciones reciben los DataFrames guardados por los SUT. No entrenan modelos.
"""

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)

ETIQUETAS = ["andino", "no_andino"]


def evaluar_clasificacion(df):
    """Devuelve métricas globales y por clase de un SUT."""
    real = df["etiqueta_real"]
    pred = df["prediccion"]
    precision, recall, f1, soporte = precision_recall_fscore_support(
        real, pred, labels=ETIQUETAS, zero_division=0
    )
    return {
        "n": len(df),
        "accuracy": float(accuracy_score(real, pred)),
        "precision_macro": float(np.mean(precision)),
        "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        "por_clase": pd.DataFrame({
            "clase": ETIQUETAS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "soporte": soporte,
        }),
    }


def matriz_confusion(df):
    """Filas: etiqueta real; columnas: predicción."""
    matriz = confusion_matrix(
        df["etiqueta_real"], df["prediccion"], labels=ETIQUETAS
    )
    return pd.DataFrame(matriz, index=ETIQUETAS, columns=ETIQUETAS)


def evaluar_por_regla(df):
    """Accuracy y F1 macro para cada regla objetivo (incluye sus contrastes)."""
    filas = []
    for regla_id, grupo in df.groupby("regla_id", sort=True):
        filas.append({
            "regla_id": regla_id,
            "n": len(grupo),
            "accuracy": accuracy_score(grupo["etiqueta_real"], grupo["prediccion"]),
            "f1_macro": f1_score(
                grupo["etiqueta_real"], grupo["prediccion"],
                labels=ETIQUETAS, average="macro", zero_division=0,
            ),
        })
    return pd.DataFrame(filas)


def evaluar_folds(df):
    """F1 macro por fold y media ± desviación estándar muestral para SUT 1."""
    filas = []
    for fold, grupo in df.groupby("fold", sort=True):
        filas.append({
            "fold": fold,
            "n": len(grupo),
            "f1_macro": f1_score(
                grupo["etiqueta_real"], grupo["prediccion"],
                labels=ETIQUETAS, average="macro", zero_division=0,
            ),
        })
    tabla = pd.DataFrame(filas)
    return tabla, {
        "media_f1_macro": float(tabla["f1_macro"].mean()),
        "desv_f1_macro": float(tabla["f1_macro"].std(ddof=1)),
    }


def evaluar_recuperacion(df):
    """Hit@1: andino -> regla correcta; no_andino -> ninguna regla.

    El campo regla_id de no_andino es solo metadato de agrupación;
    NO se exige recuperar esa regla en los ejemplos de contraste.
    """
    real = df["etiqueta_real"]
    recuperada = df["regla_recuperada"]
    sin_regla = recuperada.isna() | recuperada.astype(str).str.strip().eq("")
    acierto = (
        (real.eq("andino") & recuperada.eq(df["regla_id"]))
        | (real.eq("no_andino") & sin_regla)
    )
    return {
        "n": len(df),
        "hit_rate_1": float(acierto.mean()),
        "hit_rate_1_andino": float(acierto[real.eq("andino")].mean()),
        "hit_rate_1_no_andino": float(acierto[real.eq("no_andino")].mean()),
        "aciertos": int(acierto.sum()),
    }


def comparar_mcnemar(df_a, df_b):
    """Prueba exacta de McNemar con predicciones del mismo conjunto de casos.

    Las filas se emparejan por case_id, no por el orden en los CSV.
    """
    columnas = ["case_id", "etiqueta_real", "prediccion"]
    a = df_a[columnas].rename(columns={
        "etiqueta_real": "real_a", "prediccion": "pred_a"
    })
    b = df_b[columnas].rename(columns={
        "etiqueta_real": "real_b", "prediccion": "pred_b"
    })
    if a["case_id"].duplicated().any() or b["case_id"].duplicated().any():
        raise ValueError("Los case_id deben ser únicos dentro de cada SUT.")
    pares = a.merge(b, on="case_id", how="inner", validate="one_to_one")
    if len(pares) != len(a) or len(pares) != len(b):
        raise ValueError("Los clasificadores deben evaluar los mismos case_id.")
    if not pares["real_a"].equals(pares["real_b"]):
        raise ValueError("Las etiquetas reales difieren entre los clasificadores.")

    acierto_a = pares["pred_a"].eq(pares["real_a"])
    acierto_b = pares["pred_b"].eq(pares["real_b"])
    solo_a = int((acierto_a & ~acierto_b).sum())
    solo_b = int((~acierto_a & acierto_b).sum())
    discordantes = solo_a + solo_b
    p_valor = (
        float(binomtest(solo_a, discordantes, p=0.5).pvalue)
        if discordantes else 1.0
    )
    return {
        "n": len(pares),
        "ambos_correctos": int((acierto_a & acierto_b).sum()),
        "solo_a_correcto": solo_a,
        "solo_b_correcto": solo_b,
        "ambos_incorrectos": int((~acierto_a & ~acierto_b).sum()),
        "discordantes": discordantes,
        "p_valor_exacto": p_valor,
    }
