"""Componentes compartidos por SUT 2 (LLM), SUT 3 (RAG) y SUT 4 (RecMS).

El generador se carga una única vez desde el notebook principal en Google Colab.
Los tres SUT utilizan el mismo modelo, plantilla y formato de salida.
"""

import re
from pathlib import Path

MODELO_LLM = "meta-llama/Llama-3.1-8B-Instruct"
MAX_NEW_TOKENS = 256

RAIZ = Path(__file__).resolve().parent.parent
RUTA_REGLAS = RAIZ / "datos" / "reglas_andino.txt"

# Se utilizará cuando un recuperador no seleccione ninguna regla.
SIN_REGLA = "No se identificaron coincidencias con las reglas lingüísticas disponibles."


def cargar_llama():
    """Carga Llama 3.1 8B Instruct cuantizado en 4 bits (requiere GPU)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

    cuantizacion = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(MODELO_LLM)
    modelo = AutoModelForCausalLM.from_pretrained(
        MODELO_LLM,
        quantization_config=cuantizacion,
        device_map="auto",
    )
    return pipeline("text-generation", model=modelo, tokenizer=tokenizer)


def cargar_reglas(ruta=RUTA_REGLAS):
    """Lee las nueve reglas y conserva su texto original con ID, nombre y ejemplos."""
    texto = Path(ruta).read_text(encoding="utf-8")
    separador = "=" * 50 + "\nREGLA\n" + "=" * 50
    reglas = {}

    for fragmento in texto.split(separador):
        fragmento = fragmento.strip()
        if not fragmento:
            continue
        coincidencia = re.search(r"\bID:\s*(R\d{2})\b", fragmento)
        if not coincidencia:
            raise ValueError("Una regla del documento no tiene un ID válido.")
        regla_id = coincidencia.group(1)
        if regla_id in reglas:
            raise ValueError(f"ID de regla repetido: {regla_id}")
        reglas[regla_id] = separador + "\n\n" + fragmento

    return reglas


def construir_prompt(texto, contexto=None):
    """Comparte las instrucciones; solo los SUT con RAG añaden contexto."""
    prompt = (
        "Eres un especialista en lingüística del español peruano.\n\n"
        "Tu tarea es clasificar el siguiente texto utilizando únicamente "
        "una de las siguientes categorías:\n\n"
        "- Andino\n"
        "- No andino\n\n"
        'Examina las construcciones gramaticales observables en el texto y fundamenta tu decisión en ellas, evitando utilizar únicamente asociaciones de vocabulario o temática.\nSi se proporciona una regla lingüística recuperada, comprueba si el fenómeno que describe efectivamente aparece en el texto. La recuperación de una regla no implica necesariamente que se cumpla.\n'
    )
    if contexto is not None:
        prompt += "\nContexto lingüístico recuperado:\n\n" + str(contexto) + "\n"

    prompt += (
        "\nTexto:\n\n" + str(texto) + "\n\n"
        "Responde exactamente con dos líneas, sin texto adicional.\n"
        "Para la primera línea, escribe solo Andino o No andino después de los dos puntos:\n\n"
        "Clasificación: [etiqueta]\n"
        "Explicación: [explicación breve de la decisión]"
    )
    return prompt


def generar_respuesta(generador, prompt):
    """Genera una respuesta con decodificación determinista."""
    mensajes = [{"role": "user", "content": prompt}]
    salida = generador(
        mensajes,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        return_full_text=False,
    )
    respuesta = salida[0]["generated_text"]
    if isinstance(respuesta, list):
        respuesta = respuesta[-1]["content"]
    return respuesta.strip()


def interpretar_respuesta(respuesta):
    """Extrae únicamente las clases permitidas, además de la explicación."""
    match = re.search(
        r"(?im)^[ \t*]*Clasificaci[oó]n[ \t*]*:[ \t*]*(No[ _-]andino|Andino)[ \t*.]*$",
        respuesta,
    )
    if match is None:
        raise ValueError(f"Clasificación ausente o inválida: {respuesta!r}")

    prediccion = "no_andino" if match.group(1).lower().startswith("no") else "andino"
    explicacion_match = re.search(
        r"(?ims)^[ \t*]*Explicaci[oó]n[ \t*]*:[ \t*]*(.*)$",
        respuesta,
    )
    explicacion = explicacion_match.group(1).strip() if explicacion_match else ""
    return {
        "prediccion": prediccion,
        "explicacion": explicacion,
        "respuesta_cruda": respuesta,
    }


def clasificar_con_llama(texto, generador, contexto=None):
    """Clasifica con o sin contexto, conservando idéntico proceso generativo."""
    prompt = construir_prompt(texto, contexto=contexto)
    respuesta = generar_respuesta(generador, prompt)
    return interpretar_respuesta(respuesta)
