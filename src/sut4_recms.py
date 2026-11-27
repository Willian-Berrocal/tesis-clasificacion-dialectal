"""SUT 4: clasificación con Llama y recuperación morfosintáctica (Stanza).

Los nueve detectores se conservan del notebook original, incluyendo el orden de
prioridad. Una coincidencia aporta la regla completa del mismo repositorio que
SUT 3; si no hay coincidencia, se envía el mensaje neutral compartido.

Uso:
    from src.sut4_recms import cargar_recuperador, recuperar_regla, clasificar_recms
"""

import re

import pandas as pd

from src.comun import RUTA_REGLAS, SIN_REGLA, cargar_reglas, clasificar_con_llama



def _forma(palabra):
    # Comparamos palabras sin distinguir mayúsculas de minúsculas.
    return palabra.text.casefold()


def _lema(palabra):
    # El lema agrupa formás flexiónadas: "llegó", "llegaron" -> "llegar".
    # Si Stanza no provee lema, utilizamos la forma original.
    return (palabra.lemma or palabra.text).casefold()


def _rasgos(palabra):
    # Convertimos rasgos de Stanza: "Gender=Fem|Number=Sing" -> diccionario.
    return dict(parte.split("=", 1) for parte in (palabra.feats or "").split("|") if "=" in parte)


def _es_verbo(palabra):
    # Incluimos verbos plenos y auxiliares.
    return palabra.upos in {"VERB", "AUX"}


def _forma_gerundio(palabra):
    # Primero revisamos la etiqueta gramatical y luego las terminaciónes
    # como respaldo cuando el analizador no identifica el gerundio.
    return (_rasgos(palabra).get("VerbForm") == "Ger"
            or (palabra.upos == "VERB" and bool(re.search(r"(ando|iendo|yendo)$", _forma(palabra)))))


def _es_participio(palabra):
    # Detectamos participios por el rasgo de Stanza o por terminación.
    return (_rasgos(palabra).get("VerbForm") == "Part"
            or (_es_verbo(palabra) and bool(re.search(r"(ado|ido|cho|to|so)$", _forma(palabra)))))


def _es_finito(palabra):
    # Un verbo finito tiene flexión de persona/tiempo, a diferencia del
    # infinitivo, gerundio y participio. Se mantiene el respaldo original.
    return (_es_verbo(palabra)
            and _rasgos(palabra).get("VerbForm", "Fin") == "Fin")


def _fragmento(oracion):
    # Evidencia completa cuando el patrón no se limita a tokens contiguos.
    return oracion.text


#R01
def _doble_posesivo(oracion):
    # Buscamos "de + poseedor + su/sus + sustantivo" en una ventana corta.
    # La ventana evita capturar frases alejadas o separadas por puntuación.
    ws = oracion.words
    for i, palabra in enumerate(ws):
        if _forma(palabra) != "de":
            continue
        for j in range(i + 2, min(i + 8, len(ws))):
            if any(ws[k].upos == "PUNCT" for k in range(i + 1, j)):
                break
            if _forma(ws[j]) not in {"su", "sus"}:
                continue
            poseedor = ws[i + 1:j]
            if (1 <= len(poseedor) <= 5
                and any(w.upos in {"NOUN", "PROPN", "PRON"} for w in poseedor)
                and not any(w.upos in {"VERB", "AUX", "SCONJ"} for w in poseedor)
                and j + 1 < len(ws)
                and ws[j + 1].upos in {"NOUN", "PROPN", "ADJ"}):
                return " ".join(w.text for w in ws[i:j+2])
    return None

#R02
def _orden_ov(oracion):
    ws = oracion.words

    for verbo in ws:
        # Trabajar únicamente con verbos conjugados.
        if verbo.upos != "VERB" or not _es_finito(verbo):
            continue

        rasgos_v = _rasgos(verbo)

        # Buscar posibles sujetos pospuestos que Stanza
        # haya etiquetado erróneamente como objetos.
        posteriores = [
            w for w in ws
            if w.id > verbo.id
            and w.head == verbo.id
            and w.deprel in {"obj", "iobj"}
            and (
                w.upos == "PROPN"
                or (
                    w.upos == "NOUN"
                    and any(x.head == w.id and x.deprel == "det"
                            for x in ws)
                )
            )
        ]

        # Comprobar si existe un sujeto pospuesto reconocido.
        sujeto_posterior = any(
            w.id > verbo.id
            and w.head == verbo.id
            and (w.deprel or "").startswith("nsubj")
            for w in ws
        )

        for objeto in ws:
            # Buscar sustantivos anteriores al verbo
            # que dependan sintácticamente de él.
            if (objeto.id >= verbo.id
                    or objeto.head != verbo.id
                    or objeto.upos not in {"NOUN", "PROPN"}):
                continue

            # Excluir construcciones interrogativas y relativas.
            if any(
                w.head == objeto.id
                and (
                    "PronType=Int" in (w.feats or "")
                    or "PronType=Rel" in (w.feats or "")
                )
                for w in ws
            ):
                continue

            # Caso 1: Stanza identifica correctamente el objeto.
            if objeto.deprel in {"obj", "iobj"}:
                return f"{objeto.text} → {verbo.text}"

            # Caso 2: objeto introducido por "a", etiquetado
            # como complemento argumental (obl:arg).
            if objeto.deprel == "obl:arg":
                tiene_a = any(
                    w.head == objeto.id
                    and w.deprel == "case"
                    and _forma(w) == "a"
                    for w in ws
                )

                if tiene_a and (
                    sujeto_posterior
                    or rasgos_v.get("Person") in {"1", "2"}
                ):
                    return f"{objeto.text} → {verbo.text}"

            # Caso 3: Stanza confunde el objeto con el sujeto.
            if not (objeto.deprel or "").startswith("nsubj"):
                continue

            numero = _rasgos(objeto).get("Number")

            # Un sintagma coordinado funciona normalmente
            # como plural: "Arroz y verduras".
            if any(
                w.head == objeto.id
                and w.deprel == "conj"
                and w.upos in {"NOUN", "PROPN", "PRON"}
                for w in ws
            ):
                numero = "Plur"

            # Un sustantivo de tercera persona no puede
            # ser sujeto de un verbo en primera o segunda.
            persona_distinta = rasgos_v.get("Person") in {"1", "2"}

            # Comprobar discordancia de número.
            numero_distinto = (
                numero is not None
                and rasgos_v.get("Number") is not None
                and numero != rasgos_v["Number"]
            )

            # Detectar nombres propios después del verbo,
            # posiblemente confundidos con objetos.
            nombre_posterior = any(
                w.upos == "PROPN" for w in posteriores
            )

            if (
                persona_distinta
                or (
                    posteriores
                    and (
                        numero_distinto
                        or (objeto.upos == "NOUN" and nombre_posterior)
                    )
                )
            ):
                return f"{objeto.text} → {verbo.text}"

    return None

#R03
def _subordinada(oracion):
    # "De lo que" debe aparecer al comienzo de la oración o tras puntuación.
    # Pedimos al menos dos verbos finitos posteriores: uno subordinado y
    # otro principal, para reducir coincidencias casuales.
    ws = oracion.words
    for i in range(len(ws) - 3):
        if [_forma(w) for w in ws[i:i+3]] != ["de", "lo", "que"]:
            continue
        if i > 0 and ws[i - 1].upos != "PUNCT":
            continue  # Evitar 'hablamos de lo que...'
        siguientes = ws[i+3:]
        if sum(_es_finito(w) for w in siguientes) >= 2:
            return " ".join(w.text for w in ws[i:i+6])
    return None

#R04
def _limitativo_nomas(oracion):
    # Heurística léxica: la presencia de "nomás" basta.
    # No se desambigua su función discursiva en este detectór.
    return "nomás" if any(_forma(w) == "nomás" for w in oracion.words) else None

#R05
def _reportativo_dice(oracion):
    # Solo consideramos "dice" al final de la oración (sin puntuación).
    # Exigimos otro verbo finito para distinguirlo de un simple "dice" verbal.
    ws = [w for w in oracion.words if w.upos != "PUNCT"]
    if (len(ws) >= 3
            and _forma(ws[-1]) == "dice"
            and any(_es_finito(w) and _lema(w) != "decir" for w in ws[:-1])):
        return _fragmento(oracion)
    return None

#R06
def _pues_enfatico(oracion):
    # Identificamos "pues"
    return "pues" if any(_forma(w) == "pues" for w in oracion.words) else None

#R07
def _discordancia(oracion):
    ws = oracion.words
    ids = {w.id: w for w in ws if isinstance(w.id, int)}

    def rasgos_nominales(w):
        # Obtener género y número. Si falta el género,
        # intentar recuperarlo del determinante.
        rasgos = _rasgos(w).copy()

        if "Gender" not in rasgos:
            for x in ws:
                if (x.head == w.id
                        and (x.deprel or "").startswith("det")):
                    genero = _rasgos(x).get("Gender")
                    if genero:
                        rasgos["Gender"] = genero
                        break

        return rasgos

    def rasgos_sujeto(sujeto):
        rasgos = rasgos_nominales(sujeto)

        # Identificar sujetos compuestos:
        # "El libro y el cuaderno" equivale a plural.
        coordinados = [
            w for w in ws
            if w.head == sujeto.id
            and (w.deprel or "").split(":")[0] == "conj"
            and w.upos in {"NOUN", "PROPN", "PRON"}
        ]

        if coordinados:
            rasgos["Number"] = "Plur"

            # En sujetos mixtos, el masculino es el género
            # de concordancia habitual.
            generos = [
                rasgos_nominales(w).get("Gender")
                for w in [sujeto, *coordinados]
            ]

            if "Masc" in generos:
                rasgos["Gender"] = "Masc"
            elif all(g == "Fem" for g in generos):
                rasgos["Gender"] = "Fem"
            else:
                rasgos.pop("Gender", None)

        return rasgos

    def distinto(a, b, categorias=("Gender", "Number")):
        # Solo comparar rasgos presentes en ambas palabras.
        return any(
            a.get(c) and b.get(c) and a[c] != b[c]
            for c in categorias
        )

    # 1. Concordancia directa: sustantivo-adjetivo
    # y sujeto-predicado adjetival.
    for w in ws:
        if w.upos not in {"ADJ", "VERB", "AUX"}:
            continue

        if w.upos in {"VERB", "AUX"} and not _es_participio(w):
            continue

        # Ejemplo: "La mochila pesado".
        if (w.deprel or "").startswith("amod") and w.head in ids:
            nombre = ids[w.head]

            if (nombre.upos in {"NOUN", "PROPN"}
                    and distinto(rasgos_nominales(nombre), _rasgos(w))):
                return f"{nombre.text} / {w.text}"

        # Ejemplo: "Mi tía estaba agotado".
        for sujeto in ws:
            if (sujeto.head == w.id
                    and (sujeto.deprel or "").startswith("nsubj")
                    and distinto(rasgos_sujeto(sujeto), _rasgos(w))):
                return f"{sujeto.text} / {w.text}"

    # 2. Concordancia sujeto-verbo.
    for verbo in ws:
        if not _es_finito(verbo):
            continue

        # Localizar sujetos mediante dependencias sintácticas.
        sujetos = [
            w for w in ws
            if w.head == verbo.id
            and (w.deprel or "").startswith("nsubj")
        ]

        # Ejemplo: "Los zapatos sigue".
        # Solo los rasgos de número se comparan con el verbo.
        for sujeto in sujetos:
            if distinto(
                rasgos_sujeto(sujeto),
                _rasgos(verbo),
                ("Number",)
            ):
                return f"{sujeto.text} / {verbo.text}"

        # 3. Concordancia sujeto-adjetivo predicativo.
        # Ejemplo: "Mi tía llegó agotado".
        # El sujeto y el adjetivo dependen del mismo verbo.
        for w in ws:
            if (w.head != verbo.id
                    or (w.deprel or "").split(":")[0]
                    not in {"obj", "xcomp", "advcl", "dep"}):
                continue

            es_adjetivo = w.upos == "ADJ" or _es_participio(w)

            # Respaldo para adjetivos etiquetados como NOUN.
            # Ejemplo: "Las flores quedaron marchito".
            if not es_adjetivo and w.upos == "NOUN":
                sin_determinante = not any(
                    x.head == w.id
                    and (x.deprel or "").startswith("det")
                    for x in ws
                )

                # Heurística limitada a ciertas terminaciones.
                es_adjetivo = (
                    w.id == verbo.id + 1
                    and sin_determinante
                    and bool(re.search(
                        r"(?:it|os|iv|ur)[oa]s?$",
                        _forma(w)
                    ))
                )

            if es_adjetivo:
                for sujeto in sujetos:
                    if distinto(
                        rasgos_sujeto(sujeto),
                        _rasgos(w)
                    ):
                        return f"{sujeto.text} / {w.text}"

    return None

#R08
def _lo_espurio(oracion):
    # Detectamos "lo" inmediatamente antes de un verbo de una lista
    # de usos predominantemente intransitivos. No examinamos cualquier "lo".
    # Solo verbos predominantemente intransitivos, en estos usos.
    # No se incluye 'dormir': 'lo durmió' puede ser transitivo normal.
    intransitivos = {
        "llegar", "morir", "caer", "amanecer", "venir", "salir",
        "entrar", "nacer", "regresar", "volver", "suceder", "ocurrir",
    }
    ws = oracion.words
    for i in range(len(ws) - 1):
        if (_forma(ws[i]) == "lo"
                and _lema(ws[i + 1]) in intransitivos
                and _es_verbo(ws[i + 1])):
            return f"{ws[i].text} {ws[i + 1].text}"
    return None

#R09
def _gerundio_andino(oracion):
    # Identificamos gerundio antes de una forma finita de "estar".
    # Permitimos solo ciertos elementos breves entre ambos, hasta 3 tokens.
    ws = oracion.words
    elementos_intermedios = {"nomás", "nomas", "me", "te", "se", "nos", "le", "les", "ya"}
    for i, w in enumerate(ws):
        if not _forma_gerundio(w):
            continue
        for j in range(i + 1, min(i + 5, len(ws))):
            intermedios = ws[i+1:j]
            if (all(_forma(x) in elementos_intermedios for x in intermedios)
                    and _lema(ws[j]) == "estar"
                    and _es_finito(ws[j])):
                return " ".join(x.text for x in ws[i:j+1])
    return None


# Prioridad ORIGINAL del notebook: primera coincidencia entre las nueve reglas.
# El orden es deliberado y NO corresponde al orden numérico de sus identificadores.
DETECTORES = [
    ("R01", _doble_posesivo),
    ("R08", _lo_espurio),
    ("R03", _subordinada),
    ("R09", _gerundio_andino),
    ("R05", _reportativo_dice),
    ("R02", _orden_ov),
    ("R07", _discordancia),
    ("R04", _limitativo_nomas),
    ("R06", _pues_enfatico),
]


def cargar_recuperador(ruta_reglas=RUTA_REGLAS, analizador=None):
    """Carga Stanza una sola vez y el mismo repositorio de reglas del SUT 3.

    `analizador` permite inyectar un objeto compatible para pruebas sin Stanza.
    Requiere instalar stanza y descargar sus recursos de español previamente.
    """
    if analizador is None:
        import stanza

        analizador = stanza.Pipeline(
            lang="es",
            processors="tokenize,mwt,pos,lemma,depparse",
            use_gpu=False,
            verbose=False,
        )

    reglas = cargar_reglas(ruta_reglas)
    return {"analizador": analizador, "reglas": reglas}


def recuperar_regla(texto, recuperador):
    """Recupera una regla completa o ninguna; no usa etiquetas ni regla_id gold."""
    if not isinstance(texto, str):
        raise TypeError("La frase debe ser de tipo str")

    if texto.strip():
        documento = recuperador["analizador"](texto)
        for regla_id, detector in DETECTORES:
            for oracion in documento.sentences:
                evidencia = detector(oracion)
                if evidencia is not None:
                    return {
                        "regla_id": regla_id,
                        "texto": recuperador["reglas"][regla_id],
                        "evidencia": evidencia,
                    }

    return {"regla_id": None, "texto": None, "evidencia": None}


def clasificar_recms(texto, generador, recuperador):
    """Mismo generador y formato de contexto que SUT 3, sin sesgar la etiqueta."""
    recuperada = recuperar_regla(texto, recuperador)
    contexto = recuperada["texto"] if recuperada["regla_id"] else SIN_REGLA
    resultado = clasificar_con_llama(texto, generador, contexto=contexto)
    resultado["regla_recuperada"] = recuperada["regla_id"]
    resultado["evidencia_recuperacion"] = recuperada["evidencia"]
    return resultado


def predecir_dataset(df, generador, recuperador):
    """Clasifica las filas entregadas, incluidas muestras pequeñas de prueba."""
    salida = df[["case_id", "regla_id", "test_case", "label"]].rename(
        columns={"test_case": "texto", "label": "etiqueta_real"}
    ).copy()
    salida.insert(1, "sut", "sut4_recms")

    predicciones = [clasificar_recms(texto, generador, recuperador) for texto in salida["texto"]]
    salida["prediccion"] = [r["prediccion"] for r in predicciones]
    salida["regla_recuperada"] = [r["regla_recuperada"] for r in predicciones]
    salida["evidencia_recuperacion"] = [r["evidencia_recuperacion"] for r in predicciones]
    salida["explicacion"] = [r["explicacion"] for r in predicciones]
    salida["respuesta_cruda"] = [r["respuesta_cruda"] for r in predicciones]
    return salida
