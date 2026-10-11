import re

from langchain_core.documents import Document

# Un título ocupa una línea entera: "== History ==", "=== Formation: 1966–1968 ===", ...
# Grupo 1: los "=" (su longitud es el nivel). Grupo 2: el texto del título.
PATRON_TITULO = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", re.MULTILINE)

# Secciones de nivel 2 que no aportan nada al RAG (listas de enlaces y bibliografía)
SECCIONES_DESCARTADAS = {"See also", "References", "Citations", "Notes", "Sources",
                         "External links", "Further reading", "Bibliography"}

TAMANO_MINIMO = 300
TAMANO_MAXIMO = 2000
def dividir_en_secciones(texto: str) -> list[tuple[str, str]]:
    """Divide un artículo en [(título_completo, cuerpo)].

    El título completo incluye los niveles superiores, p. ej. "History: Formation: 1966–1968".
    El texto anterior al primer título es la introducción del artículo.
    """
    secciones = []
    jerarquia = {}            # nivel → título actual de ese nivel
    titulo_actual = "Introduction"
    inicio_cuerpo = 0

    for coincidencia in PATRON_TITULO.finditer(texto):
        # Guardamos el cuerpo de la sección anterior (lo que hay entre su título y este)
        secciones.append((titulo_actual, texto[inicio_cuerpo:coincidencia.start()].strip()))

        nivel = len(coincidencia.group(1))
        jerarquia[nivel] = coincidencia.group(2)
        # Al abrir un título, los subniveles del anterior dejan de valer
        for nivel_inferior in [n for n in jerarquia if n > nivel]:
            del jerarquia[nivel_inferior]

        titulo_actual = ": ".join(jerarquia[n] for n in sorted(jerarquia))
        inicio_cuerpo = coincidencia.end()

    secciones.append((titulo_actual, texto[inicio_cuerpo:].strip()))

    return [
        (titulo, cuerpo) for titulo, cuerpo in secciones
        if cuerpo and titulo.split(": ")[0] not in SECCIONES_DESCARTADAS
    ]


def chunking(texto: str) -> list[Document]:
    secciones = dividir_en_secciones(texto)
    # TODO (tuyo):
    #  1. Si la PRIMERA sección es corta (no tiene anterior), juntarla con la siguiente.
    #  2. Secciones muy largas → dividir por párrafos (separados por "\n") hasta TAMANO_MAXIMO.
    #  3. Crear un Document por fragmento: cabecera "Artista — Título: " + texto en page_content,
    #     y artist_id, artista, seccion y url en metadata (están en data/wiki/indice.json).
    #     Pista: para eso la función necesitará recibir también los datos del artista.

    # Las secciones cortas se juntan con la anterior
    resultado = []
    for titulo, cuerpo in secciones:
        if len(cuerpo) < TAMANO_MINIMO and resultado:
            titulo_anterior, cuerpo_anterior = resultado[-1]
            resultado[-1] = (f"{titulo_anterior} / {titulo}", f"{cuerpo_anterior}\n{cuerpo}")
        else:
            resultado.append((titulo, cuerpo))

    return resultado


if __name__ == "__main__":
    from pathlib import Path

    texto = (Path(__file__).parent / "data" / "wiki" / "22_Led_Zeppelin.txt").read_text(encoding="utf-8")

    for titulo, cuerpo in chunking(texto):
        print(f"{len(cuerpo):>6} caracteres | {titulo}")
