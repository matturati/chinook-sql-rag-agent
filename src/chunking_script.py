import re

from langchain_core.documents import Document

# A heading takes a whole line: "== History ==", "=== Formation: 1966–1968 ===", ...
# Group 1: the "=" signs (their length is the level). Group 2: the heading text.
PATRON_TITULO = re.compile(r"^(={2,4})\s*(.+?)\s*\1\s*$", re.MULTILINE)

# Level-2 sections that add nothing to the RAG (lists of links and bibliography)
SECCIONES_DESCARTADAS = {"See also", "References", "Citations", "Notes", "Sources",
                         "External links", "Further reading", "Bibliography"}

TAMANO_MINIMO = 300
TAMANO_MAXIMO = 2000

def dividir_en_secciones(texto: str) -> list[tuple[str, str]]:
    """Splits an article into [(full_title, body)].

    The full title includes the parent levels, e.g. "History: Formation: 1966–1968".
    The text before the first heading is the article's introduction.
    """
    secciones = []
    jerarquia = {}            # level → current title at that level
    titulo_actual = "Introduction"
    inicio_cuerpo = 0

    for coincidencia in PATRON_TITULO.finditer(texto):
        # Save the body of the previous section (the text between its heading and this one)
        secciones.append((titulo_actual, texto[inicio_cuerpo:coincidencia.start()].strip()))

        nivel = len(coincidencia.group(1))
        jerarquia[nivel] = coincidencia.group(2)
        # A new heading closes the sublevels of the previous one
        for nivel_inferior in [n for n in jerarquia if n > nivel]:
            del jerarquia[nivel_inferior]

        titulo_actual = ": ".join(jerarquia[n] for n in sorted(jerarquia))
        inicio_cuerpo = coincidencia.end()

    secciones.append((titulo_actual, texto[inicio_cuerpo:].strip()))

    return [
        (titulo, cuerpo) for titulo, cuerpo in secciones
        if cuerpo and titulo.split(": ")[0] not in SECCIONES_DESCARTADAS
    ]


def chunking(texto: str, artista: dict) -> list[Document]:
    secciones = dividir_en_secciones(texto)

    # Short sections are merged into the previous one.
    # Not handled: if the FIRST section is short, it has no previous one and stays on its own.
    # It doesn't happen in the current 20 articles (the shortest introduction is 343 characters),
    # but it should be merged with the next one if more artists are downloaded.
    resultado = []
    for titulo, cuerpo in secciones:
        if len(cuerpo) < TAMANO_MINIMO and resultado:
            titulo_anterior, cuerpo_anterior = resultado[-1]
            resultado[-1] = (f"{titulo_anterior} / {titulo}", f"{cuerpo_anterior}\n{cuerpo}")
        else:
            resultado.append((titulo, cuerpo))

    # Long sections are split by paragraphs, accumulated up to TAMANO_MAXIMO.
    # LangChain's RecursiveCharacterTextSplitter already does this (and adds overlap),
    # but it is implemented by hand on purpose, to understand how the splitting works.
    resultado_limpio = []
    for titulo,cuerpo in resultado:
        if len(cuerpo) > TAMANO_MAXIMO:
            lineas = [linea for linea in cuerpo.split("\n") if linea.strip()]
            fragmento = ""
            for linea in lineas:
                if (len(linea) + len(fragmento) + 1) > TAMANO_MAXIMO and fragmento:
                    resultado_limpio.append((titulo,fragmento))
                    fragmento = linea
                else:
                    if fragmento: 
                        fragmento += "\n" + linea
                    else:
                        fragmento += linea
            if fragmento:
                resultado_limpio.append((titulo,fragmento))
        else:
            resultado_limpio.append((titulo,cuerpo))
        
    documentos = []
    for titulo,cuerpo in resultado_limpio:
        metadata = {
            "artist_id":artista["artist_id"],
            "name":artista["artista"],
            "url":artista["url"],
            "section":titulo
        }
        documentos.append(Document(page_content=f"{artista["artista"]} - {titulo}\n" + cuerpo,metadata=metadata))

    return documentos


if __name__ == "__main__":
    import json
    from pathlib import Path

    carpeta_wiki = Path(__file__).resolve().parent.parent / "data" / "wiki"
    indice = json.loads((carpeta_wiki / "indice.json").read_text(encoding="utf-8"))

    todos = []
    for artista in indice:
        texto = (carpeta_wiki / artista["archivo"]).read_text(encoding="utf-8")
        documentos = chunking(texto, artista)
        todos.extend(documentos)
        print(f"{len(documentos):>4} fragmentos | {artista['artista']}")

    print(f"\nTotal: {len(todos)} fragmentos de {len(indice)} artículos")
    print("\nEjemplo (primer fragmento):")
    print(todos[0].metadata)
    print(todos[0].page_content[:300])
