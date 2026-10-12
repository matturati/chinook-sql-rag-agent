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


def chunking(texto: str, artista: dict) -> list[Document]:
    secciones = dividir_en_secciones(texto)

    # Las secciones cortas se juntan con la anterior.
    # Caso no cubierto: si la PRIMERA sección es corta, no tiene anterior y queda sola.
    # En los 20 artículos actuales no pasa (la introducción más corta tiene 343 caracteres),
    # pero habría que juntarla con la siguiente si se descargan más artistas.
    resultado = []
    for titulo, cuerpo in secciones:
        if len(cuerpo) < TAMANO_MINIMO and resultado:
            titulo_anterior, cuerpo_anterior = resultado[-1]
            resultado[-1] = (f"{titulo_anterior} / {titulo}", f"{cuerpo_anterior}\n{cuerpo}")
        else:
            resultado.append((titulo, cuerpo))

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

    carpeta_wiki = Path(__file__).parent / "data" / "wiki"
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
