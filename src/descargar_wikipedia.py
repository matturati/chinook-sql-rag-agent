"""Downloads from (English) Wikipedia the articles of Chinook's best-selling artists.

Saves each article as plain text in data/wiki/ (section headings are kept as
"== Section ==") and an index with the artist, title and URL in data/wiki/indice.json.

Source: Wikipedia, CC BY-SA 4.0 license (the URL of each article must be credited).

Usage: uv run src/descargar_wikipedia.py [number of artists, default 20]
"""
import json
import re
import sqlite3
import sys
import time
from pathlib import Path
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen

DATA = Path(__file__).resolve().parent.parent / "data"
DB = DATA / "chinook.sqlite"
DESTINO = DATA / "wiki"
API = "https://en.wikipedia.org/w/api.php"
# Wikipedia requires a User-Agent that identifies the program
USER_AGENT = "agentes-tienda-musica/0.1 (proyecto de aprendizaje)"

# Chinook also sells TV episodes; these genres mean the "artist" is a TV series
GENEROS_TV = {"TV Shows", "Drama", "Comedy", "Science Fiction", "Sci Fi & Fantasy"}

# For each type: Wikipedia disambiguation suffixes (in order of preference) and words
# that must appear in the introduction for the article to be accepted
BUSQUEDA = {
    "musica": (("(band)", "(musician)", "(singer)", "(rapper)", ""),
               ("band", "musician", "singer", "rapper", "composer", "guitarist", "duo", "group")),
    "serie": (("(TV series)", "(American TV series)", ""),
              ("television series", "tv series", "sitcom", "drama series")),
}
# Cases the automatic search gets wrong: name in Chinook → exact Wikipedia title
TITULOS_MANUALES = {
    "Battlestar Galactica (Classic)": "Battlestar Galactica (1978 TV series)",
}
ESPERA = 1.0  # seconds between requests, to avoid overloading the API


def top_artistas(n):
    """Returns [(ArtistId, Name, genres)] for the n artists that have generated the most revenue."""
    conexion = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        artistas = conexion.execute(
            """SELECT a.ArtistId, a.Name
               FROM Artist a
               JOIN Album al ON a.ArtistId = al.ArtistId
               JOIN Track t ON al.AlbumId = t.AlbumId
               JOIN InvoiceLine i ON t.TrackId = i.TrackId
               GROUP BY a.ArtistId
               ORDER BY SUM(i.UnitPrice * i.Quantity) DESC
               LIMIT ?""",
            (n,),
        ).fetchall()
        return [
            (artist_id, nombre, {g for (g,) in conexion.execute(
                """SELECT DISTINCT g.Name FROM Album al
                   JOIN Track t ON al.AlbumId = t.AlbumId
                   JOIN Genre g ON t.GenreId = g.GenreId
                   WHERE al.ArtistId = ?""", (artist_id,))})
            for artist_id, nombre in artistas
        ]
    finally:
        conexion.close()


def llamar_api(params, reintentos=5):
    url = f"{API}?{urlencode({**params, 'format': 'json', 'formatversion': 2})}"
    for intento in range(reintentos):
        try:
            with urlopen(Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as respuesta:
                return json.load(respuesta)
        except HTTPError as e:
            if e.code != 429 or intento == reintentos - 1:
                raise
            # Too many requests: wait as long as Wikipedia says or, if it doesn't say, a bit longer each time
            espera = int(e.headers.get("Retry-After") or 2 ** (intento + 1))
            print(f"    (Wikipedia pide esperar; reintento en {espera} s)")
            time.sleep(espera)


def obtener_pagina(titulo):
    """Returns the article's (title, text), or None if it doesn't exist or is a disambiguation page."""
    datos = llamar_api({
        "action": "query", "prop": "extracts|pageprops", "explaintext": 1,
        "redirects": 1, "titles": titulo,
    })
    pagina = datos["query"]["pages"][0]
    if pagina.get("missing") or "disambiguation" in pagina.get("pageprops", {}):
        return None
    return pagina["title"], pagina.get("extract", "")


def buscar_articulo(nombre, generos):
    """Tries the name with each disambiguation suffix and keeps the first one that fits."""
    if nombre in TITULOS_MANUALES:
        return obtener_pagina(TITULOS_MANUALES[nombre])
    tipo = "serie" if generos and generos <= GENEROS_TV else "musica"
    sufijos, palabras_clave = BUSQUEDA[tipo]
    for sufijo in sufijos:
        resultado = obtener_pagina(f"{nombre} {sufijo}".strip())
        time.sleep(ESPERA)
        if resultado is None:
            continue
        titulo, texto = resultado
        # Only the first paragraph is checked: that's where Wikipedia says what the article is about
        introduccion = texto[:600].lower()
        if any(palabra in introduccion for palabra in palabras_clave):
            return titulo, texto

    # If no exact title fits (different capitalization, "(Classic)"...), fall back to the search API
    pista = "band" if tipo == "musica" else "television series"
    resultados = llamar_api({"action": "query", "list": "search", "srsearch": f"{nombre} {pista}", "srlimit": 3})
    time.sleep(ESPERA)
    for candidato in resultados["query"]["search"]:
        resultado = obtener_pagina(candidato["title"])
        time.sleep(ESPERA)
        if resultado and any(p in resultado[1][:600].lower() for p in palabras_clave):
            return resultado
    return None


def nombre_archivo(artist_id, nombre):
    return f"{artist_id}_{re.sub(r'[^A-Za-z0-9]+', '_', nombre).strip('_')}.txt"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    DESTINO.mkdir(parents=True, exist_ok=True)

    indice, fallos = [], []
    for artist_id, nombre, generos in top_artistas(n):
        encontrado = buscar_articulo(nombre, generos)
        if encontrado is None:
            fallos.append(nombre)
            print(f"  ✗ {nombre}: no encontrado")
            continue
        titulo, texto = encontrado
        archivo = nombre_archivo(artist_id, nombre)
        (DESTINO / archivo).write_text(texto, encoding="utf-8")
        url = f"https://en.wikipedia.org/wiki/{titulo.replace(' ', '_')}"
        indice.append({"artist_id": artist_id, "artista": nombre, "titulo": titulo,
                       "url": url, "archivo": archivo})
        print(f"  ✓ {nombre} → {titulo} ({len(texto):,} caracteres)")

    (DESTINO / "indice.json").write_text(json.dumps(indice, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n{len(indice)} artículos guardados en {DESTINO}")
    if fallos:
        print(f"No encontrados ({len(fallos)}): {', '.join(fallos)}")


if __name__ == "__main__":
    main()
