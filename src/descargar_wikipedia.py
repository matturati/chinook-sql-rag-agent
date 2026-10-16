"""Descarga de Wikipedia (en inglés) los artículos de los artistas de Chinook que más venden.

Guarda cada artículo como texto plano en data/wiki/ (los títulos de sección quedan como
"== Sección ==") y un índice con el artista, el título y la URL en data/wiki/indice.json.

Fuente: Wikipedia, licencia CC BY-SA 4.0 (hay que citar la URL de cada artículo).

Uso: uv run src/descargar_wikipedia.py [número de artistas, por defecto 20]
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
# Wikipedia exige un User-Agent que identifique el programa
USER_AGENT = "agentes-tienda-musica/0.1 (proyecto de aprendizaje)"

# Chinook también vende episodios de series; estos géneros indican que el "artista" es una serie
GENEROS_TV = {"TV Shows", "Drama", "Comedy", "Science Fiction", "Sci Fi & Fantasy"}

# Para cada tipo: sufijos de desambiguación de Wikipedia (en orden de preferencia) y palabras
# que deben aparecer en la introducción para aceptar el artículo
BUSQUEDA = {
    "musica": (("(band)", "(musician)", "(singer)", "(rapper)", ""),
               ("band", "musician", "singer", "rapper", "composer", "guitarist", "duo", "group")),
    "serie": (("(TV series)", "(American TV series)", ""),
              ("television series", "tv series", "sitcom", "drama series")),
}
# Casos que la búsqueda automática resuelve mal: nombre en Chinook → título exacto en Wikipedia
TITULOS_MANUALES = {
    "Battlestar Galactica (Classic)": "Battlestar Galactica (1978 TV series)",
}
ESPERA = 1.0  # segundos entre peticiones, para no saturar la API


def top_artistas(n):
    """Devuelve [(ArtistId, Name, géneros)] de los n artistas que más dinero han generado."""
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
            # Demasiadas peticiones: esperamos lo que diga Wikipedia o, si no lo dice, cada vez más
            espera = int(e.headers.get("Retry-After") or 2 ** (intento + 1))
            print(f"    (Wikipedia pide esperar; reintento en {espera} s)")
            time.sleep(espera)


def obtener_pagina(titulo):
    """Devuelve (título, texto) del artículo, o None si no existe o es de desambiguación."""
    datos = llamar_api({
        "action": "query", "prop": "extracts|pageprops", "explaintext": 1,
        "redirects": 1, "titles": titulo,
    })
    pagina = datos["query"]["pages"][0]
    if pagina.get("missing") or "disambiguation" in pagina.get("pageprops", {}):
        return None
    return pagina["title"], pagina.get("extract", "")


def buscar_articulo(nombre, generos):
    """Prueba el nombre con los sufijos de desambiguación y se queda con el primero que encaje."""
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
        # Solo miramos el primer párrafo: ahí Wikipedia dice qué es el tema del artículo
        introduccion = texto[:600].lower()
        if any(palabra in introduccion for palabra in palabras_clave):
            return titulo, texto

    # Si ningún título exacto encaja (mayúsculas distintas, "(Classic)"...), usamos el buscador
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
