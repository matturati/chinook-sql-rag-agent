"""Descarga la base de datos Chinook (SQLite) en data/.

Fuente: https://github.com/lerocha/chinook-database (Luis Rocha, licencia MIT).
"""
from pathlib import Path
from urllib.request import urlretrieve

URL = "https://github.com/lerocha/chinook-database/releases/latest/download/Chinook_Sqlite.sqlite"
DESTINO = Path(__file__).resolve().parent.parent / "data" / "chinook.sqlite"


def main():
    if DESTINO.exists():
        print(f"Ya existe: {DESTINO}")
        return
    DESTINO.parent.mkdir(parents=True, exist_ok=True)
    print(f"Descargando Chinook en {DESTINO}...")
    urlretrieve(URL, DESTINO)
    print(f"Listo ({DESTINO.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
