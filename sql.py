import sqlite3
conexion = sqlite3.connect("02_tienda_musica/data/chinook.sqlite")

filas = conexion.execute("""
    SELECT a.Name, round(sum(i.Quantity*i.UnitPrice),2)
    FROM Artist a
    JOIN Album al ON a.ArtistId = al.ArtistId
    JOIN Track t ON al.AlbumId = t.AlbumId
    JOIN InvoiceLine i ON t.TrackId = i.TrackId
    GROUP BY a.ArtistId
    ORDER BY sum(i.Quantity*i.UnitPrice) desc LIMIT 5
""").fetchall()
for fila in filas:
    print(fila)