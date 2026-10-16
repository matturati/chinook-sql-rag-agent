"""Evaluación del agente SQL: ejecuta el agente con un conjunto de preguntas y mide cuántas acierta.

Métrica principal, *execution accuracy* (como en los benchmarks Spider y BIRD): se toma la última
consulta SQL que el agente ejecutó con éxito, se vuelve a ejecutar completa y se compara su
resultado con el de una consulta de referencia escrita a mano.
- Se permiten columnas de más en el resultado del agente (p. ej. el total además del nombre).
- Los números se comparan con tolerancia de 0.01 y los textos sin distinguir mayúsculas.
- Si la pregunta pide un orden (top N), el orden también cuenta.

Métrica secundaria: si la respuesta final en lenguaje natural menciona los valores esperados.

Como el modelo no es del todo determinista (incluso con temperature=0), la evaluación se puede
repetir varias veces y se informa de la media y del rango.

Uso:
    uv run src/evaluar_sql.py                    # qwen3:14b, 1 pasada
    uv run src/evaluar_sql.py qwen3:14b 3        # 3 pasadas
    uv run src/evaluar_sql.py gpt-oss:20b
"""
import json
import re
import sqlite3
import sys
import time
import unicodedata
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.errors import GraphRecursionError
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import agente  # las herramientas y el nodo del modelo del agente

DB = Path(__file__).resolve().parent.parent / "data" / "chinook.sqlite"
RESULTADOS = Path(__file__).resolve().parent.parent / "results" / "eval_sql.json"

# (pregunta, consulta de referencia, ¿importa el orden de las filas?)
# La referencia solo selecciona las columnas que responden a la pregunta.
PREGUNTAS = [
    ("¿Cuántos clientes hay registrados?",
     "SELECT COUNT(*) FROM Customer", False),
    ("¿Cuántas canciones hay en el catálogo?",
     "SELECT COUNT(*) FROM Track", False),
    ("¿Cuáles son los 5 artistas que más dinero han generado en ventas?",
     "SELECT a.Name FROM Artist a JOIN Album al ON a.ArtistId = al.ArtistId JOIN Track t ON al.AlbumId = t.AlbumId "
     "JOIN InvoiceLine i ON t.TrackId = i.TrackId GROUP BY a.ArtistId ORDER BY SUM(i.Quantity * i.UnitPrice) DESC LIMIT 5", True),
    ("¿Qué país ha generado más dinero en facturación?",
     "SELECT BillingCountry FROM Invoice GROUP BY BillingCountry ORDER BY SUM(Total) DESC LIMIT 1", False),
    ("¿Qué género musical tiene más canciones en el catálogo?",
     "SELECT g.Name FROM Track t JOIN Genre g ON t.GenreId = g.GenreId GROUP BY g.GenreId ORDER BY COUNT(*) DESC LIMIT 1", False),
    ("¿Cuántos álbumes de AC/DC hay en la tienda?",
     "SELECT COUNT(*) FROM Album al JOIN Artist a ON a.ArtistId = al.ArtistId WHERE a.Name = 'AC/DC'", False),
    ("¿Qué empleado de soporte ha generado más dinero en ventas a través de los clientes que tiene asignados?",
     "SELECT e.FirstName, e.LastName FROM Employee e JOIN Customer c ON c.SupportRepId = e.EmployeeId "
     "JOIN Invoice i ON i.CustomerId = c.CustomerId GROUP BY e.EmployeeId ORDER BY SUM(i.Total) DESC LIMIT 1", False),
    ("¿Cuántas facturas se emitieron en el año 2010?",
     "SELECT COUNT(*) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2010'", False),
    ("¿Cuál fue la facturación total del año 2011?",
     "SELECT SUM(Total) FROM Invoice WHERE strftime('%Y', InvoiceDate) = '2011'", False),
    ("¿Qué cliente ha gastado más dinero en total?",
     "SELECT c.FirstName, c.LastName FROM Customer c JOIN Invoice i ON i.CustomerId = c.CustomerId "
     "GROUP BY c.CustomerId ORDER BY SUM(i.Total) DESC LIMIT 1", False),
    ("¿Cuál es la canción más larga del catálogo?",
     "SELECT Name FROM Track ORDER BY Milliseconds DESC LIMIT 1", False),
    ("¿Cuántas unidades de canciones del género Rock se han vendido?",
     "SELECT SUM(il.Quantity) FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId "
     "JOIN Genre g ON g.GenreId = t.GenreId WHERE g.Name = 'Rock'", False),
    ("¿Cuáles son los 2 países con más clientes?",
     "SELECT Country FROM Customer GROUP BY Country ORDER BY COUNT(*) DESC LIMIT 2", True),
    ("¿Qué álbum ha generado más dinero en ventas?",
     "SELECT al.Title FROM Album al JOIN Track t ON t.AlbumId = al.AlbumId JOIN InvoiceLine il ON il.TrackId = t.TrackId "
     "GROUP BY al.AlbumId ORDER BY SUM(il.UnitPrice * il.Quantity) DESC LIMIT 1", False),
    ("¿Cuál es el precio medio de una canción del catálogo?",
     "SELECT AVG(UnitPrice) FROM Track", False),
    ("¿Cuántos empleados tiene la empresa?",
     "SELECT COUNT(*) FROM Employee", False),
    ("¿A quién reporta la empleada Jane Peacock?",
     "SELECT b.FirstName, b.LastName FROM Employee e JOIN Employee b ON e.ReportsTo = b.EmployeeId "
     "WHERE e.FirstName = 'Jane' AND e.LastName = 'Peacock'", False),
    ("¿Qué tipo de formato (media type) se ha vendido más en unidades?",
     "SELECT m.Name FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId JOIN MediaType m ON m.MediaTypeId = t.MediaTypeId "
     "GROUP BY m.MediaTypeId ORDER BY SUM(il.Quantity) DESC LIMIT 1", False),
    ("¿Cuántos artistas no tienen ningún álbum en la tienda?",
     "SELECT COUNT(*) FROM Artist a LEFT JOIN Album al ON al.ArtistId = a.ArtistId WHERE al.AlbumId IS NULL", False),
    ("¿Cuántas canciones tiene el álbum 'Let There Be Rock'?",
     "SELECT COUNT(*) FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId WHERE al.Title = 'Let There Be Rock'", False),
    ("¿Cuántas canciones de Iron Maiden hay en el catálogo?",
     "SELECT COUNT(*) FROM Track t JOIN Album al ON al.AlbumId = t.AlbumId JOIN Artist a ON a.ArtistId = al.ArtistId "
     "WHERE a.Name = 'Iron Maiden'", False),
    ("¿Qué país tiene el mayor importe medio por factura?",
     "SELECT BillingCountry FROM Invoice GROUP BY BillingCountry ORDER BY AVG(Total) DESC LIMIT 1", False),
    ("¿Qué compositor aparece en más canciones del catálogo?",
     "SELECT Composer FROM Track WHERE Composer IS NOT NULL GROUP BY Composer ORDER BY COUNT(*) DESC LIMIT 1", False),
    ("¿Qué año tuvo la mayor facturación?",
     "SELECT strftime('%Y', InvoiceDate) y FROM Invoice GROUP BY y ORDER BY SUM(Total) DESC LIMIT 1", False),
    ("¿Cuánto dinero han generado en ventas las canciones de Metallica?",
     "SELECT SUM(il.UnitPrice * il.Quantity) FROM InvoiceLine il JOIN Track t ON t.TrackId = il.TrackId "
     "JOIN Album al ON al.AlbumId = t.AlbumId JOIN Artist a ON a.ArtistId = al.ArtistId WHERE a.Name = 'Metallica'", False),
    ("¿Cuántas canciones del catálogo no se han vendido nunca?",
     "SELECT COUNT(*) FROM Track t WHERE NOT EXISTS (SELECT 1 FROM InvoiceLine il WHERE il.TrackId = t.TrackId)", False),
]

# El mismo system prompt que usa agente.py
SYSTEM_PROMPT = (
    "Always use the tools if it is needed. First, gather information of the tables and their structure: call "
    "describe_tables() once with the list of all the tables you need, instead of one call per table. Then query when "
    "it is needed, make sure to use the correct names of tables and columns, use the primary key for making the "
    "aggregations (GROUP BY). You have a read-only connection, you can´t modify anything. If a query returns an error "
    "it will be sent to you, try to correct it, use the information tools to make sure you have not misspelled "
    "anything. Finally, give the user an explanation of the results, never only the rows."
    f" These are the tables of the database: {agente.list_tables.invoke({})}"
)


# ---------- Comparación de resultados ----------

def normalizar_texto(valor) -> str:
    texto = unicodedata.normalize("NFKD", str(valor)).encode("ascii", "ignore").decode()
    return texto.strip().lower()


def como_numero(valor):
    """Devuelve el valor como float si lo es (o es un texto numérico, como el año '2010'); si no, None."""
    if isinstance(valor, bool) or valor is None:
        return None
    try:
        return float(valor)
    except (TypeError, ValueError):
        return None


def celda_en_fila(celda_ref, fila_agente) -> bool:
    """¿El valor de referencia aparece en alguna columna de la fila del agente?"""
    numero = como_numero(celda_ref)
    if numero is not None:
        return any(n is not None and abs(n - numero) <= 0.01 for n in map(como_numero, fila_agente))
    # Texto: se compara contra cada celda y contra la fila entera (por si el agente concatena nombre y apellido)
    texto = normalizar_texto(celda_ref)
    fila_entera = " ".join(normalizar_texto(c) for c in fila_agente)
    return texto in fila_entera


def fila_coincide(fila_ref, fila_agente) -> bool:
    return all(celda_en_fila(celda, fila_agente) for celda in fila_ref)


def resultados_coinciden(ref, agente_filas, importa_orden) -> bool:
    if len(ref) != len(agente_filas):
        return False
    if importa_orden:
        return all(fila_coincide(r, a) for r, a in zip(ref, agente_filas))
    # Sin orden: cada fila de referencia debe emparejarse con una fila distinta del agente
    disponibles = list(agente_filas)
    for fila_ref in ref:
        for i, fila_agente in enumerate(disponibles):
            if fila_coincide(fila_ref, fila_agente):
                del disponibles[i]
                break
        else:
            return False
    return True


def numeros_en_texto(texto: str) -> list[float]:
    """Extrae números aceptando separadores de miles y decimales en formato inglés o español."""
    numeros = []
    for bruto in re.findall(r"\d[\d.,  ]*\d|\d", texto):
        bruto = bruto.replace(" ", "").replace(" ", "")
        candidatos = {bruto.replace(",", ""), bruto.replace(".", "").replace(",", ".")}
        for c in candidatos:
            try:
                numeros.append(float(c))
            except ValueError:
                pass
    return numeros


def respuesta_menciona(ref, respuesta: str) -> bool:
    texto = normalizar_texto(respuesta)
    numeros = numeros_en_texto(respuesta)
    for fila in ref:
        for celda in fila:
            numero = como_numero(celda)
            if numero is not None:
                if not any(abs(n - numero) <= 0.01 for n in numeros):
                    return False
            elif normalizar_texto(celda) not in texto:
                return False
    return True


def ejecutar(sql: str):
    conexion = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return conexion.execute(sql).fetchall()
    finally:
        conexion.close()


# ---------- Agente ----------

def construir_agente(modelo: str):
    herramientas = [agente.list_tables, agente.describe_tables, agente.execute_query]
    # El nodo "model" de agente.py usa la variable global "llm" del módulo
    agente.llm = ChatOllama(model=modelo, num_ctx=16000, temperature=0, reasoning=True).bind_tools(herramientas)
    grafo = StateGraph(MessagesState)
    grafo.add_node("model", agente.model)
    grafo.add_node("tools", ToolNode(tools=herramientas))
    grafo.add_edge(START, "model")
    grafo.add_conditional_edges("model", tools_condition)
    grafo.add_edge("tools", "model")
    return grafo.compile()


def analizar_mensajes(mensajes):
    """Devuelve (última SQL con éxito, nº llamadas a herramientas, nº errores, respuesta final,
    si la respuesta estaba en el razonamiento en vez de en el contenido)."""
    estados = {m.tool_call_id: m.status for m in mensajes if isinstance(m, ToolMessage)}
    ultima_sql, llamadas, errores = None, 0, 0
    for m in mensajes:
        if isinstance(m, AIMessage):
            for llamada in m.tool_calls:
                llamadas += 1
                if estados.get(llamada["id"]) == "error":
                    errores += 1
                elif llamada["name"] == "execute_query":
                    ultima_sql = llamada["args"].get("query")
    respuesta, en_razonamiento = "", False
    if isinstance(mensajes[-1], AIMessage):
        respuesta = mensajes[-1].content
        # qwen3 + Ollama a veces deja la respuesta final dentro del razonamiento y el contenido vacío
        if not respuesta.strip():
            respuesta = mensajes[-1].additional_kwargs.get("reasoning_content", "")
            en_razonamiento = bool(respuesta)
    return ultima_sql, llamadas, errores, respuesta, en_razonamiento


def una_pasada(agente_compilado, modelo: str) -> dict:
    resultados = []
    inicio_total = time.time()

    for n, (pregunta, sql_ref, importa_orden) in enumerate(PREGUNTAS, 1):
        ref = ejecutar(sql_ref)
        inicio = time.time()
        registro = {"pregunta": pregunta, "sql_referencia": sql_ref, "resultado_referencia": ref}
        try:
            estado = agente_compilado.invoke(
                {"messages": [SystemMessage(SYSTEM_PROMPT), HumanMessage(pregunta)]},
                config={"recursion_limit": 20},
            )
            sql_agente, llamadas, errores, respuesta, en_razonamiento = analizar_mensajes(estado["messages"])
            filas_agente = ejecutar(sql_agente) if sql_agente else []
            registro.update({
                "sql_agente": sql_agente,
                "acierto_sql": bool(sql_agente) and resultados_coinciden(ref, filas_agente, importa_orden),
                "respuesta_correcta": respuesta_menciona(ref, respuesta),
                "llamadas_herramientas": llamadas,
                "errores_herramientas": errores,
                "respuesta": respuesta,
                "respuesta_en_razonamiento": en_razonamiento,
            })
        except GraphRecursionError:
            registro.update({"acierto_sql": False, "respuesta_correcta": False, "fallo": "límite de pasos"})
        except Exception as e:  # cualquier otro fallo cuenta como pregunta fallada, pero se registra
            registro.update({"acierto_sql": False, "respuesta_correcta": False, "fallo": repr(e)})
        registro["segundos"] = round(time.time() - inicio, 1)
        resultados.append(registro)

        marca = "✓" if registro["acierto_sql"] else "✗"
        print(f"{marca} [{n:>2}/{len(PREGUNTAS)}] {registro['segundos']:>6.1f}s  {pregunta}", flush=True)

    total = len(resultados)
    completadas = [r for r in resultados if "llamadas_herramientas" in r]
    resumen = {
        "modelo": modelo,
        "preguntas": total,
        "execution_accuracy": round(sum(r["acierto_sql"] for r in resultados) / total, 3),
        "respuesta_final_correcta": round(sum(r["respuesta_correcta"] for r in resultados) / total, 3),
        "respuestas_vacias_en_contenido": sum(r.get("respuesta_en_razonamiento", False) for r in resultados),
        "media_llamadas_herramientas": round(sum(r["llamadas_herramientas"] for r in completadas) / max(len(completadas), 1), 2),
        "preguntas_con_errores_sql_recuperados": sum(1 for r in completadas if r["errores_herramientas"] and r["acierto_sql"]),
        "segundos_media_por_pregunta": round(sum(r["segundos"] for r in resultados) / total, 1),
        "minutos_totales": round((time.time() - inicio_total) / 60, 1),
    }
    print("\n" + "  ".join(f"{k}={v}" for k, v in resumen.items()), flush=True)
    return {"resumen": resumen, "detalle": resultados}


def evaluar(modelo: str, pasadas: int):
    agente_compilado = construir_agente(modelo)
    resultados = []
    for i in range(1, pasadas + 1):
        print(f"\n===== Pasada {i}/{pasadas} ({modelo}) =====", flush=True)
        resultados.append(una_pasada(agente_compilado, modelo))

    def agregado(clave):
        valores = [r["resumen"][clave] for r in resultados]
        return {"media": round(sum(valores) / len(valores), 3), "min": min(valores), "max": max(valores)}

    global_ = {
        "modelo": modelo,
        "pasadas": pasadas,
        "preguntas_por_pasada": len(PREGUNTAS),
        "execution_accuracy": agregado("execution_accuracy"),
        "respuesta_final_correcta": agregado("respuesta_final_correcta"),
        "segundos_media_por_pregunta": agregado("segundos_media_por_pregunta"),
        # Preguntas que fallaron en alguna pasada, para revisarlas a mano
        "preguntas_falladas": sorted({d["pregunta"] for r in resultados for d in r["detalle"] if not d["acierto_sql"]}),
    }
    RESULTADOS.write_text(json.dumps({"global": global_, "pasadas": resultados}, ensure_ascii=False, indent=2, default=str),
                          encoding="utf-8")

    print("\n===== Resultado global =====")
    for clave, valor in global_.items():
        print(f"{clave}: {valor}")
    print(f"\nDetalle guardado en {RESULTADOS}")


if __name__ == "__main__":
    evaluar(sys.argv[1] if len(sys.argv) > 1 else "qwen3:14b", int(sys.argv[2]) if len(sys.argv) > 2 else 1)
