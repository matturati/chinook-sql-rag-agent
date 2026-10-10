from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Literal,List
from langgraph.errors import GraphRecursionError
from langchain_core.tools import ToolException
import sqlite3
import logging
logging.basicConfig(
    level=logging.WARNING
)

conexion = sqlite3.connect("file:02_tienda_musica/data/chinook.sqlite?mode=ro",uri=True)
logger = logging.getLogger(__name__)


def format_rows(cursor, rows) -> list:
    columns = []
    for data in cursor.description:
        columns.append(data[0])

    
    rows_clean = []
    for row in rows:
        rows_clean.append(dict(zip(columns,row)))

    return rows_clean

@tool
def list_tables()->List[str]:
    """
    Return the names of the tables found in the database
    """
    res = conexion.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    clean = []
    for elem in res:
        clean.append(elem[0])
    return clean

@tool
def describe_table(table_name:str)->dict:
    """
    Given the name of a table returns the schema of a table (CREATE TABLE statement with columns, types, primary and foreign keys) and 5 example rows

    Input:
        - table_name: Name of the table in the database
    """

    res = conexion.execute("SELECT sql FROM sqlite_master WHERE type = 'table' and name = ?",[table_name]).fetchall()


    if not res:
        raise ToolException(f"The table {table_name} doesn´t exist, use list_tables tool for fetching the table names")

    cursor = conexion.execute(f"SELECT * FROM [{table_name}] LIMIT 5")
    rows = cursor.fetchall()

    return {"Table info":res[0][0],"Example of rows":format_rows(cursor,rows)}

@tool
def execute_query(query:str) -> dict:
    """
    Execute a SQLite query and return the results
    The connection is read-only, don´t try to modify anything in it.
    Results are limited to 20 rows. Use COUNT, GROUP BY or LIMIT instead of retrieving full tables.

    Input:
        - query: Query to execute
    """
    try:
        cursor = conexion.execute(query)
    except sqlite3.OperationalError as e:
        if e.sqlite_errorname == "SQLITE_READONLY":
            logger.warning(f"The agent is trying to modify the database. Query executed: {query}")
        raise ToolException(f"The query has errors. Error: '{e}'")
    rows = cursor.fetchmany(21)

    shortened = "No"
    if len(rows) == 21:
        shortened = "Showing first 20 rows; there are more. Refine your query with WHERE, GROUP BY or LIMIT"
    
    return {"Resultant rows":format_rows(cursor,rows[:20]),"Result was limited":shortened}

describe_table.handle_tool_error = True
execute_query.handle_tool_error = True
if __name__ == "__main__":
    pruebas = [
        "SELECT Name FROM Genre LIMIT 3",   # correcta
        "SELECT * FROM Track",              # se recorta a 20 filas
        "SELECT Nombre FROM Artist",        # columna inexistente
        "DELETE FROM Artist",               # escritura en solo lectura
    ]
    for consulta in pruebas:
        print(f"\n>>> {consulta}")
        print(execute_query.invoke({"query": consulta}))