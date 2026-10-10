from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Literal,List
from langgraph.errors import GraphRecursionError
from langchain_core.tools import ToolException
import sqlite3
conexion = sqlite3.connect("02_tienda_musica/data/chinook.sqlite")

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
def describe_table(table_name:str)->List[str]:
    """
    Given the name of a table returns its columns

    Input:
        - table_name: Name of the table in the database
    """
    res = conexion.execute(f"PRAGMA table_info({table_name})").fetchall()
    if not res:
        raise ToolException(f"The table {table_name} doesn´t exist, use list_tables tool for fetching the table names")

    clean = []
    for elem in res:
        text = f"{elem[1]} ({elem[2]})"
        if elem[5] >= 1:
            text += " PRIMARY KEY"
        clean.append(text)
    return clean 



describe_table.handle_tool_error = True

print(describe_table.invoke({"table_name":"PlaylistTrack"}))