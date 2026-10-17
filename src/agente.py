from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Literal,List
from langgraph.errors import GraphRecursionError
from langchain_core.tools import ToolException
from langgraph.prebuilt import ToolNode, tools_condition

from rag import buscar,obtener_coleccion, artistas_indexados

import sqlite3
import logging
from pathlib import Path
logging.basicConfig(
    level=logging.WARNING
)

logger = logging.getLogger(__name__)

DB = Path(__file__).resolve().parent.parent / "data" / "chinook.sqlite"

def connect_sql():
    return sqlite3.connect(f"file:{DB}?mode=ro",uri=True)

def format_rows(cursor, rows) -> list:
    columns = []
    for data in cursor.description:
        columns.append(data[0])

    
    rows_clean = []
    for row in rows:
        rows_clean.append(dict(zip(columns,row)))

    return rows_clean

@tool
def search_wikipedia(query:str, artist_name:str | None = None)-> list[dict]:
    """
    Search and obtain information about artists (bands, musicians and TV series) from Wikipedia. Use this tool when the user wants
    to know information about an artist, never answer without trying to get information.
    This tool should NEVER be used to search about sales, prices or the store's catalog (which albums or tracks it sells). Use the database instead.
    This tool should be used to gather information about the artist history, biography, style etc.
    This tool only contains the 20 artists with most sales in the shop. It doesn't access Wikipedia online, only previously downloaded articles.
    Whenever you use information, reference the url that is returned.

    Input:
        - query: Question about an artist that should be searched. Write a full question, not just a keyword. Write the query in English, since the articles are in English (this gives slightly better results).
        - artist_name: Pass it whenever the question is about a specific artist; omit it only for questions that are not about one artist.
    
    Output:
        - list of dictionaries with the information of the document and the url of the source.
    """

    return buscar(query,obtener_coleccion(),artist_name=artist_name)



@tool
def list_tables()->List[str]:
    """
    Return the names of the tables found in the database
    """
    
    conexion = connect_sql()    
    try:
        res = conexion.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        clean = []
        for elem in res:
            clean.append(elem[0])
    finally:
        conexion.close()
    return clean

@tool
def describe_tables(table_name_list:List[str])->dict:
    """
    Given a list of table names, for each one returns the schema of a table (CREATE TABLE statement with columns, types, primary and foreign keys) and 5 example rows. Ask all the tables you need at once.

    Input:
        - table_name_list: List of table names of the database
    """

    conexion = connect_sql()
    try:
        tables_info = {}
        found = 0
        for table_name in table_name_list:
            res = conexion.execute("SELECT sql FROM sqlite_master WHERE type = 'table' and name = ?",[table_name]).fetchall()


            if not res:
                tables_info[table_name] = {"Table info":"Not found in the database"}
            else:
                found += 1
                cursor = conexion.execute(f"SELECT * FROM [{table_name}] LIMIT 5")
                rows = cursor.fetchall()
                
                tables_info[table_name] = {"Table info":res[0][0],"Example of rows":format_rows(cursor,rows)}
        if found == 0:
            raise ToolException(f"None of the tables {table_name_list} exist. The valid table names are: {list_tables.invoke({})}")
    finally:
        conexion.close()
    return tables_info

@tool
def execute_query(query:str) -> dict:
    """
    Execute a SQLite query and return the results
    The connection is read-only, don´t try to modify anything in it.
    Results are limited to 20 rows. Use COUNT, GROUP BY or LIMIT instead of retrieving full tables.

    Input:
        - query: Query to execute
    """
    
    conexion = connect_sql()
    try:
        cursor = conexion.execute(query)

        rows = cursor.fetchmany(21)

        shortened = "No"
        if len(rows) == 21:
            shortened = "Showing first 20 rows; there are more. Refine your query with WHERE, GROUP BY or LIMIT"

    except sqlite3.OperationalError as e:
        if e.sqlite_errorname == "SQLITE_READONLY":
            logger.warning(f"The agent is trying to modify the database. Query executed: {query}")
        raise ToolException(f"The query has errors. Error: '{e}' Use the tool describe_table to get the correct name")
    finally:
        conexion.close()
    return {"Resultant rows":format_rows(cursor,rows[:20]),"Result was limited":shortened}

def model(state: MessagesState)-> dict:
    respuesta = llm.invoke(state["messages"])
    # qwen3 + Ollama sometimes puts the final answer inside the reasoning and leaves the content empty:
    # if there are no tool calls and no content, the reasoning is used as the answer
    if not respuesta.tool_calls and not respuesta.content.strip():
        razonamiento = respuesta.additional_kwargs.get("reasoning_content", "")
        if razonamiento:
            respuesta = respuesta.model_copy(update={"content": razonamiento})
    return {"messages": [respuesta]}

describe_tables.handle_tool_error = True
execute_query.handle_tool_error = True

if __name__ == "__main__":
    tools = [list_tables,describe_tables,execute_query,search_wikipedia]
    graph = StateGraph(MessagesState)

    llm = ChatOllama(model="qwen3:14b",num_ctx=16000,temperature=0,reasoning=True).bind_tools(tools)
    tool_node = ToolNode(tools=tools)
    system_message = SystemMessage(
            "Always use the tools if it is needed. First, gather information of the tables and their structure: call describe_tables() once with the list of all the tables you need, instead of one call per table. Then query when it is needed, make sure to use the correct names of tables and columns, use the primary key for making the aggregations (GROUP BY). You have a read-only connection, you can´t modify anything. If a query returns an error it will be sent to you, try to correct it, use the information tools to make sure you have not misspelled anything. Finally, give the user an explanation of the results, never only the rows." \
            " " \
            f"These are the tables of the database: {list_tables.invoke({})}. "\
            "Only use search_wikipedia when the user explicitly asks about an artist's (bands, musicians and TV series) history, biography, members, origin or style. Do not use it to add extra information the user did not ask for. " \
            "artist_name must be the name of a band, musician or TV series as it appears in the store, not a band member. If the tool says there are no documents, retry with the correct artist from the available list, or without artist_name. " \
            "Answer using only the information obtained from the tool and reference the source (url). If the tool doesn't return the information, say so instead of answering from your own knowledge. " \
            "If a question needs both sales data and information about the artist, use both the database and search_wikipedia. " \
            "If one tool call depends on the result of another (e.g. you need the artist name from the database before searching Wikipedia), make the calls one at a time and wait for each result. "\
            "Answer in the same language as the user's question."
        )

    # Example questions: SQL only, RAG only, and mixed (SQL + RAG in both directions)
    preguntas = [
        "top 5 de artistas por facturación",                                              # SQL
        "¿Quién era el batería de Led Zeppelin?",                                         # RAG
        "¿Cuál es el artista que más dinero ha generado en ventas y dónde se formó?",     # SQL → RAG
        "¿En qué banda cantaba Freddie Mercury y cuántas canciones suyas hay en la tienda?",  # RAG → SQL
        "¿Cuántas canciones de Aerosmith hay en la tienda y de dónde es el grupo?",       # artist without articles
    ]

    graph.add_node("model",model)
    graph.add_node("tools",tool_node)

    graph.add_edge(START,"model")
    graph.add_conditional_edges(
        "model",
        tools_condition,
        {"tools": "tools", "__end__": "__end__"},
    )
    graph.add_edge("tools","model")

    agent = graph.compile()
    obtener_coleccion()
    artistas_indexados()
    # Caching

    for pregunta in preguntas:
        print(f"\n\n========== {pregunta} ==========")
        messages = [system_message, HumanMessage(pregunta)]
        try:
            for chunk in agent.stream({"messages":messages},config={"recursion_limit":20},stream_mode="updates"):
                for node, update in chunk.items():
                    print(f"\n----- Nodo: {node} -----")
                    for msg in update["messages"]:
                        msg.pretty_print()
                        reasoning = msg.additional_kwargs.get("reasoning_content")
                        if reasoning:
                            print(f"\n[Reasoning]\n{reasoning}")
        except GraphRecursionError:
            print("The recursion limit was reached. Try again or simplify the prompt")