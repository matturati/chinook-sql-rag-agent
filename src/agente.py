from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Literal,List
from langgraph.errors import GraphRecursionError
from langchain_core.tools import ToolException
from langgraph.prebuilt import ToolNode, tools_condition

from rag import buscar,obtener_coleccion, artistas_indexados

import json
import re
import sqlite3
import uuid
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
    Whenever you use information from a chunk, cite its id in square brackets right after the sentence, e.g. [90_2].

    Input:
        - query: Question about an artist that should be searched. Write a full question, not just a keyword. Write the query in English, since the articles are in English (this gives slightly better results).
        - artist_name: Pass it whenever the question is about a specific artist; omit it only for questions that are not about one artist.
    
    Output:
        - list of dictionaries with the chunk id, the text of the chunk and the url of the source.
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

def citar_fuentes(texto: str, mensajes: list) -> str:
    """Replaces the chunk ids cited by the model ([90_2]) with numbers ([1]) and adds the source URLs at the end.

    The model decides which chunks it used; the code writes the URLs and drops ids it never received.
    """
    urls = {}  # chunk id → URL, from the search_wikipedia results of this conversation
    for m in mensajes:
        if isinstance(m, ToolMessage) and m.name == "search_wikipedia":
            try:
                chunks = json.loads(m.content)
            except (json.JSONDecodeError, TypeError):
                continue
            urls.update({c["id"]: c["url"] for c in chunks if c.get("id")})

    fuentes = []  # unique URLs, in order of first citation
    def reemplazar(cita):
        numeros = []
        for id_ in re.split(r"\s*,\s*", cita.group(1)):
            url = urls.get(id_)
            if url is None:  # an id the model never received
                continue
            if url not in fuentes:
                fuentes.append(url)
            numeros.append(fuentes.index(url) + 1)
        return "".join(f"[{n}]" for n in dict.fromkeys(numeros))

    texto = re.sub(r"\[(\d+_\d+(?:\s*,\s*\d+_\d+)*)\]", reemplazar, texto)
    if fuentes:
        texto += "\n\nSources:\n" + "\n".join(f"[{n}] {url}" for n, url in enumerate(fuentes, 1))
    return texto


def recuperar_tool_calls(respuesta: AIMessage, texto: str) -> AIMessage:
    """qwen3 + Ollama sometimes writes the tool call as text (<tool_call>{...}</tool_call>) instead of a real one.

    Ollama didn't recognise it, so it would be taken as the final answer and the agent would stop:
    parse it here and turn it into a real tool call. Invalid JSON or unknown tools are left as they are.
    `texto` is the text that would become the answer (the content, or the reasoning if the content is empty).
    """
    bloques = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", texto, flags=re.DOTALL)
    nombres = {t.name for t in TOOLS}
    tool_calls = []
    for bloque in bloques:
        try:
            llamada = json.loads(bloque)
        except json.JSONDecodeError:
            continue
        if llamada.get("name") in nombres:
            tool_calls.append({"name": llamada["name"], "args": llamada.get("arguments", {}),
                               "id": f"call_from_text_{uuid.uuid4().hex[:8]}", "type": "tool_call"})
    if not tool_calls:
        return respuesta
    logger.warning(f"Recovered {len(tool_calls)} tool call(s) written as text")
    texto = re.sub(r"<tool_call>.*?</tool_call>", "", respuesta.content, flags=re.DOTALL).strip()
    return AIMessage(content=texto, tool_calls=tool_calls, additional_kwargs=respuesta.additional_kwargs, id=respuesta.id)


def model(state: MessagesState)-> dict:
    respuesta = llm.invoke(state["messages"])
    if respuesta.tool_calls:
        return {"messages": [respuesta]}

    contenido = respuesta.content
    # qwen3 + Ollama sometimes puts the final answer inside the reasoning and leaves the content empty:
    # if there are no tool calls and no content, the reasoning is used as the answer
    if not contenido.strip():
        contenido = respuesta.additional_kwargs.get("reasoning_content", "")

    # The text that would become the answer may be a tool call written as text (in the content or in the reasoning)
    respuesta = recuperar_tool_calls(respuesta, contenido)
    if not respuesta.tool_calls:
        respuesta = respuesta.model_copy(update={"content": citar_fuentes(contenido, state["messages"])})
    return {"messages": [respuesta]}

describe_tables.handle_tool_error = True
execute_query.handle_tool_error = True

TOOLS = [list_tables,describe_tables,execute_query,search_wikipedia]

SYSTEM_PROMPT = (
            "Always use the tools if it is needed. First, gather information of the tables and their structure: call describe_tables() once with the list of all the tables you need, instead of one call per table. Then query when it is needed, make sure to use the correct names of tables and columns, use the primary key for making the aggregations (GROUP BY). You have a read-only connection, you can´t modify anything. If a query returns an error it will be sent to you, try to correct it, use the information tools to make sure you have not misspelled anything. Finally, give the user an explanation of the results, never only the rows." \
            " " \
            f"These are the tables of the database: {list_tables.invoke({})}. "\
            "The store sells music and also TV series episodes (the series appear in the Artist table). " \
            "The database only contains sales and catalog data: it has no information about where an artist was formed, its members, its history, who created a TV series or what it is about. " \
            "Any question (or part of a question) about those topics REQUIRES calling search_wikipedia, also to find out which artist a band member or a description refers to. " \
            "Do not use search_wikipedia to add extra information the user did not ask for. " \
            "Never call a tool with placeholder values such as '[Artist Name]': wait for the result that gives you the real value. " \
            "artist_name must be the name of a band, musician or TV series as it appears in the store, not a band member. If the tool says there are no documents, retry with the correct artist from the available list, or without artist_name. " \
            "Answer using only the information obtained from the tool. Cite the chunks you use by their id in square brackets right after the sentence, e.g. [90_2]; don't write the URLs, they are added automatically. If the tool doesn't return the information, say so instead of answering from your own knowledge. " \
            "If a question needs both sales data and information about the artist, use both the database and search_wikipedia. " \
            "If one tool call depends on the result of another (e.g. you need the artist name from the database before searching Wikipedia), make the calls one at a time and wait for each result. "\
            "Answer in the same language as the user's question."
        )


def construir_agente(modelo: str = "qwen3:14b"):
    """Builds and compiles the agent graph with the given Ollama model."""
    global llm  # the "model" node uses the module-level llm
    llm = ChatOllama(model=modelo,num_ctx=16000,temperature=0,reasoning=True).bind_tools(TOOLS)

    graph = StateGraph(MessagesState)
    graph.add_node("model",model)
    graph.add_node("tools",ToolNode(tools=TOOLS))

    graph.add_edge(START,"model")
    graph.add_conditional_edges(
        "model",
        tools_condition,
        {"tools": "tools", "__end__": "__end__"},
    )
    graph.add_edge("tools","model")

    # Create the Chroma client once, before ToolNode runs tools in parallel threads
    obtener_coleccion()
    artistas_indexados()

    return graph.compile()


if __name__ == "__main__":
    agent = construir_agente()
    system_message = SystemMessage(SYSTEM_PROMPT)

    # Example questions: SQL only, RAG only, and mixed (SQL + RAG in both directions)
    preguntas = [
        "top 5 de artistas por facturación",                                              # SQL
        "¿Quién era el batería de Led Zeppelin?",                                         # RAG
        "¿Cuál es el artista que más dinero ha generado en ventas y dónde se formó?",     # SQL → RAG
        "¿En qué banda cantaba Freddie Mercury y cuántas canciones suyas hay en la tienda?",  # RAG → SQL
        "¿Cuántas canciones de Aerosmith hay en la tienda y de dónde es el grupo?",       # artist without articles
    ]

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