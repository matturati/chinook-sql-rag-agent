"""End-to-end evaluation of the agent (SQL + RAG).

Each question has:
- tipo: "rag", "sql", "sql_rag" (SQL first, then Wikipedia), "rag_sql" (Wikipedia first, then SQL)
  or "no_indexado" (an artist in the store without Wikipedia articles).
- datos_clave: values the final answer must contain (text is compared without accents or case,
  numbers with a 0.01 tolerance). Empty when the answer is open and only the judge can grade it.
- prohibidos: values the answer must NOT contain (facts the model knows but the tools don't return).
- usa / no_usa: tools that must / must not be called.
- referencia: reference answer for the LLM judge.

Every reference was checked against the database and the downloaded Wikipedia articles,
not against general knowledge: if the articles don't say it, the agent can't be expected to.

This script only runs the agent and the automatic checks, and saves everything to
results/eval_agente.json. The LLM judge runs afterwards on the saved answers.

Usage:
    uv run src/evaluar_agente_global.py              # qwen3:14b, 1 run
    uv run src/evaluar_agente_global.py qwen3:14b 3  # 3 runs
"""
import json
import re
import sys
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.errors import GraphRecursionError

import agente
from evaluar_sql import normalizar_texto, numeros_en_texto, respuesta_menciona

RESULTADOS = Path(__file__).resolve().parent.parent / "results" / "eval_agente.json"


PREGUNTAS = [
    # ---------- RAG only ----------
    {"pregunta": "¿Quién era el batería de Led Zeppelin?",
     "tipo": "rag", "datos_clave": ["Bonham"], "prohibidos": [],
     "usa": ["search_wikipedia"], "no_usa": ["execute_query"],
     "referencia": "John Bonham was Led Zeppelin's drummer, from 1968 until his death in 1980."},
    {"pregunta": "¿En qué ciudad y en qué año se formó U2?",
     "tipo": "rag", "datos_clave": ["Dublin", 1976], "prohibidos": [],
     "usa": ["search_wikipedia"], "no_usa": ["execute_query"],
     "referencia": "U2 was formed in Dublin in 1976."},
    {"pregunta": "¿Quiénes fundaron Metallica?",
     "tipo": "rag", "datos_clave": ["Hetfield", "Ulrich"], "prohibidos": [],
     "usa": ["search_wikipedia"], "no_usa": ["execute_query"],
     "referencia": "Metallica was formed in Los Angeles in 1981 by James Hetfield and Lars Ulrich."},
    {"pregunta": "¿De qué trata la serie Lost?",
     "tipo": "rag", "datos_clave": [], "prohibidos": [],
     "usa": ["search_wikipedia"], "no_usa": ["execute_query"],
     "referencia": "Lost follows the survivors of Oceanic Airlines flight 815 after the plane crashes on a "
                   "mysterious tropical island; it has elements of supernatural fiction."},
    {"pregunta": "¿Por qué se separó Creedence Clearwater Revival?",
     "tipo": "rag", "datos_clave": [1972], "prohibidos": [],
     "usa": ["search_wikipedia"], "no_usa": ["execute_query"],
     "referencia": "CCR disbanded acrimoniously in 1972: Tom Fogerty had quit at the end of 1970 and John "
                   "Fogerty was at odds with the other members over business and artistic control, which "
                   "led to lawsuits."},
    {"pregunta": "¿En qué empresa trabajan los personajes de The Office?",
     "tipo": "rag", "datos_clave": ["Dunder Mifflin"], "prohibidos": [],
     "usa": ["search_wikipedia"], "no_usa": ["execute_query"],
     "referencia": "They work at the Scranton, Pennsylvania branch of the fictional Dunder Mifflin Paper Company."},

    # ---------- SQL first, then Wikipedia ----------
    {"pregunta": "¿Cuál es el artista que más dinero ha generado en ventas y dónde se formó?",
     "tipo": "sql_rag", "datos_clave": ["Iron Maiden", 138.6, "Leyton"], "prohibidos": [],
     "usa": ["execute_query", "search_wikipedia"], "no_usa": [],
     "referencia": "Iron Maiden, with $138.60 in sales. The band was formed in Leyton, East London, in 1975."},
    {"pregunta": "¿Cuál es el segundo artista que más dinero ha generado en ventas y en qué ciudad se formó?",
     "tipo": "sql_rag", "datos_clave": ["U2", "Dublin"], "prohibidos": [],
     "usa": ["execute_query", "search_wikipedia"], "no_usa": [],
     "referencia": "U2, with $105.93 in sales. The band was formed in Dublin in 1976."},
    {"pregunta": "¿De qué artista es el álbum que más dinero ha generado en ventas, y quién lo creó?",
     "tipo": "sql_rag", "datos_clave": ["Battlestar Galactica", "Larson"], "prohibidos": [],
     "usa": ["execute_query", "search_wikipedia"], "no_usa": [],
     "referencia": "The album is 'Battlestar Galactica (Classic), Season 1' ($35.82), from the TV series "
                   "Battlestar Galactica (Classic), created by Glen A. Larson."},

    # ---------- Wikipedia first, then SQL ----------
    {"pregunta": "¿En qué banda cantaba Freddie Mercury y cuántas canciones de esa banda hay en la tienda?",
     "tipo": "rag_sql", "datos_clave": ["Queen", 45], "prohibidos": [],
     "usa": ["search_wikipedia", "execute_query"], "no_usa": [],
     "referencia": "Freddie Mercury was the lead singer of Queen. The store has 45 Queen tracks."},
    {"pregunta": "¿En qué banda toca Lars Ulrich y cuántos álbumes suyos hay en la tienda?",
     "tipo": "rag_sql", "datos_clave": ["Metallica", 10], "prohibidos": [],
     "usa": ["search_wikipedia", "execute_query"], "no_usa": [],
     "referencia": "Lars Ulrich is Metallica's drummer and co-founder. The store has 10 Metallica albums."},
    {"pregunta": "¿Cuánto dinero ha generado en la tienda la serie sobre los supervivientes del vuelo 815 de Oceanic?",
     "tipo": "rag_sql", "datos_clave": ["Lost", 81.59], "prohibidos": [],
     "usa": ["search_wikipedia", "execute_query"], "no_usa": [],
     "referencia": "The series is Lost, which has generated $81.59 in sales."},

    # ---------- Artists in the store without Wikipedia articles ----------
    # The model knows the answers (Boston, Sydney), but the tools don't return them
    {"pregunta": "¿Cuántas canciones de Aerosmith hay en la tienda y de dónde es el grupo?",
     "tipo": "no_indexado", "datos_clave": [15], "prohibidos": ["Boston"],
     "usa": ["execute_query"], "no_usa": [],
     "referencia": "The store has 15 Aerosmith tracks. The documents have no information about the band's origin."},
    {"pregunta": "¿Dónde se formó AC/DC y cuántos álbumes suyos hay en la tienda?",
     "tipo": "no_indexado", "datos_clave": [2], "prohibidos": ["Sydney", "Sidney", "Australia"],
     "usa": ["execute_query"], "no_usa": [],
     "referencia": "The store has 2 AC/DC albums. The documents have no information about where the band was formed."},

    # ---------- SQL only: Wikipedia must not be used ----------
    {"pregunta": "¿Cuáles son los 5 artistas que más dinero han generado en ventas?",
     "tipo": "sql", "datos_clave": ["Iron Maiden", "U2", "Metallica", "Led Zeppelin", "Lost"], "prohibidos": [],
     "usa": ["execute_query"], "no_usa": ["search_wikipedia"],
     "referencia": "Iron Maiden ($138.60), U2 ($105.93), Metallica ($90.09), Led Zeppelin ($86.13) and Lost ($81.59)."},
    {"pregunta": "¿Cuántas canciones de Queen hay en la tienda?",
     "tipo": "sql", "datos_clave": [45], "prohibidos": [],
     "usa": ["execute_query"], "no_usa": ["search_wikipedia"],
     "referencia": "The store has 45 Queen tracks."},
    {"pregunta": "¿Qué género musical tiene más canciones en el catálogo?",
     "tipo": "sql", "datos_clave": ["Rock"], "prohibidos": [],
     "usa": ["execute_query"], "no_usa": ["search_wikipedia"],
     "referencia": "Rock, with 1,297 tracks."},
]


def ejecutar_agente(agent, pregunta: str) -> dict:
    """Runs the agent on one question and returns the answer, the tool calls and the tool results."""
    inicio = time.time()
    try:
        estado = agent.invoke(
            {"messages": [SystemMessage(agente.SYSTEM_PROMPT), HumanMessage(pregunta)]},
            config={"recursion_limit": 20},
        )
        mensajes = estado["messages"]
        fallo = None
    except GraphRecursionError:
        mensajes, fallo = [], "recursion limit"

    llamadas = [{"name": l["name"], "args": l["args"]}
                for m in mensajes if isinstance(m, AIMessage) for l in m.tool_calls]
    resultados = [str(m.content) for m in mensajes if isinstance(m, ToolMessage)]
    respuesta = mensajes[-1].content if mensajes and isinstance(mensajes[-1], AIMessage) else ""

    return {"respuesta": respuesta, "llamadas": llamadas, "resultados": resultados,
            "segundos": round(time.time() - inicio, 1), "fallo": fallo}


def numeros_respaldados(respuesta: str, fuentes: str) -> bool:
    """Is every number in the answer present in the tool results or in the question?"""
    respuesta = re.sub(r"^\s*\d+[.)]\s", "", respuesta, flags=re.MULTILINE)  # list numbering "1. ", "2) "
    respuesta = re.sub(r"https?://\S+", "", respuesta)                          # URLs
    respuesta = re.sub(r"\[\d+\]", "", respuesta)                               # citations "[1]"
    # numeros_en_texto joins "October 28, 1981" into a single number, so plain integers are added too
    numeros_fuentes = numeros_en_texto(fuentes) + [float(x) for x in re.findall(r"\d+", fuentes)]
    for token in re.findall(r"\d[\d.,  ]*\d|\d", respuesta):
        # A token like "1.297" can be read in several ways; it's backed if any of them is
        if not any(abs(n - f) <= 0.01 for n in numeros_en_texto(token) for f in numeros_fuentes):
            return False
    return True


def comprobar(p: dict, r: dict) -> dict:
    """Automatic checks of one answer. None means the check doesn't apply to this question."""
    respuesta = r["respuesta"]
    usadas = {l["name"] for l in r["llamadas"]}
    return {
        "datos_clave": respuesta_menciona([p["datos_clave"]], respuesta),
        "sin_prohibidos": not any(normalizar_texto(x) in normalizar_texto(respuesta) for x in p["prohibidos"]),
        "herramientas": all(h in usadas for h in p["usa"]) and not any(h in usadas for h in p["no_usa"]),
        "cita_url": "wikipedia.org" in respuesta if p["tipo"] in ("rag", "sql_rag", "rag_sql") else None,
        "numeros_respaldados": numeros_respaldados(respuesta, "\n".join(r["resultados"]) + "\n" + p["pregunta"]),
    }


def evaluar(modelo: str, pasadas: int):
    agent = agente.construir_agente(modelo)
    detalle = []
    for pasada in range(1, pasadas + 1):
        print(f"\n===== Run {pasada}/{pasadas} ({modelo}) =====", flush=True)
        for n, p in enumerate(PREGUNTAS, 1):
            r = ejecutar_agente(agent, p["pregunta"])
            comprobaciones = comprobar(p, r)
            ok = all(v for v in comprobaciones.values() if v is not None) and not r["fallo"]
            detalle.append({"pasada": pasada, "pregunta": p["pregunta"], "tipo": p["tipo"],
                            "referencia": p["referencia"], **r,
                            "comprobaciones": comprobaciones, "ok_automatico": ok})
            fallidas = [k for k, v in comprobaciones.items() if v is False]
            print(f"{'✓' if ok else '✗'} [{n:>2}/{len(PREGUNTAS)}] {r['segundos']:>6.1f}s  {p['pregunta']}"
                  + (f"  → falla: {', '.join(fallidas)}" if fallidas else ""), flush=True)

    # Summary: % of answers that pass each check (only where it applies) and % fully OK per question type
    def porcentaje(valores):
        valores = [v for v in valores if v is not None]
        return round(sum(valores) / len(valores), 3) if valores else None

    resumen = {
        "modelo": modelo,
        "pasadas": pasadas,
        "ok_automatico": porcentaje([d["ok_automatico"] for d in detalle]),
        "por_comprobacion": {c: porcentaje([d["comprobaciones"][c] for d in detalle]) for c in detalle[0]["comprobaciones"]},
        "por_tipo": {t: porcentaje([d["ok_automatico"] for d in detalle if d["tipo"] == t])
                     for t in dict.fromkeys(p["tipo"] for p in PREGUNTAS)},
        "media_llamadas": round(sum(len(d["llamadas"]) for d in detalle) / len(detalle), 2),
        "segundos_media": round(sum(d["segundos"] for d in detalle) / len(detalle), 1),
    }
    RESULTADOS.write_text(json.dumps({"resumen": resumen, "detalle": detalle}, ensure_ascii=False, indent=2),
                          encoding="utf-8")
    print("\n" + json.dumps(resumen, ensure_ascii=False, indent=2))
    print(f"\nSaved to {RESULTADOS}")


if __name__ == "__main__":
    evaluar(sys.argv[1] if len(sys.argv) > 1 else "qwen3:14b", int(sys.argv[2]) if len(sys.argv) > 2 else 1)
