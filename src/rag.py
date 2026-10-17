import chromadb
from pathlib import Path
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
import json
from chunking_script import chunking
from functools import cache

DATA = Path(__file__).resolve().parent.parent / "data"
RAG_PATH = DATA/"chroma"
MODELO_EMBEDDINGS = "qwen3-embedding:0.6b"

@cache
def artistas_indexados() -> dict[str, str]:
    metadatas = obtener_coleccion().get(include=["metadatas"])["metadatas"]
    return {m["name"].lower(): m["name"] for m in metadatas}

@cache
def obtener_coleccion(modelo: str=MODELO_EMBEDDINGS):


    chroma_client = chromadb.PersistentClient(path=RAG_PATH)

    collection = chroma_client.get_or_create_collection(name=f"Chinook-{modelo.replace(":","_")}",embedding_function=OllamaEmbeddingFunction(model_name=modelo))

    return collection


def indexar():


    carpeta_wiki = DATA / "wiki"
    indice = json.loads((carpeta_wiki / "indice.json").read_text(encoding="utf-8"))
    coleccion = obtener_coleccion()

    for artista in indice:
        texto = (carpeta_wiki / artista["archivo"]).read_text(encoding="utf-8")
        documentos = chunking(texto, artista)

        content = [document.page_content for document in documentos]
        metadata = [document.metadata for document in documentos]

        coleccion.upsert(ids=[f"{artista["artist_id"]}_{pos}" for pos in range(len(content))],
                              documents=content,metadatas=metadata)


def buscar(query: str,coleccion, k: int=5,artist_name:str | None = None) -> list[dict]:

    artistas = artistas_indexados()

    where = None
    aviso = None
    if artist_name is not None and artist_name.lower() in artistas:
        where = {"name":artistas[artist_name.lower()]}
    elif artist_name is not None:
        # Unknown artist (e.g. a band member): search all artists instead of returning nothing,
        # so the model gets the answer from the documents and not from its own knowledge
        aviso = (f"There are no documents about {artist_name}. Showing results from all artists instead; "
                 f"they may not be about {artist_name}.\nAvailable artists: {", ".join(artistas.values())}")

    resultados = coleccion.query(query_texts=[query],n_results=k,where=where)
    documentos = [{"document":text,"url":meta["url"]} for text,meta in zip(resultados["documents"][0],resultados["metadatas"][0])]

    if aviso:
        documentos.insert(0, {"document":aviso,"url":None})
    return documentos




if __name__ == "__main__":
    if obtener_coleccion().count() == 0:
        indexar()
    coleccion = obtener_coleccion()
    print(f"Chunks indexed: {coleccion.count()}")

    # Manual search tests
    pruebas = [
        ("Where was the band formed?", "iron maiden"),   # indexed artist, lowercase
        ("Where was the band formed?", "Bob Dylan"),     # artist not indexed
        ("Which band did Freddie Mercury sing in?", "Freddie Mercury"),  # band member, not an artist
        ("Who was the drummer?", None),                  # no artist filter
    ]
    for query, artista in pruebas:
        print(f"\n===== {query} | {artista} =====")
        for resultado in buscar(query, coleccion, artist_name=artista):
            print(f"  {resultado['url']} | {resultado['document'][:100]!r}")