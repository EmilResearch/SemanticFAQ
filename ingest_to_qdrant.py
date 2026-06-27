"""
ingest_to_qdrant.py
-------------------
Script 2 — Embedding in batch e popolamento del Vector DB (Qdrant locale).

Pipeline:
  1. Legge `data.json`.
  2. "Esplode" tutte le domande di tutti i blocchi mantenendo il riferimento
     all'answer_id (id del blocco) e alla category.
  3. Genera gli embedding in batch (EMBEDDING_BATCH_SIZE per chiamata).
  4. (Re)crea la collection Qdrant e fa upsert di tutti i vettori.

Idempotenza: ad ogni esecuzione la collection viene ricreata da zero,
così non si accumulano duplicati nascosti.

Uso:
    python ingest_to_qdrant.py
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from pathlib import Path

from dotenv import load_dotenv
from tqdm import tqdm

import litellm
from litellm import embedding

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

import config

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("[ERROR] GEMINI_API_KEY non trovata. Copia .env.example in .env e inserisci la chiave.")
    sys.exit(1)

litellm.suppress_debug_info = True


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_data(path: str) -> list[dict]:
    """Carica data.json e fa validazione minima."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] {path} non trovato. Esegui prima `python prepare_data.py`.")
        sys.exit(1)

    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not data:
        print(f"[ERROR] {path} vuoto o non valido.")
        sys.exit(1)

    return data


def explode_questions(blocks: list[dict]) -> list[dict]:
    """
    Espande ogni blocco nelle sue varianti di domanda, mantenendo
    il riferimento al blocco padre (answer_id).
    """
    rows = []
    for block in blocks:
        for q in block.get("domande", []):
            q = q.strip()
            if not q:
                continue
            rows.append({
                "answer_id": block["id"],
                "category": block.get("category", ""),
                "question_text": q,
                "source_file": block.get("source_file", ""),
            })
    return rows


def embed_in_batches(texts: list[str], batch_size: int) -> list[list[float]]:
    """
    Genera gli embedding in batch usando LiteLLM.
    L'ordine dei vettori restituiti corrisponde all'ordine dei testi in input.
    """
    vectors: list[list[float]] = []
    n = len(texts)

    for start in tqdm(range(0, n, batch_size), desc="Embedding batches", unit="batch"):
        batch = texts[start:start + batch_size]
        resp = embedding(
            model=config.EMBEDDING_MODEL,
            input=batch,
            num_retries=config.LITELLM_NUM_RETRIES,
        )
        # LiteLLM normalizza l'output OpenAI-style: resp["data"] è una lista di dict
        # con chiave "embedding". L'ordine è garantito uguale all'input.
        batch_vecs = [item["embedding"] for item in resp["data"]]
        if len(batch_vecs) != len(batch):
            raise RuntimeError(
                f"Mismatch lunghezza batch embedding: atteso {len(batch)}, ricevuto {len(batch_vecs)}"
            )
        vectors.extend(batch_vecs)

    return vectors


def create_collection(client: QdrantClient) -> None:
    """Ricrea la collection da zero — idempotenza garantita."""
    # Usa i nuovi metodi per evitare il DeprecationWarning
    if client.collection_exists(config.COLLECTION_NAME):
        client.delete_collection(config.COLLECTION_NAME)
        
    client.create_collection(
        collection_name=config.COLLECTION_NAME,
        vectors_config=qmodels.VectorParams(
            size=config.VECTOR_SIZE,
            distance=qmodels.Distance.COSINE,
        ),
    )
    print(f"[INFO] Collection '{config.COLLECTION_NAME}' ricreata "
          f"(size={config.VECTOR_SIZE}, distance=COSINE).")
          
          
def upsert_points(client: QdrantClient, rows: list[dict], vectors: list[list[float]]) -> None:
    """Inserisce ogni domanda (con il suo embedding) come punto separato."""
    points = []
    for row, vec in zip(rows, vectors):
        points.append(
            qmodels.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{row['answer_id']}:{row['question_text']}")),
                vector=vec,
                payload={
                    "answer_id": row["answer_id"],
                    "category": row["category"],
                    "question_text": row["question_text"],
                    "source_file": row["source_file"],
                },
            )
        )

    # Upsert in batch per non saturare la memoria su collection grandi
    BATCH = 256
    for start in tqdm(range(0, len(points), BATCH), desc="Upserting", unit="batch"):
        client.upsert(
            collection_name=config.COLLECTION_NAME,
            points=points[start:start + BATCH],
            wait=True,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    print("=" * 70)
    print("FAQ Retrieval — Script 2: ingest_to_qdrant.py")
    print(f"  Embedding model : {config.EMBEDDING_MODEL}")
    print(f"  Vector size     : {config.VECTOR_SIZE}")
    print(f"  Qdrant path     : {config.QDRANT_PATH}")
    print(f"  Collection      : {config.COLLECTION_NAME}")
    print(f"  Batch size      : {config.EMBEDDING_BATCH_SIZE}")
    print("=" * 70)

    blocks = load_data(config.DATA_JSON)
    print(f"\n[INFO] Blocchi caricati: {len(blocks)}")

    rows = explode_questions(blocks)
    if not rows:
        print("[ERROR] Nessuna domanda trovata nei blocchi.")
        sys.exit(1)
    print(f"[INFO] Domande totali da embeddare: {len(rows)}")

    texts = [r["question_text"] for r in rows]
    print(f"[INFO] Avvio embedding in batch da {config.EMBEDDING_BATCH_SIZE}...")
    vectors = embed_in_batches(texts, batch_size=config.EMBEDDING_BATCH_SIZE)
    print(f"[INFO] Embedding generati: {len(vectors)}")

    # Inizializza Qdrant in modalità locale (path su disco, no server richiesto)
    client = QdrantClient(path=config.QDRANT_PATH)
    create_collection(client)
    upsert_points(client, rows, vectors)

    # Verifica
    info = client.get_collection(config.COLLECTION_NAME)
    print("\n" + "=" * 70)
    print(f"COMPLETATO. Punti in collection: {info.points_count}")
    print(f"Qdrant data: {config.QDRANT_PATH}")
    print("=" * 70)


if __name__ == "__main__":
    main()
