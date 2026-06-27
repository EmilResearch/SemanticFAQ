"""
chatbot.py
----------
Script 3 — CLI Chatbot con Top-K Aggregation.

Caratteristiche chiave:
  * Nessun LLM al runtime: zero allucinazioni, latenza minima, costo prossimo a zero.
  * Aggregazione degli score per `answer_id`: se più varianti di domanda
    della stessa risposta matchano la query, i loro score si sommano,
    rendendo il vincitore molto più robusto rispetto al singolo top-1.
  * Soglia OOD configurabile via --threshold.
  * Log automatico delle query "missed" su missed_queries.log.

Uso:
    python chatbot.py
    python chatbot.py --threshold 0.75
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

import litellm
from litellm import embedding

from qdrant_client import QdrantClient

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
def load_answers_dict(path: str) -> dict[str, dict]:
    """Carica data.json e costruisce un dict id -> blocco per lookup O(1)."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] {path} non trovato. Esegui prima `prepare_data.py` e `ingest_to_qdrant.py`.")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}


def embed_query(query: str) -> list[float]:
    """Embedding singolo per la query utente."""
    resp = embedding(
        model=config.EMBEDDING_MODEL,
        input=[query],
        num_retries=config.LITELLM_NUM_RETRIES,
    )
    return resp["data"][0]["embedding"]


def log_missed_query(query: str, top_score: float) -> None:
    """Append-only log delle query sotto soglia."""
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    line = f"{ts}\tscore={top_score:.4f}\tquery={query}\n"
    with open(config.MISSED_QUERIES_LOG, "a", encoding="utf-8") as f:
        f.write(line)


def aggregate_scores(search_results) -> dict[str, dict]:
    """
    Aggrega gli score dei Top-K risultati Qdrant per `answer_id`.

    Ritorna un dict:
      { answer_id: {
            "score": float (somma),
            "hits": int (numero di varianti che hanno matchato),
            "best_question": str (la variante con lo score singolo più alto),
            "best_single_score": float
        }, ... }
    """
    agg: dict[str, dict] = defaultdict(lambda: {
        "score": 0.0,
        "hits": 0,
        "best_question": "",
        "best_single_score": -1.0,
    })

    for r in search_results:
        payload = r.payload or {}
        aid = payload.get("answer_id")
        if not aid:
            continue
        bucket = agg[aid]
        bucket["score"] += float(r.score)
        bucket["hits"] += 1
        if r.score > bucket["best_single_score"]:
            bucket["best_single_score"] = float(r.score)
            bucket["best_question"] = payload.get("question_text", "")

    return agg


# -----------------------------------------------------------------------------
# Core
# -----------------------------------------------------------------------------
def answer_query(
    query: str,
    client: QdrantClient,
    answers_dict: dict[str, dict],
    threshold: float,
    debug: bool = False,
) -> None:
    """Esegue il retrieval + aggregazione e stampa la risposta."""
    try:
        qvec = embed_query(query)
    except Exception as e:
        print(f"[ERROR] Embedding fallito: {e}")
        return

    results = client.query_points(
        collection_name=config.COLLECTION_NAME,
        query=qvec,
        limit=config.TOP_K_RETRIEVAL,
        with_payload=True,
    ).points
    
    if not results:
        print("\nNon ho trovato informazioni pertinenti.")
        log_missed_query(query, 0.0)
        return

    agg = aggregate_scores(results)
    if not agg:
        print("\nNon ho trovato informazioni pertinenti.")
        log_missed_query(query, 0.0)
        return

    # Vincitore = answer_id con score AGGREGATO più alto
    winner_id, winner = max(agg.items(), key=lambda kv: kv[1]["score"])
    top_agg_score = winner["score"]

    if debug:
        print("\n--- DEBUG TOP-K ---")
        for r in results:
            print(f"  score={r.score:.4f}  aid={r.payload.get('answer_id')}  q={r.payload.get('question_text')}")
        print("--- DEBUG AGG ---")
        for aid, b in sorted(agg.items(), key=lambda kv: -kv[1]["score"]):
            print(f"  aid={aid}  agg={b['score']:.4f}  hits={b['hits']}  best_single={b['best_single_score']:.4f}")
        print("-------------------")

    # Controllo soglia sullo score AGGREGATO (non sul singolo top-1)
    if top_agg_score < threshold:
        print("\nNon ho trovato informazioni pertinenti su questo argomento nella mia base di conoscenza.")
        log_missed_query(query, top_agg_score)
        return

    block = answers_dict.get(winner_id)
    if not block:
        # Caso patologico: l'answer_id in Qdrant non esiste più in data.json
        print("\n[WARN] answer_id non trovato nel document store. Ri-esegui l'ingestion.")
        log_missed_query(query, top_agg_score)
        return

    print("\n" + "─" * 70)
    print(block["risposta"])
    print("─" * 70)
    print(f"Fonte: {block.get('source_file', 'n/d')}")
    if debug:
        print(f"[debug] score_aggregato={top_agg_score:.4f}  hits={winner['hits']}  "
              f"best_single={winner['best_single_score']:.4f}")


# -----------------------------------------------------------------------------
# CLI Loop
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="CLI chatbot Q-Q matching su Qdrant locale.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.DEFAULT_THRESHOLD,
        help=f"Soglia minima sullo score aggregato (default: {config.DEFAULT_THRESHOLD})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Mostra dettagli interni (Top-K e tabella di aggregazione).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FAQ Retrieval — Script 3: chatbot.py")
    print(f"  Embedding model : {config.EMBEDDING_MODEL}")
    print(f"  Qdrant path     : {config.QDRANT_PATH}")
    print(f"  Collection      : {config.COLLECTION_NAME}")
    print(f"  Top-K           : {config.TOP_K_RETRIEVAL}")
    print(f"  Threshold       : {args.threshold}")
    print(f"  Debug           : {args.debug}")
    print("=" * 70)
    print("Scrivi una domanda. Digita 'exit' o 'quit' per uscire.\n")

    answers_dict = load_answers_dict(config.DATA_JSON)
    client = QdrantClient(path=config.QDRANT_PATH)

    # Sanity check: la collection esiste?
    try:
        info = client.get_collection(config.COLLECTION_NAME)
        print(f"[INFO] Collection ok — {info.points_count} punti caricati.\n")
    except Exception:
        print(f"[ERROR] Collection '{config.COLLECTION_NAME}' non trovata. "
              f"Esegui prima `python ingest_to_qdrant.py`.")
        sys.exit(1)

    try:
        while True:
            try:
                query = input("Tu> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nA presto!")
                break

            if not query:
                continue
            if query.lower() in {"exit", "quit", ":q"}:
                print("A presto!")
                break

            answer_query(
                query=query,
                client=client,
                answers_dict=answers_dict,
                threshold=args.threshold,
                debug=args.debug,
            )
            print()
    finally:
        # Qdrant in modalità locale tiene un lockfile: chiusura esplicita
        try:
            client.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
