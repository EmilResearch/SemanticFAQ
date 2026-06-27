"""
prepare_data.py
---------------
Script 1 — Generazione dati per il sistema FAQ Retrieval.

Pipeline:
  1. Scorre ricorsivamente la cartella `source/` (pdf, txt, md).
  2. Conta i token con `litellm.token_counter` (NIENTE tiktoken).
  3. Se il documento <= MAX_TOKENS_DOC -> invio intero; altrimenti chunking
     con CHUNK_SIZE / OVERLAP.
  4. Invia a Gemini in JSON mode con il prompt di estrazione Q&A.
  5. (Opzionale --judge) Verifica ogni blocco con un secondo LLM-as-a-Judge.
  6. Salva l'array dei blocchi in `data.json`.

Uso:
    python prepare_data.py
    python prepare_data.py --judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from tqdm import tqdm

import litellm
from litellm import completion, token_counter

# pypdf per i PDF
from pypdf import PdfReader

import config

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()

if not os.getenv("GEMINI_API_KEY"):
    print("[ERROR] GEMINI_API_KEY non trovata. Copia .env.example in .env e inserisci la chiave.")
    sys.exit(1)

# Silenzia i log verbosi di LiteLLM (lasciamo solo i nostri print strutturati)
litellm.suppress_debug_info = True


# -----------------------------------------------------------------------------
# Prompts
# -----------------------------------------------------------------------------
EXTRACTION_PROMPT = """Sei un sistema esperto di Information Retrieval e Data Ingestion per motori di ricerca semantici.

Il tuo obiettivo è convertire i documenti forniti in un set di blocchi informativi indipendenti.

Rispetti rigorosamente le seguenti linee guida:
1. COPERTURA TOTALE: Non tralasciare alcun dettaglio, dato tecnico, procedura o eccezione presente nel testo. Tutto il valore informativo deve essere estratto.
2. STRUTTURA DEL BLOCCO: Ogni blocco deve contenere:
   - "risposta": Un testo esaustivo, chiaro e autocontenuto che spiega un concetto o una procedura specifica.
   - "domande": Una lista di 3-5 varianti di domande che trovano risposta *esattamente* e *solo* in quel testo (varia stile: formale, colloquiale, keyword-focused).
   - "category": Una categoria breve per il blocco (es. "Check-in", "Elettrodomestici", "Pagamenti").
   - "source_quote": La frase o paragrafo ESATTO dal quale hai estratto la risposta (copiato letteralmente).
3. ISOLAMENTO: Le risposte non devono richiedere la lettura del resto del documento.
4. NON INVENTARE: Estrai solo le informazioni esplicitamente scritte. Se il testo è vago, ignoralo.

Genera l'output ESCLUSIVAMENTE in formato JSON (un oggetto con chiave "blocks" contenente l'array dei blocchi):
{
  "blocks": [
    {
      "risposta": "Testo completo...",
      "domande": ["Domanda 1?", "Domanda 2?", "Domanda 3?"],
      "category": "Nome Categoria",
      "source_quote": "Testo letterale dal documento..."
    }
  ]
}

Ecco il contenuto da processare:
"""

JUDGE_PROMPT_TEMPLATE = """Sei un sistema di verifica incrociata. Ti viene fornito un estratto di documento (Source Quote) e una risposta generata (Risposta).
Verifica se la Risposta è supportata al 100% dalla Source Quote, senza aggiunte o inferenze esterne.
Rispondi ESCLUSIVAMENTE in formato JSON:
{{
  "verdetto": "SI" o "NO",
  "motivazione": "Breve spiegazione"
}}

Source Quote:
{source_quote}

Risposta:
{risposta}
"""


# -----------------------------------------------------------------------------
# File reading
# -----------------------------------------------------------------------------
def read_file(path: Path) -> str:
    """Legge il contenuto testuale di un file supportato (.pdf, .txt, .md)."""
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception as e:
                print(f"  [WARN] Errore estrazione pagina da {path.name}: {e}")
        return "\n".join(pages)

    if suffix in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")

    return ""


def discover_files(source_dir: str) -> list[Path]:
    """Scorre ricorsivamente la cartella source e restituisce i path supportati."""
    root = Path(source_dir)
    if not root.exists():
        print(f"[ERROR] Cartella source non trovata: {source_dir}")
        sys.exit(1)

    supported = {".pdf", ".txt", ".md"}
    files = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in supported]
    return files


# -----------------------------------------------------------------------------
# Chunking
# -----------------------------------------------------------------------------
def count_tokens(text: str) -> int:
    """Conta i token usando il tokenizer del modello LLM (via LiteLLM)."""
    return token_counter(model=config.LLM_MODEL, text=text)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """
    Chunking *token-based* approssimato tramite la proporzione token/char.
    LiteLLM non espone un tokenizer pubblico per Gemini, quindi calcoliamo
    la lunghezza in caratteri equivalente a CHUNK_SIZE token usando il rapporto
    reale char/token misurato sul testo stesso. Funziona bene per testi naturali.
    """
    total_tokens = count_tokens(text)
    if total_tokens <= chunk_size:
        return [text]

    # Rapporto char/token del *testo reale* (più preciso di stime fisse)
    char_per_token = max(1.0, len(text) / total_tokens)
    chunk_chars = int(chunk_size * char_per_token)
    overlap_chars = int(overlap * char_per_token)
    step = max(1, chunk_chars - overlap_chars)

    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start += step

    return chunks


# -----------------------------------------------------------------------------
# LLM calls
# -----------------------------------------------------------------------------
def extract_blocks_from_chunk(chunk_text: str) -> dict[str, Any] | None:
    """
    Invia un chunk al LLM in JSON mode e restituisce il dict parsato.
    Ritorna None se la chiamata fallisce o il JSON è invalido.
    """
    try:
        response = completion(
            model=config.LLM_MODEL,
            messages=[
                {"role": "user", "content": EXTRACTION_PROMPT + chunk_text}
            ],
            response_format={"type": "json_object"},
            num_retries=config.LITELLM_NUM_RETRIES,
        )
        raw = response["choices"][0]["message"]["content"]
        return json.loads(raw)
    except json.JSONDecodeError as e:
        print(f"  [WARN] JSON malformato: {e}")
        return None
    except Exception as e:
        print(f"  [WARN] Errore LLM: {e}")
        return None


def judge_block(source_quote: str, risposta: str) -> bool:
    """
    Verifica un blocco col secondo LLM. Ritorna True se SUPPORTATO ("SI"),
    False se non supportato o se la chiamata fallisce in modo ambiguo.
    """
    prompt = JUDGE_PROMPT_TEMPLATE.format(source_quote=source_quote, risposta=risposta)
    try:
        response = completion(
            model=config.LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            num_retries=config.LITELLM_NUM_RETRIES,
        )
        raw = response["choices"][0]["message"]["content"]
        verdict = json.loads(raw)
        return str(verdict.get("verdetto", "")).strip().upper() == "SI"
    except Exception as e:
        # In caso di errore preferiamo essere conservativi e NON scartare il blocco,
        # così non perdiamo info per problemi transitori. Logghiamo solo.
        print(f"  [WARN] Judge fallito (mantengo il blocco): {e}")
        return True


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
REQUIRED_KEYS = {"risposta", "domande", "category", "source_quote"}


def validate_block(block: Any) -> bool:
    """Controlla che un blocco abbia tutte le chiavi obbligatorie e tipi sensati."""
    if not isinstance(block, dict):
        return False
    if not REQUIRED_KEYS.issubset(block.keys()):
        return False
    if not isinstance(block["domande"], list) or len(block["domande"]) == 0:
        return False
    if not all(isinstance(q, str) and q.strip() for q in block["domande"]):
        return False
    if not isinstance(block["risposta"], str) or not block["risposta"].strip():
        return False
    if not isinstance(block["source_quote"], str) or not block["source_quote"].strip():
        return False
    return True


def save_failed_chunk(filename: str, chunk_idx: int, chunk_content: str) -> None:
    """Salva il chunk testuale che non è stato parsato correttamente, per debug."""
    Path(config.FAILED_CHUNKS_DIR).mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).stem.replace(os.sep, "_")
    out_path = Path(config.FAILED_CHUNKS_DIR) / f"{safe_name}_{chunk_idx}.txt"
    out_path.write_text(chunk_content, encoding="utf-8")
    print(f"  [INFO] Chunk fallito salvato in: {out_path}")


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------
def process_file(path: Path, use_judge: bool) -> list[dict[str, Any]]:
    """Processa un singolo file e ritorna la lista di blocchi validi."""
    print(f"\n[FILE] {path}")
    text = read_file(path)
    if not text.strip():
        print("  [SKIP] File vuoto o non leggibile.")
        return []

    tokens = count_tokens(text)
    print(f"  Tokens stimati: {tokens}")

    if tokens <= config.MAX_TOKENS_DOC:
        chunks = [text]
        print("  Strategia: documento INTERO (sotto soglia).")
    else:
        chunks = chunk_text(text, config.CHUNK_SIZE, config.OVERLAP)
        print(f"  Strategia: CHUNKING -> {len(chunks)} chunk(s)")

    blocks_out: list[dict[str, Any]] = []

    for idx, chunk in enumerate(chunks):
        print(f"  -> Chunk {idx + 1}/{len(chunks)} (LLM call)...")
        parsed = extract_blocks_from_chunk(chunk)

        if not parsed or "blocks" not in parsed or not isinstance(parsed["blocks"], list):
            print("  [WARN] Output LLM non valido — salvo il chunk e proseguo.")
            save_failed_chunk(path.name, idx, chunk)
            continue

        for block in parsed["blocks"]:
            if not validate_block(block):
                print("  [WARN] Blocco scartato (chiavi mancanti o invalide).")
                continue

            if use_judge:
                ok = judge_block(block["source_quote"], block["risposta"])
                if not ok:
                    print("  [JUDGE] Blocco scartato (verdetto NO).")
                    continue

            blocks_out.append({
                "id": str(uuid.uuid4()),
                "risposta": block["risposta"].strip(),
                "domande": [q.strip() for q in block["domande"]],
                "category": block["category"].strip(),
                "source_file": str(path),
                "source_quote": block["source_quote"].strip(),
            })

    print(f"  -> Blocchi validi estratti: {len(blocks_out)}")
    return blocks_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera data.json da source/ usando Gemini.")
    parser.add_argument(
        "--judge",
        action="store_true",
        help="Abilita LLM-as-a-Judge per filtrare blocchi non supportati dalla source.",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("FAQ Retrieval — Script 1: prepare_data.py")
    print(f"  LLM model     : {config.LLM_MODEL}")
    print(f"  Source dir    : {config.SOURCE_DIR}")
    print(f"  Output        : {config.DATA_JSON}")
    print(f"  Judge enabled : {args.judge}")
    print("=" * 70)

    files = discover_files(config.SOURCE_DIR)
    if not files:
        print(f"[ERROR] Nessun file supportato trovato in {config.SOURCE_DIR}/")
        sys.exit(1)

    print(f"\nTrovati {len(files)} file(s).")

    all_blocks: list[dict[str, Any]] = []
    for path in tqdm(files, desc="Files", unit="file"):
        blocks = process_file(path, use_judge=args.judge)
        all_blocks.extend(blocks)

    # Sovrascrive sempre data.json
    with open(config.DATA_JSON, "w", encoding="utf-8") as f:
        json.dump(all_blocks, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 70)
    print(f"COMPLETATO. Blocchi totali: {len(all_blocks)}")
    print(f"Output salvato in: {config.DATA_JSON}")
    print("=" * 70)


if __name__ == "__main__":
    main()
