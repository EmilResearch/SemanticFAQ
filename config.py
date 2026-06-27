"""
config.py
---------
Configurazione centralizzata del progetto FAQ Retrieval System.
Tutte le costanti "magic" del sistema vivono qui per facilitare tuning e manutenzione.
"""

# =============================================================================
# MODELLI LLM ED EMBEDDING (via LiteLLM)
# =============================================================================
# Nota: i nomi dei modelli seguono la convenzione "provider/model" di LiteLLM.
LLM_MODEL = "gemini/gemini-2.5-flash"
EMBEDDING_MODEL = "gemini/gemini-embedding-001"

# Dimensione del vettore prodotto da text-embedding-004
VECTOR_SIZE = 3072

# =============================================================================
# QDRANT (modalità locale embedded — no server, no Docker)
# =============================================================================
QDRANT_PATH = "./qdrant_data"
COLLECTION_NAME = "faq_matching"

# =============================================================================
# CHUNKING & TOKEN BUDGET (Script 1 — prepare_data.py)
# =============================================================================
# Se il documento è <= MAX_TOKENS_DOC, lo mandiamo intero al LLM.
# Altrimenti spezziamo in chunk da CHUNK_SIZE token con OVERLAP token di sovrapposizione.
MAX_TOKENS_DOC = 10_000
CHUNK_SIZE = 8_000
OVERLAP = 500

# =============================================================================
# RETRIEVAL (Script 3 — chatbot.py)
# =============================================================================
# Quanti match candidati recuperiamo da Qdrant prima dell'aggregazione per answer_id.
TOP_K_RETRIEVAL = 10

# Soglia minima sullo score *aggregato* per considerare la risposta affidabile (OOD guard).
DEFAULT_THRESHOLD = 0.70

# =============================================================================
# EMBEDDING BATCHING (Script 2 — ingest_to_qdrant.py)
# =============================================================================
# Numero di domande inviate per singola chiamata di embedding.
EMBEDDING_BATCH_SIZE = 100

# =============================================================================
# RESILIENZA / RETRY
# =============================================================================
# Numero di retry automatici (backoff esponenziale gestito internamente da LiteLLM)
# per gestire rate-limit e errori transitori delle API Gemini.
LITELLM_NUM_RETRIES = 3

# =============================================================================
# PATH FILES
# =============================================================================
SOURCE_DIR = "./source"
DATA_JSON = "./data.json"
FAILED_CHUNKS_DIR = "./failed_chunks"
MISSED_QUERIES_LOG = "./missed_queries.log"
