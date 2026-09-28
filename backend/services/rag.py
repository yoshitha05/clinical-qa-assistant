"""The RAG pipeline: chunk -> embed -> FAISS index -> retrieve -> Gemini generate.

FAISS only ever stores vectors and hands back vector ids (integers).
Mongo (see db.py) is what turns a vector id back into readable chunk text.
"""
import io
import os
import threading

import faiss
import google.generativeai as genai
import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768
GEN_MODEL = "gemini-3.6-flash"

_lock = threading.Lock()
_index = None
_index_path = None

# Absolute path to backend/services/, so the default index location is the
# same regardless of the working directory the server was launched from.
_SERVICES_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_INDEX_DIR = os.path.join(_SERVICES_DIR, "..", "faiss_index")


def configure_gemini():
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])


def _index_files():
    base = os.environ.get("FAISS_INDEX_DIR") or _DEFAULT_INDEX_DIR
    base = os.path.abspath(base)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, "index.faiss")


def get_index():
    """Load the FAISS index from disk once, or create a fresh empty one."""
    global _index, _index_path
    if _index is None:
        _index_path = _index_files()
        if os.path.exists(_index_path):
            _index = faiss.read_index(_index_path)
            print(f"[rag] loaded FAISS index from {_index_path} ({_index.ntotal} vectors)")
        else:
            # IndexIDMap lets us assign our own integer ids to vectors,
            # so a vector id can be stored directly on the Mongo chunk document.
            _index = faiss.IndexIDMap(faiss.IndexFlatL2(EMBED_DIM))
            print(f"[rag] no existing index at {_index_path}, starting fresh")
    return _index


def _save_index():
    faiss.write_index(_index, _index_path)

def extract_text_from_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages).strip()


# Chunk size/overlap tuned per document type: research papers tend to have
# longer, denser paragraphs (methods, results, references) where a bigger
# chunk keeps a full thought together; discharge summaries and clinical
# notes are already terse, so smaller chunks keep retrieval precise.
CHUNK_PROFILES = {
    "research_paper": {"chunk_size": 1400, "chunk_overlap": 200},
    "clinical_note": {"chunk_size": 700, "chunk_overlap": 100},
    "discharge_summary": {"chunk_size": 800, "chunk_overlap": 120},
}
DEFAULT_CHUNK_PROFILE = {"chunk_size": 800, "chunk_overlap": 120}


def chunk_text(raw_text: str, doc_type: str = None) -> list[str]:
    profile = CHUNK_PROFILES.get(doc_type, DEFAULT_CHUNK_PROFILE)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=profile["chunk_size"],
        chunk_overlap=profile["chunk_overlap"],
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(raw_text) if c.strip()]


EMBED_BATCH_SIZE = 100  # Gemini's embed_content accepts a batch of texts per call


def embed_texts(texts: list[str], task_type: str) -> np.ndarray:
    """task_type is 'retrieval_document' for chunks being stored,
    or 'retrieval_query' for a question being asked -- Gemini's embedding
    model optimizes the vector differently depending on which side it's for.

    Batches multiple texts into a single API call instead of one call per
    text - a 30-chunk document used to mean 30 sequential network round
    trips (the main reason uploads felt slow); now it's a single call (or a
    couple, for very large documents)."""
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")

    vectors = []
    for i in range(0, len(texts), EMBED_BATCH_SIZE):
        batch = texts[i:i + EMBED_BATCH_SIZE]
        result = genai.embed_content(
            model=EMBED_MODEL,
            content=batch,
            task_type=task_type,
            output_dimensionality=EMBED_DIM,
        )
        vectors.extend(result["embedding"])
    return np.array(vectors, dtype="float32")


def add_vectors(vectors: np.ndarray, ids: list[int]):
    with _lock:
        index = get_index()
        index.add_with_ids(vectors, np.array(ids, dtype="int64"))
        _save_index()

def remove_vectors(ids: list[int]):
    with _lock:
        index = get_index()
        index.remove_ids(np.array(ids, dtype="int64"))
        _save_index()

def next_vector_id(count: int) -> list[int]:
    """FAISS doesn't hand out ids for us with IndexIDMap, so we track the
    next free id ourselves based on how many vectors are already stored."""
    index = get_index()
    start = int(index.ntotal)
    return list(range(start, start + count))


def search(query_vector: np.ndarray, top_k: int = 5) -> list[int]:
    index = get_index()
    if index.ntotal == 0:
        return []
    distances, ids = index.search(query_vector.reshape(1, -1), min(top_k, index.ntotal))
    return [int(i) for i in ids[0] if i != -1]


def rank_chunks(query_vector: np.ndarray, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """Ranks a KNOWN, specific set of chunks (each already carrying its own
    stored `embedding` - see db.create_chunk) by closeness to the query.

    Used when a question is restricted to one document: a global top-k
    search across the whole FAISS index (search(), above) can easily miss
    that document's own chunks entirely once the account has many other
    documents' vectors crowding out the nearest neighbors, so instead we
    score only that document's own chunks directly. This deliberately
    avoids FAISS's reconstruct()/make_direct_map(), which aren't supported
    on every IndexIDMap build (confirmed broken on this install)."""
    scored = [c for c in chunks if c.get("embedding")]
    if not scored:
        return []
    vectors = np.array([c["embedding"] for c in scored], dtype="float32")
    query = query_vector.reshape(1, -1)
    distances = np.sum((vectors - query) ** 2, axis=1)
    order = np.argsort(distances)[:top_k]
    return [scored[i] for i in order]


PERSONA_BY_DOC_TYPE = {
    "discharge_summary": (
        "You are a clinical assistant reading a hospital discharge summary. "
        "Use precise medical terminology and quote exact values (dosages, vitals, dates) "
        "exactly as written rather than rounding or paraphrasing them."
    ),
    "clinical_note": (
        "You are a clinical assistant reading a clinical or progress note. "
        "Focus on the patient's current status, the clinician's reasoning, and any "
        "plan or next steps mentioned."
    ),
    "research_paper": (
        "You are a research assistant reading an academic paper. When relevant, refer "
        "to specific sections, algorithms, tables, or figures by the name or number "
        "the paper gives them, and preserve technical terminology precisely."
    ),
}
DEFAULT_PERSONA = (
    "You are a document assistant. Answer using precise language grounded in the "
    "provided text."
)


def generate_answer(question: str, context_chunks: list[str], doc_type: str = None) -> str:
    persona = PERSONA_BY_DOC_TYPE.get(doc_type, DEFAULT_PERSONA)
    context = "\n\n---\n\n".join(context_chunks) if context_chunks else "(no relevant passages found)"
    prompt = (
        f"{persona} You're chatting with a user inside a document Q&A app. "
        "If their message is a greeting, thanks, or casual small talk rather than a "
        "question about a document, reply naturally and briefly the way a friendly "
        "assistant would - you do not need the context passages for that. "
        "If it IS a question about a document, answer using ONLY the context passages "
        "below, and if the answer is not contained in the context, say so plainly "
        "instead of guessing.\n\n"
        f"Context passages:\n{context}\n\n"
        f"User: {question}\n\n"
        "Response:"
    )
    model = genai.GenerativeModel(GEN_MODEL)
    response = model.generate_content(prompt)
    return response.text.strip()