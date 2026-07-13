from flask import Flask, request, jsonify
from flask_cors import CORS  # Import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import anthropic
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import os
import pickle
import hashlib
import hmac
import json
import sys
import logging
from functools import wraps

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Comma-separated list of allowed origins, e.g. "https://example.com,https://app.example.com"
# Defaults to "*" (open) if unset.
allowed_origins = os.environ.get("ALLOWED_ORIGINS", "*")
CORS(app, origins=allowed_origins.split(",") if allowed_origins != "*" else "*")

ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
anthropic_client = anthropic.Anthropic()
if not os.environ.get("ANTHROPIC_API_KEY") and not os.environ.get("ANTHROPIC_AUTH_TOKEN"):
    logger.warning(
        "No ANTHROPIC_API_KEY/ANTHROPIC_AUTH_TOKEN set — chat requests will fail until credentials are configured."
    )

# Shared API key required on the chat endpoint (every request costs a billed Claude
# call, so this must not be left open). Set via env var; unconfigured = refuse, not open.
API_KEY = os.environ.get("API_KEY")
if not API_KEY:
    logger.warning("API_KEY is not set — the chat endpoint will refuse all requests until it is configured.")

def require_api_key(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if not API_KEY:
            return jsonify({"error": "Service not configured"}), 503
        provided = request.headers.get("X-API-Key", "")
        if not hmac.compare_digest(provided, API_KEY):
            return jsonify({"error": "Unauthorized"}), 401
        return view(*args, **kwargs)
    return wrapped

# In-memory rate-limit store — fine for the single-worker Procfile config; move to
# Redis (storage_uri="redis://...") if you ever scale gunicorn beyond one worker.
limiter = Limiter(get_remote_address, app=app, default_limits=[], storage_uri="memory://")
CHAT_RATE_LIMIT = os.environ.get("CHAT_RATE_LIMIT", "10 per minute")

# Paths for saving embeddings and loading documents
embedding_file_path = os.path.join(os.getcwd(), "document_embeddings.pkl")
documents_file_path = os.path.join(os.getcwd(), "documents.json")

# Load pre-trained model and tokenizer for embeddings
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
try:
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    model = AutoModel.from_pretrained(embedding_model_name)
except Exception:
    logger.exception("Failed to load embedding model '%s'", embedding_model_name)
    sys.exit(1)

# Function to load the knowledge base from a structured JSON file.
# Each entry is {"id": <unique str>, "category": <str>, "text": <str>}.
# See documents.example.json for the schema and placeholder content to start from.
def load_documents(document_file=documents_file_path):
    try:
        with open(document_file, "r", encoding="utf-8") as f:
            entries = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"{document_file} not found. Copy documents.example.json to documents.json "
            "and replace its contents with your real knowledge base entries."
        )

    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{document_file} must contain a non-empty JSON array of entries")

    seen_ids = set()
    for entry in entries:
        entry_id = entry.get("id", "").strip() if isinstance(entry, dict) else ""
        text = entry.get("text", "").strip() if isinstance(entry, dict) else ""
        if not entry_id or not text:
            raise ValueError(f"Every entry in {document_file} needs a non-empty 'id' and 'text': {entry!r}")
        if entry_id in seen_ids:
            raise ValueError(f"Duplicate entry id '{entry_id}' in {document_file}")
        seen_ids.add(entry_id)

    return entries

# Function to convert text to embeddings
def embed_text(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)  # Get the embeddings
    return embeddings.cpu().numpy()

# Function to prepare FAISS index and save/load embeddings
def prepare_faiss_index(documents, embedding_file=embedding_file_path):
    documents_hash = hashlib.sha256(
        json.dumps(documents, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    cached = None
    if os.path.exists(embedding_file):
        with open(embedding_file, "rb") as f:
            cached = pickle.load(f)

    if isinstance(cached, dict) and cached.get("hash") == documents_hash:
        # documents.json hasn't changed since the cache was built — reuse it
        print(f"Loading document embeddings from {embedding_file}...")
        document_vectors = cached["vectors"]
    else:
        # Embed and save documents if the cache is missing or stale
        print("Embedding documents and saving them to file...")
        document_vectors = [embed_text(doc["text"]) for doc in documents]
        document_vectors = np.vstack(document_vectors)
        with open(embedding_file, "wb") as f:
            pickle.dump({"hash": documents_hash, "vectors": document_vectors}, f)

    # Initialize FAISS index
    embedding_dim = document_vectors.shape[1]
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(document_vectors)
    return index

# Load documents and prepare FAISS index
try:
    documents = load_documents()
    index = prepare_faiss_index(documents)
except Exception:
    logger.exception("Failed to load documents or build FAISS index")
    sys.exit(1)

# Distance above which a retrieval is considered "no relevant match" and the
# fallback answer is returned instead of asking Claude to answer from weak context.
# Unset by default — MiniLM L2 distance scale depends on the embedding data, so a
# guessed threshold could reject valid queries. Enable by setting this env var after
# checking the "Retrieval top distance" values logged per query in production.
_raw_threshold = os.environ.get("RETRIEVAL_DISTANCE_THRESHOLD")
RETRIEVAL_DISTANCE_THRESHOLD = float(_raw_threshold) if _raw_threshold else None

NO_CONTEXT_ANSWER = "عذرًا، لا تتوفر لدي معلومات كافية للإجابة عن هذا السؤال."

# Retrieve documents from FAISS index based on the query
def retrieve(query):
    query_vector = embed_text(query)  # Embed the query
    k = min(3, len(documents))
    distances, indices = index.search(query_vector, k=k)  # Top k results
    best_distance = float(distances[0][0])
    matched_ids = [documents[i]["id"] for i in indices[0]]
    logger.info("Retrieval top distance: %.4f, matched ids: %s", best_distance, matched_ids)

    if RETRIEVAL_DISTANCE_THRESHOLD is not None and best_distance > RETRIEVAL_DISTANCE_THRESHOLD:
        return None

    results = [documents[i]["text"].strip() for i in indices[0]]  # Return the top matching documents
    return " ".join(results)  # Combine the results for better context

# Ask Claude to answer the question using only the retrieved context
def generate_answer(question, context):
    response = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system=(
            "You are a support assistant. Answer the user's question using ONLY the "
            "information in the provided context. Respond in the same language as the "
            "question. If the context does not contain the answer, say you don't have "
            "enough information — do not use outside knowledge."
        ),
        messages=[{
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {question}",
        }],
    )
    return next(block.text for block in response.content if block.type == "text")

# Health check endpoint for platform readiness/liveness probes
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"})

# Chat endpoint that handles queries
@app.route('/', methods=['POST'])
@limiter.limit(CHAT_RATE_LIMIT)
@require_api_key
def chat():
    payload = request.get_json(silent=True) or {}
    user_input = payload.get("query")
    if not user_input:
        return jsonify({"error": "Missing 'query' field in request body"}), 400

    retrieved_docs = retrieve(user_input)

    if retrieved_docs is None:
        answer = NO_CONTEXT_ANSWER
    else:
        try:
            answer = generate_answer(user_input, retrieved_docs)
        except anthropic.RateLimitError:
            logger.exception("Anthropic rate limit hit")
            return jsonify({"error": "The assistant is receiving too many requests. Please try again shortly."}), 429
        except anthropic.APIStatusError as e:
            logger.exception("Anthropic API error (status %s)", e.status_code)
            return jsonify({"error": "The assistant is temporarily unavailable. Please try again."}), 502
        except anthropic.APIConnectionError:
            logger.exception("Anthropic connection error")
            return jsonify({"error": "The assistant is temporarily unavailable. Please try again."}), 502

    # Log the conversation to stdout, captured by the platform's log stream
    logger.info("Question: %s | Answer: %s", user_input, answer)

    return jsonify({"response": answer})

# Local/dev entrypoint; production uses `gunicorn app:app` (see Procfile)
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
