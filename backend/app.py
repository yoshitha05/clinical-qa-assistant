import os

from dotenv import load_dotenv

load_dotenv()

from flask import Flask, jsonify, request, send_from_directory, g
from flask_cors import CORS
from google.api_core.exceptions import ResourceExhausted

from services import db, rag, auth

rag.configure_gemini()

app = Flask(__name__, static_folder="static", static_url_path="/")
CORS(app)

ALLOWED_DOC_TYPES = {"discharge_summary", "clinical_note", "research_paper"}


# ---------- auth ----------

@app.post("/api/auth/signup")
def signup():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""
    name = (payload.get("name") or "").strip()

    if not email or not password or not name:
        return jsonify({"error": "name, email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "password must be at least 6 characters"}), 400

    try:
        user_id = db.create_user(email, auth.hash_password(password), name)
    except ValueError as e:
        return jsonify({"error": str(e)}), 409

    token = auth.create_token(user_id, email, name)
    return jsonify({"token": token, "email": email, "name": name}), 201


@app.post("/api/auth/login")
def login():
    payload = request.get_json(silent=True) or {}
    email = (payload.get("email") or "").strip().lower()
    password = payload.get("password") or ""

    user = db.get_user_by_email(email)
    if not user or not auth.verify_password(password, user["password_hash"]):
        return jsonify({"error": "invalid email or password"}), 401

    token = auth.create_token(user["_id"], user["email"], user.get("name", ""))
    return jsonify({"token": token, "email": user["email"], "name": user.get("name", "")})


@app.get("/api/auth/me")
@auth.require_auth
def me():
    return jsonify({"user_id": g.user_id, "email": g.user_email, "name": g.user_name})


# ---------- documents ----------

@app.post("/api/upload")
@auth.require_auth
def upload_document():
    """Accepts a real .txt or .pdf file (multipart), chunks it, embeds each
    chunk, stores the vectors in FAISS and the text + metadata in MongoDB."""
    doc_type = request.form.get("doc_type", "clinical_note")
    if doc_type not in ALLOWED_DOC_TYPES:
        return jsonify({"error": f"doc_type must be one of {sorted(ALLOWED_DOC_TYPES)}"}), 400
    # Uploads attached mid-conversation from the chat input are marked hidden
    # so they don't clutter the persistent Documents list - they're only
    # usable within the chat they were attached to.
    hidden = request.form.get("hidden", "false").lower() == "true"

    if "file" not in request.files:
        return jsonify({"error": "a file is required"}), 400
    file = request.files["file"]
    filename = request.form.get("filename") or file.filename
    raw_bytes = file.read()

    lower_name = (file.filename or "").lower()
    if lower_name.endswith(".pdf"):
        raw_text = rag.extract_text_from_pdf(raw_bytes)
    elif lower_name.endswith(".txt"):
        raw_text = raw_bytes.decode("utf-8", errors="ignore")
    else:
        return jsonify({"error": "only .txt and .pdf files are supported"}), 400

    chunks = rag.chunk_text(raw_text, doc_type=doc_type)
    if not chunks:
        return jsonify({"error": "no extractable text found in document"}), 400

    try:
        # Embed first - only save the document record once we know embedding
        # actually succeeded, so a failed upload doesn't leave an empty/broken
        # document behind.
        vectors = rag.embed_texts(chunks, task_type="retrieval_document")
    except ResourceExhausted:
        return jsonify({
            "error": "The Gemini API's free-tier request limit has been reached for now. "
                     "Please wait a bit and try again."
        }), 429

    vector_ids = rag.next_vector_id(len(chunks))
    rag.add_vectors(vectors, vector_ids)

    document_id = db.create_document(filename, doc_type, raw_text, g.user_id, hidden=hidden)

    for idx, (chunk, vec_id) in enumerate(zip(chunks, vector_ids)):
        db.create_chunk(document_id, idx, chunk, vec_id, g.user_id, embedding=vectors[idx].tolist())

    return jsonify({
        "document_id": document_id,
        "filename": filename,
        "doc_type": doc_type,
        "chunks_stored": len(chunks),
    }), 201


@app.get("/api/documents")
@auth.require_auth
def list_documents():
    return jsonify(db.list_documents(g.user_id))


@app.get("/api/documents/<document_id>")
@auth.require_auth
def get_document(document_id):
    doc = db.get_document(document_id, g.user_id)
    if not doc:
        return jsonify({"error": "not found"}), 404
    return jsonify(doc)


@app.delete("/api/documents/<document_id>")
@auth.require_auth
def delete_document(document_id):
    """Deletes a document's MongoDB records and its FAISS vectors, if it
    belongs to the logged-in user."""
    vector_ids = db.delete_document(document_id, g.user_id)
    if vector_ids is None:
        return jsonify({"error": "not found"}), 404
    if vector_ids:
        rag.remove_vectors(vector_ids)
    return jsonify({"deleted": document_id}), 200


# ---------- ask ----------

@app.post("/api/ask")
@auth.require_auth
def ask_question():
    payload = request.get_json(silent=True) or {}
    question = payload.get("question")
    document_id = payload.get("document_id")
    conversation_id = payload.get("conversation_id")
    top_k = int(payload.get("top_k", 5))

    if not question:
        return jsonify({"error": "question is required"}), 400

    restricted_doc = None
    if document_id:
        restricted_doc = db.get_document(document_id, g.user_id)
        if not restricted_doc:
            return jsonify({"error": "document not found"}), 404

    if conversation_id and not db.get_conversation(conversation_id, g.user_id):
        return jsonify({"error": "conversation not found"}), 404

    # No conversation given - this is a new chat, so start one. Title it
    # right away (rather than only after a successful answer) so it still
    # shows up properly in the sidebar even if the Gemini call below fails
    # (e.g. quota exhausted) - the user can then refresh and continue once
    # the quota resets, instead of the chat vanishing.
    if not conversation_id:
        conversation_id = db.create_conversation(g.user_id, document_id)
        db.set_conversation_title_if_empty(conversation_id, question[:60])

    try:
        query_vector = rag.embed_texts([question], task_type="retrieval_query")[0]

        if document_id:
            # Restricted to one document - rank that document's own chunks
            # directly instead of searching the whole index and filtering
            # after, which can miss the document entirely once the account
            # has many other documents' chunks.
            doc_chunks = db.get_chunks_for_document(document_id, g.user_id)
            candidates = rag.rank_chunks(query_vector, doc_chunks, top_k=top_k)
        else:
            # Same reasoning as the document-scoped branch above: rank this
            # user's own chunks directly by their stored embedding instead of
            # a FAISS index search, which silently returns nothing once the
            # index has drifted (e.g. after documents were deleted).
            all_chunks = db.get_all_chunks_for_user(g.user_id)
            candidates = rag.rank_chunks(query_vector, all_chunks, top_k=top_k)

        context_chunks = [c["chunk_text"] for c in candidates]
        # Only tailor the persona/instructions when the question is restricted to
        # one document of a known type -- with "all documents" selected the
        # context could mix types, so we keep the generic instructions.
        doc_type = restricted_doc["doc_type"] if restricted_doc else None
        answer = rag.generate_answer(question, context_chunks, doc_type=doc_type)
    except ResourceExhausted:
        return jsonify({
            "error": "The Gemini API's free-tier request limit has been reached for now. "
                     "Please wait a bit and try again.",
            "conversation_id": conversation_id,
        }), 429

    db.log_query(document_id, question, answer, [c["_id"] for c in candidates], g.user_id, conversation_id)
    db.set_conversation_title_if_empty(conversation_id, question[:60])
    db.touch_conversation(conversation_id)

    return jsonify({
        "question": question,
        "answer": answer,
        "conversation_id": conversation_id,
        "sources": [
            {"chunk_id": c["_id"], "document_id": c["document_id"], "text": c["chunk_text"]}
            for c in candidates
        ],
    })


@app.get("/api/conversations")
@auth.require_auth
def list_conversations():
    return jsonify(db.list_conversations(g.user_id))


@app.get("/api/conversations/<conversation_id>/messages")
@auth.require_auth
def conversation_messages(conversation_id):
    conv = db.get_conversation(conversation_id, g.user_id)
    if not conv:
        return jsonify({"error": "not found"}), 404
    return jsonify({
        "conversation": conv,
        "messages": db.list_conversation_messages(conversation_id, g.user_id),
    })


@app.delete("/api/conversations/<conversation_id>")
@auth.require_auth
def delete_conversation(conversation_id):
    deleted = db.delete_conversation(conversation_id, g.user_id)
    if not deleted:
        return jsonify({"error": "not found"}), 404
    return jsonify({"deleted": conversation_id}), 200


@app.get("/api/documents/<document_id>/history")
@auth.require_auth
def document_history(document_id):
    return jsonify(db.list_queries(g.user_id, document_id=document_id))


@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})


# --- Serve the built React app for everything else ---
@app.get("/", defaults={"path": ""})
@app.get("/<path:path>")
def serve_react(path):
    static_dir = app.static_folder
    if path and os.path.exists(os.path.join(static_dir, path)):
        return send_from_directory(static_dir, path)
    return send_from_directory(static_dir, "index.html")


if __name__ == "__main__":
    # host="0.0.0.0" so the server is reachable from other devices on the
    # network (a phone over Wi-Fi), not just from this machine itself.
    app.run(debug=True, port=5001, host="0.0.0.0")