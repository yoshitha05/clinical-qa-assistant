"""MongoDB access layer.

Collections:
  - users:     signed-up accounts (email, hashed password)
  - documents: metadata + raw text for each uploaded file, tied to a user
  - chunks:    each chunk of text, tied back to a document, tagged with its FAISS vector id
  - queries:   a log of every question asked, the answer given, and which chunks were used, tied to a user
"""
import os
from datetime import datetime, timezone

import certifi
from bson import ObjectId
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError

_client = None
_db = None


def get_db():
    global _client, _db
    if _db is None:
        uri = os.environ["MONGODB_URI"]
        _client = MongoClient(uri, tlsCAFile=certifi.where())
        _db = _client.get_default_database()
        _db.users.create_index("email", unique=True)
    return _db


# ---------- users ----------

def create_user(email: str, password_hash: str, name: str) -> str:
    """Creates a user. Raises ValueError if the email is already registered."""
    db = get_db()
    try:
        result = db.users.insert_one({
            "email": email,
            "name": name,
            "password_hash": password_hash,
            "created_at": datetime.now(timezone.utc),
        })
    except DuplicateKeyError:
        raise ValueError("an account with this email already exists")
    return str(result.inserted_id)


def get_user_by_email(email: str):
    db = get_db()
    user = db.users.find_one({"email": email})
    if user:
        user["_id"] = str(user["_id"])
    return user


def get_user_by_id(user_id: str):
    db = get_db()
    user = db.users.find_one({"_id": ObjectId(user_id)})
    if user:
        user["_id"] = str(user["_id"])
    return user


# ---------- documents ----------

def create_document(filename: str, doc_type: str, raw_text: str, user_id: str, hidden: bool = False) -> str:
    """hidden=True is used for documents attached mid-conversation from the
    chat input: they're still fully usable for RAG/answering, but are kept
    out of the persistent Documents list/sidebar/Documents view - they only
    live in the chat they were attached to."""
    db = get_db()
    result = db.documents.insert_one({
        "filename": filename,
        "doc_type": doc_type,
        "raw_text": raw_text,
        "user_id": ObjectId(user_id),
        "upload_date": datetime.now(timezone.utc),
        "hidden": hidden,
    })
    return str(result.inserted_id)


def delete_document(document_id: str, user_id: str):
    """Deletes the document only if it belongs to user_id. Returns the FAISS
    vector ids to remove, or None if the document wasn't found/owned."""
    db = get_db()
    oid = ObjectId(document_id)
    owned = db.documents.find_one({"_id": oid, "user_id": ObjectId(user_id)}, {"_id": 1})
    if not owned:
        return None
    chunk_docs = list(db.chunks.find({"document_id": oid}, {"faiss_vector_id": 1}))
    vector_ids = [c["faiss_vector_id"] for c in chunk_docs]
    db.chunks.delete_many({"document_id": oid})
    db.documents.delete_one({"_id": oid})
    return vector_ids


def list_documents(user_id: str):
    """Only non-hidden documents - i.e. the persistent library, not documents
    attached inline mid-conversation from the chat input."""
    db = get_db()
    docs = db.documents.find(
        {"user_id": ObjectId(user_id), "hidden": {"$ne": True}}, {"raw_text": 0}
    ).sort("upload_date", -1)
    out = []
    for d in docs:
        d["_id"] = str(d["_id"])
        d["user_id"] = str(d["user_id"])
        out.append(d)
    return out


def get_document(document_id: str, user_id: str):
    db = get_db()
    doc = db.documents.find_one({"_id": ObjectId(document_id), "user_id": ObjectId(user_id)})
    if doc:
        doc["_id"] = str(doc["_id"])
        doc["user_id"] = str(doc["user_id"])
    return doc


# ---------- chunks ----------

def create_chunk(document_id: str, chunk_index: int, chunk_text: str, faiss_vector_id: int, user_id: str,
                  embedding: list[float] | None = None) -> str:
    """embedding is this chunk's own embedding vector, stored alongside it so a
    question restricted to this document can rank the document's own chunks
    directly (see rag.rank_chunks) without relying on FAISS reconstruct(),
    which isn't supported on every IndexIDMap build."""
    db = get_db()
    doc = {
        "document_id": ObjectId(document_id),
        "user_id": ObjectId(user_id),
        "chunk_index": chunk_index,
        "chunk_text": chunk_text,
        "faiss_vector_id": faiss_vector_id,
    }
    if embedding is not None:
        doc["embedding"] = embedding
    result = db.chunks.insert_one(doc)
    return str(result.inserted_id)


def get_chunks_for_document(document_id: str, user_id: str):
    """All chunks belonging to one document, in order. Used when a question
    is restricted to a specific document - we rank these directly rather
    than doing a global top-k search across every document the user owns
    and filtering afterward, which can miss this document entirely once the
    account has many other documents' chunks."""
    db = get_db()
    docs = list(db.chunks.find({
        "document_id": ObjectId(document_id),
        "user_id": ObjectId(user_id),
    }).sort("chunk_index", 1))
    for d in docs:
        d["_id"] = str(d["_id"])
        d["document_id"] = str(d["document_id"])
        d["user_id"] = str(d["user_id"])
    return docs


def get_all_chunks_for_user(user_id: str):
    """Every chunk belonging to this user's persistent (non-hidden)
    documents - i.e. the "IMPORTANT DOCUMENTS" library, not documents
    attached inline mid-chat (those are hidden=True and scoped only to the
    one chat they were uploaded in, so they must NOT leak into 'all
    documents' answers in other chats).

    Used for 'all documents' scoped questions - like get_chunks_for_document,
    this ranks chunks directly by their own stored embedding instead of
    trusting a FAISS index search, which drifts out of sync once documents
    have been deleted (FAISS vector-id reuse after removal isn't reliable
    with this setup)."""
    db = get_db()
    visible_doc_ids = [
        d["_id"]
        for d in db.documents.find(
            {"user_id": ObjectId(user_id), "hidden": {"$ne": True}}, {"_id": 1}
        )
    ]
    if not visible_doc_ids:
        return []
    docs = list(db.chunks.find({
        "document_id": {"$in": visible_doc_ids},
        "user_id": ObjectId(user_id),
    }))
    for d in docs:
        d["_id"] = str(d["_id"])
        d["document_id"] = str(d["document_id"])
        d["user_id"] = str(d["user_id"])
    return docs


def get_chunks_by_faiss_ids(faiss_ids: list[int], user_id: str):
    """Fetch chunk text (and parent document id) for a list of FAISS vector ids,
    scoped to the given user, preserving the order FAISS returned them in
    (closest match first)."""
    db = get_db()
    docs = list(db.chunks.find({
        "faiss_vector_id": {"$in": faiss_ids},
        "user_id": ObjectId(user_id),
    }))
    by_id = {d["faiss_vector_id"]: d for d in docs}
    ordered = [by_id[i] for i in faiss_ids if i in by_id]
    for d in ordered:
        d["_id"] = str(d["_id"])
        d["document_id"] = str(d["document_id"])
        d["user_id"] = str(d["user_id"])
    return ordered


# ---------- conversations ----------

def create_conversation(user_id: str, document_id: str | None = None) -> str:
    db = get_db()
    now = datetime.now(timezone.utc)
    result = db.conversations.insert_one({
        "user_id": ObjectId(user_id),
        "document_id": ObjectId(document_id) if document_id else None,
        "title": None,
        "created_at": now,
        "updated_at": now,
    })
    return str(result.inserted_id)


def set_conversation_title_if_empty(conversation_id: str, title: str):
    db = get_db()
    db.conversations.update_one(
        {"_id": ObjectId(conversation_id), "title": None},
        {"$set": {"title": title}},
    )


def touch_conversation(conversation_id: str):
    db = get_db()
    db.conversations.update_one(
        {"_id": ObjectId(conversation_id)},
        {"$set": {"updated_at": datetime.now(timezone.utc)}},
    )


def get_conversation(conversation_id: str, user_id: str):
    db = get_db()
    conv = db.conversations.find_one({"_id": ObjectId(conversation_id), "user_id": ObjectId(user_id)})
    if conv:
        conv["_id"] = str(conv["_id"])
        conv["user_id"] = str(conv["user_id"])
        conv["document_id"] = str(conv["document_id"]) if conv.get("document_id") else None
    return conv


def list_conversations(user_id: str, limit: int = 50):
    db = get_db()
    convs = list(db.conversations.find({"user_id": ObjectId(user_id)}).sort("updated_at", -1).limit(limit))
    for c in convs:
        c["_id"] = str(c["_id"])
        c["user_id"] = str(c["user_id"])
        c["document_id"] = str(c["document_id"]) if c.get("document_id") else None
    return convs


def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Deletes a conversation and all of its logged Q&A, only if it belongs
    to user_id. Returns True if something was deleted."""
    db = get_db()
    oid = ObjectId(conversation_id)
    owned = db.conversations.find_one({"_id": oid, "user_id": ObjectId(user_id)}, {"_id": 1})
    if not owned:
        return False
    db.queries.delete_many({"conversation_id": oid, "user_id": ObjectId(user_id)})
    db.conversations.delete_one({"_id": oid})
    return True


# ---------- queries (individual Q&A, grouped into conversations) ----------

def log_query(document_id: str | None, question: str, answer: str, used_chunk_ids: list[str],
              user_id: str, conversation_id: str):
    db = get_db()
    db.queries.insert_one({
        "document_id": ObjectId(document_id) if document_id else None,
        "user_id": ObjectId(user_id),
        "conversation_id": ObjectId(conversation_id),
        "question": question,
        "answer": answer,
        "used_chunk_ids": used_chunk_ids,
        "created_at": datetime.now(timezone.utc),
    })


def list_queries(user_id: str, document_id: str | None = None, limit: int = 50):
    """Flat list of Q&A, most recent first - used for the per-document history endpoint."""
    db = get_db()
    filt = {"user_id": ObjectId(user_id)}
    if document_id:
        filt["document_id"] = ObjectId(document_id)
    items = list(db.queries.find(filt).sort("created_at", -1).limit(limit))
    for i in items:
        i["_id"] = str(i["_id"])
        i["user_id"] = str(i["user_id"])
        i["document_id"] = str(i["document_id"]) if i.get("document_id") else None
        i["conversation_id"] = str(i["conversation_id"]) if i.get("conversation_id") else None
    return items


def list_conversation_messages(conversation_id: str, user_id: str):
    """Messages within one conversation, oldest first (chat order)."""
    db = get_db()
    items = list(db.queries.find({
        "conversation_id": ObjectId(conversation_id),
        "user_id": ObjectId(user_id),
    }).sort("created_at", 1))
    for i in items:
        i["_id"] = str(i["_id"])
        i["user_id"] = str(i["user_id"])
        i["conversation_id"] = str(i["conversation_id"])
        i["document_id"] = str(i["document_id"]) if i.get("document_id") else None
    return items