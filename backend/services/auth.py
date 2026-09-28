import os
import datetime
import functools

import jwt
from flask import request, jsonify, g
from werkzeug.security import generate_password_hash, check_password_hash

# Set a real, secret value for JWT_SECRET in your environment (.env locally,
# Render env vars in production). Never commit an actual secret.
JWT_SECRET = os.environ.get("JWT_SECRET", "dev-only-change-me")
JWT_ALGO = "HS256"
TOKEN_EXPIRY_HOURS = 24 * 7  # 7 days


def hash_password(plain_password: str) -> str:
    return generate_password_hash(plain_password)


def verify_password(plain_password: str, password_hash: str) -> bool:
    return check_password_hash(password_hash, plain_password)


def create_token(user_id: str, email: str, name: str = "") -> str:
    payload = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(hours=TOKEN_EXPIRY_HOURS),
        "iat": datetime.datetime.utcnow(),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGO)


def decode_token(token: str):
    """Returns the payload dict, or None if invalid/expired."""
    try:
        return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGO])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def require_auth(fn):
    """Route decorator: requires 'Authorization: Bearer <token>' header.
    On success, sets g.user_id and g.user_email before calling the route."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "missing or malformed Authorization header"}), 401

        token = auth_header.split(" ", 1)[1]
        payload = decode_token(token)
        if not payload:
            return jsonify({"error": "invalid or expired token"}), 401

        g.user_id = payload["user_id"]
        g.user_email = payload["email"]
        g.user_name = payload.get("name", "")
        return fn(*args, **kwargs)

    return wrapper