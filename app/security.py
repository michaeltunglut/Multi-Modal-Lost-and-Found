import hashlib
import os
import secrets
from datetime import datetime, timezone, timedelta
from fastapi import Depends, HTTPException, Request, Response
from .database import db, now
SESSION_COOKIE = "lf_session"
SESSION_DAYS = 14
PBKDF2_ROUNDS = 310_000
COOKIE_SECURE = os.environ.get("SESSION_COOKIE_SECURE", "").lower() in ("1", "true", "yes")

def password_digest(password: str, salt: bytes) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS).hex()

def token_digest(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()

def optional_user(request: Request):
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    with db() as c:
        row = c.execute(
            """SELECT u.id,u.name,u.email,u.role FROM sessions s
               JOIN users u ON u.id=s.user_id
               WHERE s.token_hash=? AND s.expires_at>?""",
            (token_digest(token), now()),
        ).fetchone()
    return dict(row) if row else None

def signed_in_user(user=Depends(optional_user)):
    if not user:
        raise HTTPException(status_code=401, detail="Sign in to continue.")
    return user

def admin_user(user=Depends(signed_in_user)):
    if user["role"] != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin access required.")
    return user

def set_session(response: Response, user_id: int):
    token = secrets.token_urlsafe(32)
    expiry = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
    with db() as c:
        c.execute("DELETE FROM sessions WHERE expires_at<=?", (now(),))
        c.execute("INSERT INTO sessions(token_hash,user_id,expires_at) VALUES(?,?,?)", (token_digest(token), user_id, expiry.isoformat(timespec="seconds")))
    response.set_cookie(SESSION_COOKIE, token, max_age=SESSION_DAYS*24*60*60, httponly=True, secure=COOKIE_SECURE, samesite="lax", path="/")

def session_user(row):
    return {"id": row["id"], "name": row["name"], "email": row["email"], "role": row["role"]}
