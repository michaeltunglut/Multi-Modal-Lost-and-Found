import hmac
import os
import re
import secrets
import sqlite3
from fastapi import APIRouter, Depends, Form, HTTPException, Request, Response
from ..database import db, now
from ..security import optional_user, password_digest, set_session, session_user, token_digest, SESSION_COOKIE, COOKIE_SECURE
router = APIRouter()

@router.get("/api/auth/me")
def who_am_i(user=Depends(optional_user)):
    return {"authenticated": bool(user), "user": user}

@router.post("/api/auth/register", status_code=201)
def register(response: Response, name: str = Form(...), email: str = Form(...), password: str = Form(...), admin_token: str = Form(default="")):
    name = name.strip()
    email = email.strip().casefold()
    if len(name) < 2 or len(name) > 100:
        raise HTTPException(status_code=400, detail="Enter a name between 2 and 100 characters.")
    if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
        raise HTTPException(status_code=400, detail="Enter a valid email address.")
    if len(password) < 10 or len(password) > 200:
        raise HTTPException(status_code=400, detail="Use a password between 10 and 200 characters.")
    salt = secrets.token_bytes(16)
    digest = password_digest(password, salt)
    admin_email = os.environ.get("ADMIN_EMAIL", "").strip().casefold()
    configured_admin_token = os.environ.get("ADMIN_SETUP_TOKEN", "")
    is_admin = bool(admin_email and configured_admin_token and email == admin_email and hmac.compare_digest(admin_token, configured_admin_token))
    role = "ADMIN" if is_admin else "USER"
    try:
        with db() as c:
            cur = c.execute("INSERT INTO users(name,email,password_hash,password_salt,role,created_at) VALUES(?,?,?,?,?,?)", (name,email,digest,salt.hex(),role,now()))
            user_id = cur.lastrowid
    except sqlite3.IntegrityError as e:
        raise HTTPException(status_code=409, detail="An account with that email already exists.") from e
    set_session(response, user_id)
    return {"authenticated": True, "user": {"id": user_id, "name": name, "email": email, "role": role}}

@router.post("/api/auth/login")
def login(response: Response, email: str = Form(...), password: str = Form(...)):
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE email=?", (email.strip().casefold(),)).fetchone()
    if not row or not hmac.compare_digest(password_digest(password, bytes.fromhex(row["password_salt"])), row["password_hash"]):
        raise HTTPException(status_code=401, detail="Email or password is incorrect.")
    set_session(response, row["id"])
    return {"authenticated": True, "user": session_user(row)}

@router.post("/api/auth/logout")
def logout(request: Request, response: Response):
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        with db() as c:
            c.execute("DELETE FROM sessions WHERE token_hash=?", (token_digest(token),))
    response.delete_cookie(SESSION_COOKIE, path="/", httponly=True, secure=COOKIE_SECURE, samesite="lax")
    return {"ok": True}
