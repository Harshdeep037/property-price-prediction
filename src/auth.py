"""User accounts stored in a small SQLite file. Passwords are stored hashed."""
import re
import sqlite3
from contextlib import contextmanager

from werkzeug.security import check_password_hash, generate_password_hash

EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_DUMMY_HASH = generate_password_hash("not-a-real-password")


class AuthError(ValueError):
    """A problem the user can fix (shown on the login/register page)."""


@contextmanager
def _db(path):
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with _db(path) as conn:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS users ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, "
            "email TEXT NOT NULL UNIQUE, password_hash TEXT NOT NULL, "
            "created_at TEXT DEFAULT CURRENT_TIMESTAMP)"
        )


def create_user(path, name, email, password):
    name, email, password = (name or "").strip(), (email or "").strip().lower(), password or ""
    if not 1 <= len(name) <= 60:
        raise AuthError("Enter your name (up to 60 characters).")
    if len(email) > 254 or not EMAIL_RE.match(email):
        raise AuthError("Enter a valid email address.")
    if not 8 <= len(password) <= 128:
        raise AuthError("Password must be 8 to 128 characters.")
    try:
        with _db(path) as conn:
            cur = conn.execute(
                "INSERT INTO users (name, email, password_hash) VALUES (?, ?, ?)",
                (name, email, generate_password_hash(password)),
            )
            return {"id": cur.lastrowid, "name": name}
    except sqlite3.IntegrityError:
        raise AuthError("An account with this email already exists.")


def verify_user(path, email, password):
    """Return {"id", "name"} if the email and password match, else None."""
    with _db(path) as conn:
        row = conn.execute(
            "SELECT id, name, password_hash FROM users WHERE email = ?",
            ((email or "").strip().lower(),),
        ).fetchone()
    if row is None:
        check_password_hash(_DUMMY_HASH, password or "")   # same work for unknown emails
        return None
    if check_password_hash(row["password_hash"], password or ""):
        return {"id": row["id"], "name": row["name"]}
    return None
