from app import create_app


def make_app(tmp_path, model_path=None, metrics_path=None):
    return create_app(model_path or tmp_path / "none.joblib", metrics_path or tmp_path / "none.json",
                      db_path=tmp_path / "users.db", secret_key="test-key")


def post_form(client, url, **data):
    """POST an HTML form with a valid CSRF token."""
    with client.session_transaction() as s:
        s["csrf"] = "tok"
    return client.post(url, data={"csrf_token": "tok", **data})


def register(client, name="Test User", email="test@example.com", password="password123"):
    return post_form(client, "/register", name=name, email=email, password=password)
