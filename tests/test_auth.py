"""Tests for registration, login, logout and access control."""
import sqlite3

from tests.helpers import make_app, post_form, register

PREDICT_BODY = {"longitude": -122.23, "latitude": 37.88, "housing_median_age": 41,
                "total_rooms": 880, "total_bedrooms": 129, "population": 322,
                "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY"}


def test_logged_out_home_shows_login_prompt_not_form(tmp_path):
    html = make_app(tmp_path).test_client().get("/").get_data(as_text=True)
    assert "Log in to use the predictor" in html
    assert 'name="longitude"' not in html


def test_predict_requires_login(tmp_path):
    res = make_app(tmp_path).test_client().post("/api/predict", json=PREDICT_BODY)
    assert res.status_code == 401


def test_register_logs_in_and_logout_works(tmp_path):
    c = make_app(tmp_path).test_client()
    res = register(c)
    assert res.status_code == 302 and "#predict" in res.headers["Location"]
    html = c.get("/").get_data(as_text=True)
    assert "Hi, Test User" in html and 'name="longitude"' in html
    assert post_form(c, "/logout").status_code == 302
    assert "Log in to use the predictor" in c.get("/").get_data(as_text=True)


def test_login_correct_and_wrong_password(tmp_path):
    app = make_app(tmp_path)
    register(app.test_client(), email="Test@Example.com")      # email is case-insensitive
    c = app.test_client()
    bad = post_form(c, "/login", email="test@example.com", password="wrong-password")
    assert bad.status_code == 400 and "Incorrect email or password" in bad.get_data(as_text=True)
    assert post_form(c, "/login", email="test@example.com", password="password123").status_code == 302
    assert c.get("/login").status_code == 302                  # already logged in


def test_register_validation_errors(tmp_path):
    c = make_app(tmp_path).test_client()
    assert register(c, password="short").status_code == 400
    assert register(c, email="not-an-email").status_code == 400
    assert register(c, name="  ").status_code == 400
    assert register(c).status_code == 302
    c2 = make_app(tmp_path).test_client()                      # same database file
    dup = register(c2)
    assert dup.status_code == 400 and "already exists" in dup.get_data(as_text=True)


def test_missing_csrf_token_is_rejected(tmp_path):
    c = make_app(tmp_path).test_client()
    res = c.post("/register", data={"name": "A", "email": "a@b.co", "password": "password123"})
    assert res.status_code == 400
    assert c.post("/logout").status_code == 400


def test_password_is_stored_hashed(tmp_path):
    register(make_app(tmp_path).test_client(), password="password123")
    stored = sqlite3.connect(tmp_path / "users.db").execute("SELECT password_hash FROM users").fetchone()[0]
    assert stored != "password123" and "password123" not in stored
