"""Flask backend: website, login system and prediction API.

Run:   python app.py        then open  http://127.0.0.1:5000
"""
import json
import logging
import os
import secrets

import joblib
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

from src import config
from src.auth import AuthError, create_user, init_db, verify_user
from src.validation import validate_input

logger = logging.getLogger(__name__)


def create_app(model_path=config.MODEL_PATH, metrics_path=config.METRICS_PATH,
               db_path=config.DB_PATH, secret_key=None):
    app = Flask(__name__)
    app.config.update(
        # A random key per start is safe for local use (you are logged out on restart).
        # Set the SECRET_KEY environment variable to stay logged in across restarts.
        SECRET_KEY=secret_key or os.environ.get("SECRET_KEY") or secrets.token_hex(32),
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    init_db(db_path)

    model = None
    try:
        model = joblib.load(model_path)
        logger.info("Model loaded from %s", model_path)
    except FileNotFoundError:
        logger.warning("No model at %s - run train_model.py first.", model_path)

    # ---- CSRF protection for the HTML forms ---------------------------------
    def csrf_token():
        if "csrf" not in session:
            session["csrf"] = secrets.token_hex(16)
        return session["csrf"]

    def csrf_ok():
        sent = request.form.get("csrf_token", "")
        return "csrf" in session and secrets.compare_digest(session["csrf"], sent)

    @app.context_processor
    def inject_template_values():
        return {"user": session.get("user_name"), "csrf_token": csrf_token}

    # ---- Pages ----------------------------------------------------------------
    @app.get("/")
    def home():
        fields = [
            {"name": n, "label": config.FIELD_INFO[n][0], "hint": config.FIELD_INFO[n][1],
             "step": config.FIELD_INFO[n][2], "min": lo, "max": hi}
            for n, (lo, hi) in config.INPUT_LIMITS.items()
        ]
        return render_template("index.html", fields=fields,
                               categories=config.OCEAN_CATEGORIES, example=config.EXAMPLE_INPUT)

    def handle_auth(mode):
        if "user_id" in session:
            return redirect(url_for("home"))
        error, name, email = None, "", ""
        if request.method == "POST":
            name, email = request.form.get("name", ""), request.form.get("email", "")
            password = request.form.get("password", "")
            if not csrf_ok():
                error = "Your session expired. Please try again."
            else:
                try:
                    if mode == "register":
                        user = create_user(db_path, name, email, password)
                    else:
                        user = verify_user(db_path, email, password)
                        if user is None:
                            raise AuthError("Incorrect email or password.")
                    session.clear()
                    session["user_id"], session["user_name"] = user["id"], user["name"]
                    return redirect(url_for("home") + "#predict")
                except AuthError as exc:
                    error = str(exc)
        status = 400 if error else 200
        return render_template("auth.html", mode=mode, error=error, name=name, email=email), status

    @app.route("/login", methods=["GET", "POST"])
    def login():
        return handle_auth("login")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        return handle_auth("register")

    @app.post("/logout")
    def logout():
        if not csrf_ok():
            return jsonify(error="Invalid request."), 400
        session.clear()
        return redirect(url_for("home"))

    # ---- API ------------------------------------------------------------------
    @app.get("/api/health")
    def health():
        return jsonify(status="ok", model_loaded=model is not None)

    @app.get("/api/metrics")
    def metrics():
        try:
            with open(metrics_path) as f:
                return jsonify(json.load(f))
        except FileNotFoundError:
            return jsonify(error="Metrics not found. Run train_model.py first."), 404

    @app.post("/api/predict")
    def predict():
        if "user_id" not in session:
            return jsonify(error="Please log in to use the predictor."), 401
        if model is None:
            return jsonify(error="Model not found. Run train_model.py first."), 503

        data = request.get_json(silent=True)
        clean, errors = validate_input(data)
        if errors:
            return jsonify(error="Please fix the highlighted inputs.", fields=errors), 400

        try:
            row = pd.DataFrame([clean])
            price = max(0.0, float(model.predict(row)[0]))
        except Exception:
            logger.exception("Prediction failed")
            return jsonify(error="Something went wrong while predicting. Please try again."), 500

        return jsonify(predicted_price=round(price), formatted=f"${price:,.0f}")

    @app.errorhandler(404)
    def not_found(_):
        return jsonify(error="Not found."), 404

    @app.errorhandler(500)
    def server_error(_):
        return jsonify(error="Internal server error."), 500

    return app


app = create_app()

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
