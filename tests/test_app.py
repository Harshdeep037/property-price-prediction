"""Tests for the Flask API. A tiny model is trained on the random stand-in table,
so these check the CODE (routes, validation, error handling), not real accuracy."""
import pytest

from tests.helpers import make_app, register
from train_model import main

VALID = {
    "longitude": -122.23, "latitude": 37.88, "housing_median_age": 41,
    "total_rooms": 880, "total_bedrooms": 129, "population": 322,
    "households": 126, "median_income": 8.3252, "ocean_proximity": "NEAR BAY",
}


@pytest.fixture
def client(fake_df, tmp_path):
    csv = tmp_path / "housing.csv"
    fake_df.to_csv(csv, index=False)
    main(skip_eda=True, data_path=csv, models_dir=tmp_path / "models",
         images_dir=tmp_path / "img", n_estimators=10, cv=2)
    app = make_app(tmp_path, tmp_path / "models" / "model.joblib", tmp_path / "models" / "metrics.json")
    c = app.test_client()
    register(c)            # the predictor needs a logged-in user
    return c


def test_home_and_health(client):
    assert client.get("/").status_code == 200
    body = client.get("/api/health").get_json()
    assert body == {"status": "ok", "model_loaded": True}


def test_valid_prediction(client):
    res = client.post("/api/predict", json=VALID)
    assert res.status_code == 200
    body = res.get_json()
    assert body["predicted_price"] > 0
    assert body["formatted"].startswith("$")


def test_numbers_sent_as_strings_are_accepted(client):
    as_text = {k: (str(v) if k != "ocean_proximity" else v) for k, v in VALID.items()}
    assert client.post("/api/predict", json=as_text).status_code == 200


@pytest.mark.parametrize("change, field", [
    ({"median_income": 99}, "median_income"),
    ({"median_income": "abc"}, "median_income"),
    ({"latitude": None}, "latitude"),
    ({"ocean_proximity": "MOON"}, "ocean_proximity"),
    ({"total_bedrooms": 5000}, "total_bedrooms"),     # more bedrooms than rooms
    ({"housing_median_age": True}, "housing_median_age"),
    ({"households": "nan"}, "households"),
])
def test_invalid_inputs_return_400_with_field_message(client, change, field):
    res = client.post("/api/predict", json={**VALID, **change})
    assert res.status_code == 400
    assert field in res.get_json()["fields"]


def test_missing_field_and_bad_body(client):
    incomplete = {k: v for k, v in VALID.items() if k != "population"}
    assert "population" in client.post("/api/predict", json=incomplete).get_json()["fields"]
    assert client.post("/api/predict", data="not json").status_code == 400
    assert client.post("/api/predict", json=[1, 2, 3]).status_code == 400


def test_metrics_endpoint(client):
    body = client.get("/api/metrics").get_json()
    assert body["best_model"] in body["models"]


def test_unknown_route_is_json_404(client):
    res = client.get("/nope")
    assert res.status_code == 404 and "error" in res.get_json()


def test_missing_model_gives_503(tmp_path):
    c = make_app(tmp_path).test_client()
    register(c)
    assert c.post("/api/predict", json=VALID).status_code == 503
    assert c.get("/api/metrics").status_code == 404
    assert c.get("/api/health").get_json()["model_loaded"] is False


def test_home_page_has_form_and_all_fields(client):
    html = client.get("/").get_data(as_text=True)
    assert "Predict Price" in html
    for name in list(VALID):
        assert f'name="{name}"' in html
    assert "static/js/script.js" in html and "static/css/style.css" in html
    assert client.get("/static/js/script.js").status_code == 200
    assert client.get("/static/css/style.css").status_code == 200
