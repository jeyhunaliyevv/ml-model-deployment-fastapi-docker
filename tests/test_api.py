from fastapi.testclient import TestClient
from app.main import app
from app import model_store

VALID = {"age": 35, "sex": "female", "bmi": 27.1, "children": 1,
         "smoker": "no", "region": "northeast"}

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}

def test_predict_success(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["high_cost_probability"] == 0.88
    assert d["predicted_high_cost"] is True
    assert d["threshold_usd"] == 20000.0
    assert d["decision_threshold"] == 0.5

def test_reject_negative_age(client):
    assert client.post("/predict", json={**VALID, "age": -1}).status_code == 422

def test_reject_invalid_bmi(client):
    assert client.post("/predict", json={**VALID, "bmi": 0}).status_code == 422

def test_reject_unexpected_field(client):
    assert client.post("/predict", json={**VALID, "charges": 100000}).status_code == 422

def test_reject_invalid_region(client):
    assert client.post("/predict", json={**VALID, "region": "unknown"}).status_code == 422

def test_reject_missing_required_field(client):
    d = dict(VALID); d.pop("smoker")
    assert client.post("/predict", json=d).status_code == 422

def test_model_unavailable(monkeypatch):
    def fail(): raise FileNotFoundError("missing")
    monkeypatch.setattr(model_store, "get_model", fail)
    with TestClient(app) as c:
        assert c.get("/health").status_code == 503
        r = c.post("/predict", json=VALID)
        assert r.status_code == 503
        assert "Model unavailable" in r.json()["detail"]

def test_prediction_failure(client, monkeypatch):
    class Broken:
        def predict_proba(self, x): raise RuntimeError("oops")
    monkeypatch.setattr(model_store, "get_model", lambda: Broken())
    r = client.post("/predict", json=VALID)
    assert r.status_code == 500
    assert r.json()["detail"] == "Prediction failed"
