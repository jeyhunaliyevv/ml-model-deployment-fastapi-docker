import pytest
from fastapi.testclient import TestClient
from app.main import app
from app import model_store

class DummyModel:
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.12, 0.88] for _ in range(len(X))])

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(model_store, "get_model", lambda: DummyModel())
    with TestClient(app) as client:
        yield client
