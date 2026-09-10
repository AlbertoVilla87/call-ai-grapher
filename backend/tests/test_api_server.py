import cv2
import numpy as np
from call_ai_grapher.api.server import create_app
from call_ai_grapher.pipeline.detector import CharacterDetector
from call_ai_grapher.pipeline.types import CharBox
from fastapi.testclient import TestClient


def _client():
    return TestClient(create_app())


def _page_jpeg():
    page = np.full((60, 80), 230, dtype=np.uint8)
    cv2.circle(page, (30, 30), 8, 40, -1)
    ok, buffer = cv2.imencode(".jpg", page)
    assert ok
    return buffer.tobytes()


def test_health_returns_ok():
    response = _client().get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_analyze_rejects_non_image_uploads():
    response = _client().post(
        "/api/sessions",
        files={"image": ("page.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 400
    assert "could not be read" in response.json()["detail"]


def test_analyze_and_render_roundtrip(monkeypatch):
    """The HTTP contract holds: upload -> session -> renders -> release.

    MSER itself is not exercised here (it is not deterministic on synthetic
    pages); the detector is stubbed so the test covers the API layer only.
    """

    def single_box(self, page):
        return [CharBox(x=10, y=10, width=20, height=20)]

    monkeypatch.setattr(CharacterDetector, "detect", single_box)
    client = _client()
    analyze = client.post("/api/sessions", files={"image": ("page.jpeg", _page_jpeg(), "image/jpeg")})

    assert analyze.status_code == 200
    body = analyze.json()
    assert body["session_id"]
    assert body["char_count"] == 1
    assert body["before"].startswith("data:image/jpeg;base64,")

    render = client.post("/api/sessions/" + body["session_id"] + "/render", json={"alpha": 0.8})

    assert render.status_code == 200
    rendered = render.json()
    assert rendered["after"].startswith("data:image/jpeg;base64,")
    assert rendered["elapsed_ms"] >= 0

    release = client.delete("/api/sessions/" + body["session_id"])

    assert release.status_code == 200
    assert client.post("/api/sessions/" + body["session_id"] + "/render", json={"alpha": 0.5}).status_code == 404


def test_render_clamps_alpha_out_of_range():
    client = _client()
    response = client.post("/api/sessions", files={"image": ("page.jpeg", _page_jpeg(), "image/jpeg")})
    session_id = response.json()["session_id"]

    render = client.post(f"/api/sessions/{session_id}/render", json={"alpha": 42.0})

    assert render.status_code == 200
