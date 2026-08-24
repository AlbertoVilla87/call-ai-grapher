"""FastAPI server that keeps pipeline sessions alive for the web UI.

The upload builds every stage once (detector, classifier, stylizer) and stores
them in an in-memory session, so moving the improvement regulator afterwards
only pays for stylization and recomposition. This mirrors what the browser
needs: one expensive "analyze" call and cheap "render" calls while the user
drags the alpha slider.
"""

import base64
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from call_ai_grapher.pipeline.factory import (
    build_classifier,
    build_detector,
    build_page_denoiser,
    build_stylizer,
)
from call_ai_grapher.pipeline.recomposer import Recomposer
from call_ai_grapher.pipeline.types import StyledChar
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

_MAX_SESSIONS = 8
_JPEG_QUALITY = 90


@dataclass
class Session:
    """Everything needed to restyle a page without rebuilding the pipeline."""

    page: np.ndarray
    boxes: list
    stylizer: object
    created_at: float = field(default_factory=time.time)


class RenderRequest(BaseModel):
    """Body of the render endpoint."""

    alpha: float


class AnalyzeResponse(BaseModel):
    session_id: str
    char_count: int
    labels: list[str]
    before: str


class RenderResponse(BaseModel):
    after: str
    elapsed_ms: int


class _SessionStore:
    """Thread-safe in-memory session registry with a small capacity."""

    def __init__(self, capacity: int = _MAX_SESSIONS):
        self._capacity = capacity
        self._sessions: dict = {}
        self._lock = threading.Lock()

    def put(self, session_id: str, session: Session) -> None:
        with self._lock:
            while len(self._sessions) >= self._capacity:
                oldest = min(self._sessions, key=lambda key: self._sessions[key].created_at)
                del self._sessions[oldest]
            self._sessions[session_id] = session

    def get(self, session_id: str) -> Optional[Session]:
        with self._lock:
            return self._sessions.get(session_id)

    def pop(self, session_id: str) -> None:
        with self._lock:
            self._sessions.pop(session_id, None)


def create_app() -> FastAPI:
    """Assemble the API application.

    :return: the FastAPI app ready to be served by uvicorn
    :rtype: FastAPI
    """
    app = FastAPI(title="CallAIgrapher API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    store = _SessionStore()
    render_lock = threading.Lock()

    @app.get("/api/health")
    def health():
        return {"ok": True}

    @app.post("/api/sessions", response_model=AnalyzeResponse)
    async def analyze_session(
        image: UploadFile = File(...),
        detector_backend: str = Form("mser"),
        yolo_model: str = Form("models/character_detector.pt"),
        confidence: float = Form(0.25),
        classifier_path: str = Form(""),
        stylizer_backend: str = Form("baseline"),
        stylizer_model: str = Form("models/char_stylizer.pt"),
        autoencoder_model: str = Form("models/char_autoencoder.pt"),
        alphabet_dir: str = Form("dataset/alphabet"),
        denoise_page: bool = Form(False),
    ):
        """Run detection over the uploaded page and open a rendering session.

        :param image: scanned page as PNG/JPEG bytes
        :type image: UploadFile
        :param detector_backend: detection backend name, "mser" or "yolo"
        :type detector_backend: str
        :param yolo_model: path to the trained YOLO weights (backend "yolo")
        :type yolo_model: str
        :param confidence: minimum detection confidence (backend "yolo")
        :type confidence: float
        :param classifier_path: classifier checkpoint, or empty to skip labeling
        :type classifier_path: str
        :param stylizer_backend: stylization backend name, "baseline", "neural" or "latent"
        :type stylizer_backend: str
        :param stylizer_model: pix2pix checkpoint (backend "neural")
        :type stylizer_model: str
        :param autoencoder_model: autoencoder checkpoint (backend "latent")
        :type autoencoder_model: str
        :param alphabet_dir: alphabet dataset with the pretty reference glyphs (backend "latent")
        :type alphabet_dir: str
        :param denoise_page: flatten illumination and remove specks before detection
        :type denoise_page: bool
        :return: session id, character count, labels found and the page ready to display
        :rtype: AnalyzeResponse
        """
        raw = await image.read()
        page = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if page is None:
            raise HTTPException(status_code=400, detail="That file could not be read as an image")

        try:
            preprocessor = build_page_denoiser(denoise_page)
            if preprocessor is not None:
                page = preprocessor.clean(page)
            detector = build_detector(detector_backend, yolo_model, confidence)
            classifier = build_classifier(classifier_path or None)
            stylizer = build_stylizer(
                stylizer_backend,
                stylizer_model=stylizer_model,
                autoencoder_model=autoencoder_model,
                alphabet_dir=alphabet_dir,
                classifier_model=classifier_path or None,
            )
        except Exception as error:
            logging.exception("Failed to build the pipeline")
            raise HTTPException(status_code=400, detail=f"Pipeline error: {error}") from error

        started = time.perf_counter()
        boxes = detector.detect(page)
        if classifier is not None:
            boxes = classifier.classify(page, boxes)
        logger.info("Detected %d characters in %.2fs", len(boxes), time.perf_counter() - started)

        session_id = uuid.uuid4().hex
        store.put(session_id, Session(page=page, boxes=boxes, stylizer=stylizer))
        labels = sorted({box.label for box in boxes if box.label})
        return AnalyzeResponse(
            session_id=session_id,
            char_count=len(boxes),
            labels=labels[:20],
            before=_encode(page),
        )

    @app.post("/api/sessions/{session_id}/render", response_model=RenderResponse)
    def render_session(session_id: str, request: RenderRequest):
        """Stylize every detected character at `alpha` and recompose the page.

        :param session_id: identifier returned by the analyze call
        :type session_id: str
        :param request: improvement amount in [0, 1]
        :type request: RenderRequest
        :return: the improved page and how long the scribe took
        :rtype: RenderResponse
        """
        session = store.get(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="That session has expired; upload the page again")

        alpha = min(max(request.alpha, 0.0), 1.0)
        started = time.perf_counter()
        with render_lock:
            chars = [
                StyledChar(
                    box=box,
                    original=session.page[box.y : box.y2, box.x : box.x2],
                    styled=session.stylizer.stylize(box.crop_from(session.page), alpha, label=box.label),
                )
                for box in session.boxes
            ]
            improved = Recomposer().compose(session.page, chars)
        return RenderResponse(after=_encode(improved), elapsed_ms=int((time.perf_counter() - started) * 1000))

    @app.delete("/api/sessions/{session_id}")
    def delete_session(session_id: str):
        """Release a session once the browser no longer needs it.

        :param session_id: identifier returned by the analyze call
        :type session_id: str
        :return: confirmation payload
        :rtype: dict
        """
        store.pop(session_id)
        return {"ok": True}

    _mount_web_dist(app)
    return app


def main(argv=None):
    """Launch the API server.

    The host and port can be overridden with the HOST and PORT environment
    variables; they default to 127.0.0.1:8000.

    :param argv: unused, kept for console-script parity
    :type argv: None
    """
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    import os

    import uvicorn

    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(create_app(), host=host, port=port, log_level="info")


def _mount_web_dist(app: FastAPI) -> None:
    """Serve the built web UI when it exists, so one process hosts everything.

    Looks for `web/dist` next to the repository root (or the path in the
    WEB_DIST environment variable); when present it is mounted at "/" and the
    SPA fallback serves index.html for unknown paths.

    :param app: application being assembled
    :type app: FastAPI
    """
    import os

    candidates = [os.environ.get("WEB_DIST"), Path.cwd() / "web" / "dist"]
    dist = next((Path(path).resolve() for path in candidates if path and Path(path).is_dir()), None)
    if dist is None:
        return
    app.mount("/", StaticFiles(directory=str(dist), html=True), name="web")
    logger.info("Serving web UI from %s", dist)


def _encode(image: np.ndarray) -> str:
    """Encode a BGR page into a JPEG data URL for transport to the browser.

    :param image: BGR page image
    :type image: np.ndarray
    :return: JPEG data URL
    :rtype: str
    """
    ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), _JPEG_QUALITY])
    if not ok:
        raise RuntimeError("Failed to encode the page as JPEG")
    return "data:image/jpeg;base64," + base64.b64encode(buffer.tobytes()).decode("ascii")


app = create_app()
