"""Gradio web UI for the handwriting improvement pipeline.

Upload a scanned page and the app detects and labels every character once;
the improvement slider then re-stylizes and re-composes the page instantly,
so the alpha regulator can be explored without paying for detection again.
"""
import logging

import cv2
import gradio as gr
import numpy as np

from call_ai_grapher.pipeline.factory import (
    build_classifier,
    build_detector,
    build_stylizer,
)
from call_ai_grapher.pipeline.recomposer import Recomposer
from call_ai_grapher.pipeline.types import StyledChar


def analyze_page(
    image,
    detector_backend,
    yolo_model,
    confidence,
    classifier_path,
    stylizer_backend,
    stylizer_model,
    autoencoder_model,
    alphabet_dir,
):
    """Run detection and classification over the uploaded page.

    Builds every pipeline stage up front so slider updates only pay for
    stylization and recomposition afterwards.

    :param image: uploaded page as an RGB array
    :type image: Optional[np.ndarray]
    :param detector_backend: detection backend name, "mser" or "yolo"
    :type detector_backend: str
    :param yolo_model: path to the trained YOLO weights
    :type yolo_model: str
    :param confidence: minimum detection confidence
    :type confidence: float
    :param classifier_path: classifier checkpoint, or empty to skip labeling
    :type classifier_path: str
    :param stylizer_backend: stylization backend name
    :type stylizer_backend: str
    :param stylizer_model: pix2pix checkpoint (backend "neural")
    :type stylizer_model: str
    :param autoencoder_model: autoencoder checkpoint (backend "latent")
    :type autoencoder_model: str
    :param alphabet_dir: alphabet dataset with the pretty reference glyphs
    :type alphabet_dir: str
    :return: session state, the page ready to display and a status line
    :rtype: Tuple[dict, Optional[np.ndarray], str]
    """
    if image is None:
        return None, None, "Upload a scanned page first"
    page = _to_bgr(image)
    try:
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
        return None, _to_rgb(page), f"Pipeline error: {error}"

    boxes = detector.detect(page)
    if classifier is not None:
        boxes = classifier.classify(page, boxes)

    state = {"page": page, "boxes": boxes, "stylizer": stylizer}
    status = f"Detected {len(boxes)} characters"
    labels = sorted({box.label for box in boxes if box.label})
    if labels:
        shown = ", ".join(labels[:20]) + ("..." if len(labels) > 20 else "")
        status += f" · labels: {shown}"
    return state, _to_rgb(page), status


def render_page(state, alpha):
    """Stylize every detected character at `alpha` and recompose the page.

    :param state: session state built by `analyze_page`
    :type state: Optional[dict]
    :param alpha: improvement amount in [0, 1]
    :type alpha: float
    :return: the improved page as an RGB array, or None when nothing was analyzed yet
    :rtype: Optional[np.ndarray]
    """
    if not state or alpha is None:
        return None
    page = state["page"]
    stylizer = state["stylizer"]
    chars = [
        StyledChar(
            box=box,
            original=page[box.y : box.y2, box.x : box.x2],
            styled=stylizer.stylize(box.crop_from(page), alpha, label=box.label),
        )
        for box in state["boxes"]
    ]
    improved = Recomposer().compose(page, chars)
    return _to_rgb(improved)


def create_app():
    """Assemble the Gradio interface.

    :return: the Blocks app ready to be launched
    :rtype: gr.Blocks
    """
    with gr.Blocks(title="CallAIgrapher") as demo:
        gr.Markdown("# CallAIgrapher\nImprove the handwriting of a scanned document towards a pretty style.")
        state = gr.State(value=None)

        with gr.Row():
            with gr.Column():
                page_in = gr.Image(type="numpy", label="Scanned page")
                alpha = gr.Slider(minimum=0.0, maximum=1.0, value=0.8, step=0.05, label="Improvement amount (alpha)")
                with gr.Accordion("Backends", open=False):
                    detector_backend = gr.Radio(["mser", "yolo"], value="mser", label="Detector")
                    yolo_model = gr.Textbox(
                        value="models/character_detector.pt", label="YOLO weights", info='Detector backend "yolo"'
                    )
                    confidence = gr.Slider(
                        minimum=0.0,
                        maximum=1.0,
                        value=0.25,
                        step=0.05,
                        label="Detection confidence",
                        info='Backend "yolo"',
                    )
                    classifier_path = gr.Textbox(
                        value="", label="Classifier checkpoint", info="Optional; required by the latent stylizer"
                    )
                    stylizer_backend = gr.Radio(["baseline", "neural", "latent"], value="baseline", label="Stylizer")
                    stylizer_model = gr.Textbox(
                        value="models/char_stylizer.pt", label="pix2pix checkpoint", info='Stylizer backend "neural"'
                    )
                    autoencoder_model = gr.Textbox(
                        value="models/char_autoencoder.pt",
                        label="Autoencoder checkpoint",
                        info='Stylizer backend "latent"',
                    )
                    alphabet_dir = gr.Textbox(
                        value="dataset/alphabet",
                        label="Alphabet dataset",
                        info='Pretty reference glyphs (backend "latent")',
                    )
                analyze_btn = gr.Button("Analyze page", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column():
                before = gr.Image(label="Before", interactive=False)
                after = gr.Image(label="After", interactive=False)

        inputs = [
            page_in,
            detector_backend,
            yolo_model,
            confidence,
            classifier_path,
            stylizer_backend,
            stylizer_model,
            autoencoder_model,
            alphabet_dir,
        ]
        analyze_btn.click(analyze_page, inputs=inputs, outputs=[state, before, status]).then(
            render_page, inputs=[state, alpha], outputs=[after]
        )
        page_in.upload(analyze_page, inputs=inputs, outputs=[state, before, status]).then(
            render_page, inputs=[state, alpha], outputs=[after]
        )
        alpha.change(render_page, inputs=[state, alpha], outputs=[after])
    return demo


def _to_bgr(image: np.ndarray) -> np.ndarray:
    """Convert an RGB array from Gradio into the BGR layout OpenCV expects."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a BGR pipeline image back into RGB for display."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def _main():
    logging.basicConfig(format="%(asctime)-15s %(levelname)s %(message)s", level=logging.INFO)
    create_app().launch()


if __name__ == "__main__":
    _main()
