"""Shared factories that assemble pipeline stages from plain options.

Both `improve_document.py` (CLI) and `web_ui.py` (Gradio app) build the same
stages from user-facing backend names and model paths; this module keeps a
single source of truth for that wiring.
"""


def build_detector(backend: str = "mser", model_path: str = "models/character_detector.pt", confidence: float = 0.25):
    """Build the character detection stage.

    :param backend: detection backend name, "mser" or "yolo"
    :type backend: str
    :param model_path: path to the trained YOLO weights (backend "yolo")
    :type model_path: str
    :param confidence: minimum detection confidence (backend "yolo")
    :type confidence: float
    :return: the configured detector
    :rtype: Union[CharacterDetector, YoloCharacterDetector]
    """
    if backend == "yolo":
        from call_ai_grapher.pipeline.yolo_detector import YoloCharacterDetector

        return YoloCharacterDetector(model_path, confidence=confidence)
    from call_ai_grapher.pipeline.detector import CharacterDetector

    return CharacterDetector()


def build_classifier(model_path=None):
    """Build the character classification stage.

    :param model_path: checkpoint written by train_classifier.py, or None to skip labeling
    :type model_path: Optional[str]
    :return: the configured classifier, or None when no checkpoint is given
    :rtype: Optional[CharClassifier]
    """
    if not model_path:
        return None
    from call_ai_grapher.pipeline.classifier import CharClassifier

    return CharClassifier(model_path)


def build_stylizer(
    backend: str = "baseline",
    stylizer_model: str = "models/char_stylizer.pt",
    autoencoder_model: str = "models/char_autoencoder.pt",
    alphabet_dir: str = "dataset/alphabet",
    classifier_model=None,
):
    """Build the stylization stage.

    :param backend: stylization backend name, "baseline", "neural" or "latent"
    :type backend: str
    :param stylizer_model: checkpoint written by train_stylizer.py (backend "neural")
    :type stylizer_model: str
    :param autoencoder_model: checkpoint written by train_autoencoder.py (backend "latent")
    :type autoencoder_model: str
    :param alphabet_dir: alphabet dataset with the pretty reference glyphs (backend "latent")
    :type alphabet_dir: str
    :param classifier_model: classifier checkpoint; required by the "latent" backend so each
        character finds its reference glyph
    :type classifier_model: Optional[str]
    :return: the configured stylizer
    :rtype: Union[Stylizer, NeuralStylizer, LatentStylizer]
    """
    if backend == "neural":
        from call_ai_grapher.pipeline.stylizer import NeuralStylizer

        return NeuralStylizer(stylizer_model)
    if backend == "latent":
        from call_ai_grapher.pipeline.stylizer import LatentStylizer

        if not classifier_model:
            raise ValueError(
                "The latent stylizer requires the character classifier so each character finds its reference glyph"
            )
        return LatentStylizer(autoencoder_model, alphabet_dir=alphabet_dir)
    from call_ai_grapher.pipeline.stylizer import Stylizer

    return Stylizer()
