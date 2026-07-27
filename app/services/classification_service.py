import json
import logging
from typing import Optional

from app.config import get_settings

logger = logging.getLogger(__name__)

_model = None
_label_map: Optional[dict] = None
_load_attempted = False


def _load() -> None:
    """Lazily loads the persisted Keras classifier. TensorFlow is only imported once a
    trained model actually exists on disk, so a fresh checkout with no trained model yet
    doesn't pay the TF import cost on every request."""
    global _model, _label_map, _load_attempted
    if _load_attempted:
        return
    _load_attempted = True

    settings = get_settings()
    model_path = settings.classifier_model_dir / "classifier.keras"
    label_map_path = settings.classifier_model_dir / "label_map.json"
    if not model_path.exists() or not label_map_path.exists():
        logger.warning("No trained classifier found at %s — uploads will be left uncategorized.", model_path)
        return

    import tensorflow as tf

    _model = tf.keras.models.load_model(model_path)
    _label_map = json.loads(label_map_path.read_text())


async def predict_category(embedding: list[float]) -> Optional[tuple[str, float]]:
    _load()
    if _model is None or _label_map is None:
        return None

    import numpy as np

    x = np.asarray([embedding], dtype="float32")
    probs = _model.predict(x, verbose=0)[0]
    idx = int(probs.argmax())
    category = _label_map[str(idx)]
    confidence = float(probs[idx])
    return category, confidence


def is_model_available() -> bool:
    _load()
    return _model is not None
