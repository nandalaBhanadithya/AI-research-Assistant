"""Offline training entry point: python -m app.ml.train_classifier

Stages: (1) data preprocessing — fetch/cache arXiv abstracts per category, (2) feature
engineering — embed via the same local Ollama embedding model the RAG pipeline uses,
(3) model training, (4) model evaluation, (5) model persistence. Stage (6), the
prediction API, is served by app/services/classification_service.py at request time.
"""

import asyncio
import json
import logging

import numpy as np
from sklearn.model_selection import train_test_split

from app.config import get_settings
from app.core.constants import ARXIV_CATEGORY_MAP, DOCUMENT_CATEGORIES
from app.logging_config import configure_logging
from app.ml.datasets.arxiv_category_dataset import build_dataset
from app.ml.evaluate_classifier import evaluate
from app.ml.model_def import build_classifier
from app.services.embedding_service import embed_texts

logger = logging.getLogger(__name__)

SAMPLES_PER_CATEGORY = 400
EMBEDDING_BATCH_SIZE = 32


async def _embed_all(texts: list[str]) -> np.ndarray:
    vectors = await embed_texts(texts, batch_size=EMBEDDING_BATCH_SIZE)
    return np.array(vectors, dtype="float32")


def main() -> None:
    configure_logging("INFO")
    settings = get_settings()
    cache_dir = settings.data_dir / "classifier" / "raw"
    processed_dir = settings.data_dir / "classifier" / "processed"
    model_dir = settings.classifier_model_dir

    # --- Stage 1: Data preprocessing ---
    logger.info("Building dataset from arXiv (%d samples/category)...", SAMPLES_PER_CATEGORY)
    raw_dataset = build_dataset(ARXIV_CATEGORY_MAP, SAMPLES_PER_CATEGORY, cache_dir)

    label_names = DOCUMENT_CATEGORIES
    label_to_index = {name: i for i, name in enumerate(label_names)}
    texts: list[str] = []
    labels: list[int] = []
    for label, samples in raw_dataset.items():
        for sample in samples:
            texts.append(f"{sample['title']}. {sample['abstract']}")
            labels.append(label_to_index[label])
    logger.info("Total labeled samples: %d", len(texts))

    # --- Stage 2: Feature engineering ---
    processed_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = processed_dir / "embeddings.npy"
    labels_path = processed_dir / "labels.npy"

    cached_embeddings_valid = False
    if embeddings_path.exists() and labels_path.exists():
        x = np.load(embeddings_path)
        y = np.load(labels_path)
        # Guards against a stale cache silently going out of sync with the current
        # dataset (e.g. a category that failed to fetch before now succeeding, changing
        # len(texts)) — regenerate rather than train on a mismatched embeddings cache.
        cached_embeddings_valid = len(x) == len(texts) == len(y)
        if not cached_embeddings_valid:
            logger.warning(
                "Cached embeddings (%d) don't match current dataset size (%d) — regenerating.", len(x), len(texts)
            )

    if cached_embeddings_valid:
        logger.info("Loading cached embeddings from %s", embeddings_path)
    else:
        logger.info("Generating embeddings via Ollama (%s)...", settings.ollama_embedding_model)
        x = asyncio.run(_embed_all(texts))
        y = np.array(labels, dtype="int64")
        np.save(embeddings_path, x)
        np.save(labels_path, y)

    x_train, x_temp, y_train, y_temp = train_test_split(x, y, test_size=0.3, stratify=y, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    logger.info("Split sizes -> train: %d, val: %d, test: %d", len(x_train), len(x_val), len(x_test))

    # --- Stage 3: Model training ---
    from tensorflow import keras

    model = build_classifier(input_dim=x.shape[1], num_classes=len(label_names))
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)]
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=50, batch_size=32, callbacks=callbacks, verbose=2)

    # --- Stage 4: Model evaluation ---
    results = evaluate(model, x_test, y_test, label_names, output_dir=model_dir)
    logger.info(
        "Test accuracy: %.4f (majority-class baseline: %.4f)",
        results["accuracy"],
        results["majority_class_baseline_accuracy"],
    )

    # --- Stage 5: Model persistence ---
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save(model_dir / "classifier.keras")
    (model_dir / "label_map.json").write_text(
        json.dumps({str(i): name for i, name in enumerate(label_names)}, indent=2)
    )
    (model_dir / "metadata.json").write_text(
        json.dumps(
            {
                "embedding_model": settings.ollama_embedding_model,
                "embedding_dims": int(x.shape[1]),
                "num_samples": len(texts),
                "samples_per_category": SAMPLES_PER_CATEGORY,
                "test_accuracy": results["accuracy"],
            },
            indent=2,
        )
    )
    logger.info("Model persisted to %s", model_dir)


if __name__ == "__main__":
    main()
