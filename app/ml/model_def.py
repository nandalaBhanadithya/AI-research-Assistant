def build_classifier(input_dim: int, num_classes: int):
    """Small feedforward classifier over document embeddings (see train_classifier.py
    for why embeddings are used as features instead of a separate TF-IDF/text
    vectorization pipeline)."""
    from tensorflow import keras

    model = keras.Sequential(
        [
            keras.layers.Input(shape=(input_dim,)),
            keras.layers.Dense(256, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model
