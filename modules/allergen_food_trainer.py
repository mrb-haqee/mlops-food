"""
Author: Rafli
Date: 31/05/2024

Module Trainer File
"""

import os
from typing import Dict, Optional, Callable

import tensorflow as tf
import tensorflow_transform as tft
from kerastuner import HyperParameters
from tfx.components.trainer.fn_args_utils import FnArgs

from allergen_food_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)

from allergen_food_tuner import input_fn


def get_model(
    hp: Optional[HyperParameters] = None, show_summary: bool = True
) -> tf.keras.models.Model:
    """Builds and compiles a Keras model with optional hyperparameters.

    Args:
        hp (Optional[HyperParameters]): Hyperparameters for tuning the model architecture.
        show_summary (bool, optional): Whether to display the model summary.

    Returns:
        tf.keras.models.Model: A compiled Keras model.
    """

    # One-hot categorical features
    input_features = []
    for key, dim in CATEGORICAL_FEATURES.items():
        input_features.append(
            tf.keras.Input(shape=(dim + 1,), name=transformed_name(key))
        )

    for feature in NUMERICAL_FEATURES:
        input_features.append(
            tf.keras.Input(shape=(1,), name=transformed_name(feature))
        )

    concatenate = tf.keras.layers.concatenate(input_features)
    deep = tf.keras.layers.Dense(256, activation="relu")(concatenate)
    num_hidden_layers = (
        hp.get("hidden_layers") if hp and "hidden_layers" in hp else 1
    )  # Tuning the number of hidden layers
    for _ in range(num_hidden_layers):
        deep = tf.keras.layers.Dense(units=32, activation="relu")(deep)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)
    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.get("learning_rate") if hp else 0.001),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    if show_summary:
        model.summary()

    return model


def get_serve_tf_examples_fn(
    model: tf.keras.Model, tf_transform_output: tft.TFTransformOutput
) -> Callable:
    """Creates a function that serves the model using serialized tf.Example input.

    Args:
        model (tf.keras.Model): The trained Keras model.
        tf_transform_output (TFTransformOutput): The output from the TFTransform step.

    Returns:
        Callable: A function that serves the model for TensorFlow Serving.
    """
    model.tft_layer = tf_transform_output.transform_features_layer()

    @tf.function
    def serve_tf_examples_fn(serialized_tf_examples: tf.Tensor) -> Dict[str, tf.Tensor]:
        """Returns the output to be used in the serving signature.

        Args:
            serialized_tf_examples (tf.Tensor): A tensor of serialized tf.Example.

        Returns:
            Dict[str, tf.Tensor]: A dictionary containing the model outputs.
        """
        feature_spec = tf_transform_output.raw_feature_spec()
        feature_spec.pop(LABEL_KEY)
        parsed_features = tf.io.parse_example(serialized_tf_examples, feature_spec)

        transformed_features = model.tft_layer(parsed_features)

        outputs = model(transformed_features)
        return {"outputs": outputs}

    return serve_tf_examples_fn


# TFX Trainer will call this function.
def run_fn(fn_args: FnArgs):
    """Trains and saves a TensorFlow model.

    Args:
        fn_args (FnArgs): Arguments passed to the function, including directories
                          and file paths for training, evaluation, and serving.
    """
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_output)

    train_dataset = input_fn(fn_args.train_files, tf_transform_output, 64)
    eval_dataset = input_fn(fn_args.eval_files, tf_transform_output, 64)

    hp = fn_args.hyperparameters.get("values", {}) if fn_args.hyperparameters else {}

    model = get_model(hp)

    log_dir = os.path.join(os.path.dirname(fn_args.serving_model_dir), "logs")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, update_freq="batch"
    )

    model.fit(
        train_dataset,
        steps_per_epoch=fn_args.train_steps,
        validation_data=eval_dataset,
        validation_steps=fn_args.eval_steps,
        callbacks=[tensorboard_callback],
        epochs=3,
    )

    signatures = {
        "serving_default": get_serve_tf_examples_fn(
            model, tf_transform_output
        ).get_concrete_function(
            tf.TensorSpec(shape=[None], dtype=tf.string, name="examples")
        ),
    }
    model.save(fn_args.serving_model_dir, save_format="tf", signatures=signatures)
