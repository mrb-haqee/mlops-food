"""
Author: Rafli
Date: 31/05/2024

Module Tuner File
"""

from typing import NamedTuple, Dict, Any
from allergen_food_transform import (
    CATEGORICAL_FEATURES,
    LABEL_KEY,
    NUMERICAL_FEATURES,
    transformed_name,
)
from kerastuner import RandomSearch, HyperParameters
from kerastuner.engine import base_tuner
import tensorflow as tf
from tfx.components.trainer.fn_args_utils import FnArgs
import tensorflow_transform as tft

TunerFnResult = NamedTuple(
    "TunerFnResult", [("tuner", base_tuner.BaseTuner), ("fit_kwargs", Dict[str, Any])]
)


def get_model(hp: HyperParameters) -> tf.keras.models.Model:
    """Builds and compiles a Keras model based on hyperparameters.

    Args:
        hp (HyperParameters): Hyperparameters for tuning the model architecture.

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
    # Tuning the number of hidden layers
    for _ in range(hp.Int("num_hidden_layers", 1, 5)):
        deep = tf.keras.layers.Dense(
            units=hp.Choice("units", values=[32, 64, 128, 256]), activation="relu"
        )(deep)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid")(deep)

    model = tf.keras.models.Model(inputs=input_features, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            hp.Choice("learning_rate", values=[0.001, 0.01, 0.1])
        ),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    return model


def gzip_reader_fn(filenames: tf.Tensor) -> tf.data.TFRecordDataset:
    """Reads TFRecord files with GZIP compression.

    Args:
        filenames (tf.Tensor): A tensor of filenames to read.

    Returns:
        tf.data.TFRecordDataset: A dataset of TFRecords read from the given filenames.
    """
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(
    file_pattern: str, tf_transform_output: tft.TFTransformOutput, batch_size: int = 64
) -> tf.data.Dataset:
    """Creates an input function for training and evaluation.

    Args:
        file_pattern (str): File pattern for the input data files.
        tf_transform_output (tf_transform.TFTransformOutput): A TFTransformOutput object from TFT.
        batch_size (int, optional): Batch size for training. Defaults to 64.

    Returns:
        tf.data.Dataset: A dataset for training or evaluation.
    """
    transformed_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transformed_feature_spec,
        reader=gzip_reader_fn,
        label_key=transformed_name(LABEL_KEY),
    )

    return dataset


def tuner_fn(fn_args: FnArgs) -> TunerFnResult:
    """Builds and returns a KerasTuner RandomSearch tuner for hyperparameter tuning.

    Args:
        fn_args (FnArgs): Arguments passed to the function, 
                          including directories and file paths for training.

    Returns:
        TunerFnResult: An object containing the tuner instance and the fit parameters.
    """
    tuner = RandomSearch(
        get_model,
        objective="val_binary_accuracy",
        max_trials=10,
        executions_per_trial=1,
        directory=fn_args.working_dir,
        project_name="random_search",
    )

    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    train_set = input_fn(fn_args.train_files, tf_transform_output, 64)
    val_set = input_fn(fn_args.eval_files, tf_transform_output, 64)

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_set,
            "validation_data": val_set,
            "epochs": 3,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
