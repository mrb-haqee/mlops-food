"""
Author: Rafli
Date: 31/05/2024

Module Transform File
"""

from typing import Dict

import tensorflow as tf
import tensorflow_transform as tft

CATEGORICAL_FEATURES = {
    "Main_Ingredient": 101,
    "Sweetener": 10,
    "Fat_Oil": 36,
    "Seasoning": 186,
    "Allergens": 40,
}
NUMERICAL_FEATURES = ["Price", "Rating"]
LABEL_KEY = "is_allergen"


def transformed_name(key: str) -> str:
    """Appends the suffix '_xf' to the given key.

    Args:
        key (str): The original feature key.

    Returns:
        str: The transformed feature key with '_xf' suffix.
    """
    return key + "_xf"


def convert_num_to_one_hot(label_tensor: tf.Tensor, num_labels: int = 2) -> tf.Tensor:
    """Converts a numerical label tensor to a one-hot encoded tensor.

    Args:
        label_tensor (tf.Tensor): The tensor containing numerical labels.
        num_labels (int, optional): The number of label classes. Defaults to 2.

    Returns:
        tf.Tensor: The one-hot encoded tensor.
    """
    one_hot_tensor = tf.one_hot(label_tensor, num_labels)
    return tf.reshape(one_hot_tensor, [-1, num_labels])


def preprocessing_fn(inputs: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Preprocesses the input features for training.

    Args:
        inputs (Dict[str, tf.Tensor]): A dictionary of input feature tensors.

    Returns:
        Dict[str, tf.Tensor]: A dictionary of transformed feature tensors.
    """
    outputs = {}

    for key, dim in CATEGORICAL_FEATURES.items():
        int_value = tft.compute_and_apply_vocabulary(inputs[key], top_k=dim + 1)
        outputs[transformed_name(key)] = convert_num_to_one_hot(
            int_value, num_labels=dim + 1
        )

    for feature in NUMERICAL_FEATURES:
        outputs[transformed_name(feature)] = tft.scale_to_0_1(inputs[feature])

    outputs[transformed_name(LABEL_KEY)] = tf.cast(inputs[LABEL_KEY], tf.int64)

    return outputs
