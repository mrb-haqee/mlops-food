"""
Author: Rafli
Date: 10/04/2024

The code to run the pipeline
"""

import os
from absl import logging
from tfx.orchestration import metadata, pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner

PIPELINE_NAME = "Rafli_pipeline"

# Pipeline inputs
DATA_ROOT = "data"
TRANSFORM_MODULE_FILE = "modules/allergen_food_transform.py"
TUNER_MODULE_FILE = "modules/allergen_food_tuner.py"
TRAINER_MODULE_FILE = "modules/allergen_food_trainer.py"

# Pipeline outputs
OUTPUT_BASE = "output"
SERVING_MODEL_DIR = os.path.join(OUTPUT_BASE, 'serving_model')
PIPELINE_ROOT = os.path.join(OUTPUT_BASE, PIPELINE_NAME)
METADATA_PATH = os.path.join(PIPELINE_ROOT, "metadata.sqlite")

# Configuration
config = {
    "data_dir": DATA_ROOT,
    "transform_module": TRANSFORM_MODULE_FILE,
    "tuning_module": TUNER_MODULE_FILE,
    "training_module": TRAINER_MODULE_FILE,
    "training_steps": 200,
    "eval_steps": 50,
    "serving_model_dir": SERVING_MODEL_DIR
}


def init_local_pipeline(
    components_for_pipeline, pipeline_root_directory: str
) -> pipeline.Pipeline:
    """Function to run a local pipeline

    Args:
        components_for_pipeline (list[TFX Component]): TFX components to be used in the pipeline.
        root_directory (str): The root directory for the pipeline outputs.

    Returns:
        pipeline.Pipeline: The constructed TFX pipeline.
    """
    logging.info(f"Pipeline root set to: {pipeline_root_directory}")

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=pipeline_root_directory,
        components=components_for_pipeline,
        enable_cache=True,
        metadata_connection_config=metadata.sqlite_metadata_connection_config(
            METADATA_PATH
        ),
        beam_pipeline_args=[
        "--direct_running_mode=multi_processing",
        "--direct_num_workers=0"
    ]
    )


if __name__ == "__main__":
    logging.set_verbosity(logging.INFO)

    from modules.components import init_components

    configured_components = init_components(config)

    local_pipeline = init_local_pipeline(configured_components, PIPELINE_ROOT)
    BeamDagRunner().run(pipeline=local_pipeline)
