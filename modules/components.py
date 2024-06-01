"""
Author: Rafli
Date: 31/05/2024

Initiate tfx pipeline components
"""

import os
from typing import Dict, Tuple
import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    StatisticsGen,
    SchemaGen,
    ExampleValidator,
    Transform,
    Tuner,
    Trainer,
    Evaluator,
    Pusher,
    InfraValidator,
)
from tfx.proto import example_gen_pb2, trainer_pb2, pusher_pb2
from tfx.types import Channel
from tfx.dsl.components.common.resolver import Resolver
from tfx.types.standard_artifacts import Model, ModelBlessing
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.v1.proto import ServingSpec, TensorFlowServing, LocalDockerConfig


def init_components(config: Dict) -> Tuple:
    """Initializes components for a TFX pipeline.

    Args:
        config (Dict): A dictionary containing configuration parameters.

    Returns:
        Tuple: A tuple containing initialized components for the TFX pipeline.
    """
    example_gen = component_example_gen(config["data_dir"])
    statistics_gen = component_statistics_gen(example_gen)
    schema_gen = component_schema_gen(statistics_gen)
    example_validator = component_example_validator(statistics_gen, schema_gen)
    transform = component_transform(example_gen, schema_gen, config["transform_module"])
    tuner = component_tuner(transform, schema_gen, config)
    trainer = component_trainer(transform, schema_gen, tuner, config)
    model_resolver = component_model_resolver()
    evaluator = component_evaluator(transform, trainer, model_resolver)
    pusher = component_pusher(trainer, evaluator, config["serving_model_dir"])

    components = (
        example_gen,
        statistics_gen,
        schema_gen,
        example_validator,
        transform,
        tuner,
        trainer,
        model_resolver,
        evaluator,
        pusher,
    )

    return components


def component_example_gen(data_dir: str) -> CsvExampleGen:
    """Creates a CsvExampleGen component for a TFX pipeline.

    Args:
        data_dir (str): Directory containing the input CSV files.

    Returns:
        CsvExampleGen: Configured to read from data_dir and split 
        data into train (80%) and eval (20%).
    """
    output_config = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(
            splits=[
                example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
                example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
            ]
        )
    )
    return CsvExampleGen(input_base=data_dir, output_config=output_config)


def component_statistics_gen(example_gen: CsvExampleGen) -> StatisticsGen:
    """Creates a StatisticsGen component for a TFX pipeline.

    Args:
        example_gen (ExampleGen): The ExampleGen component producing examples.

    Returns:
        StatisticsGen: A component that computes statistics on the input data.
    """
    return StatisticsGen(examples=example_gen.outputs["examples"])


def component_schema_gen(statistics_gen: StatisticsGen) -> SchemaGen:
    """Creates a SchemaGen component for a TFX pipeline.

    Args:
        statistics_gen (StatisticsGen): The StatisticsGen component producing statistics.

    Returns:
        SchemaGen: A component that generates a schema based on the computed statistics.
    """
    return SchemaGen(statistics=statistics_gen.outputs["statistics"])


def component_example_validator(
    statistics_gen: StatisticsGen, schema_gen: SchemaGen
) -> ExampleValidator:
    """Creates an ExampleValidator component for a TFX pipeline.

    Args:
        statistics_gen (StatisticsGen): The StatisticsGen component producing statistics.
        schema_gen (SchemaGen): The SchemaGen component producing schema.

    Returns:
        ExampleValidator: A component that validates examples against a schema.
    """
    return ExampleValidator(
        statistics=statistics_gen.outputs["statistics"],
        schema=schema_gen.outputs["schema"],
    )


def component_transform(
    example_gen: CsvExampleGen, schema_gen: SchemaGen, transform_module: str
) -> Transform:
    """Creates a Transform component for a TFX pipeline.

    Args:
        example_gen (ExampleGen): The ExampleGen component producing examples.
        schema_gen (SchemaGen): The SchemaGen component producing schema.
        transform_module (str): Path to the module containing the transformation logic.

    Returns:
        Transform: A component that performs data transformations based on the provided module.
    """
    return Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.abspath(transform_module),
    )


def component_tuner(transform: Transform, schema_gen: SchemaGen, config: dict) -> Tuner:
    """Creates a Tuner component for a TFX pipeline.

    Args:
        transform (Transform): The Transform component producing transformed examples.
        schema_gen (SchemaGen): The SchemaGen component producing schema.
        config (dict): Configuration parameters including tuning module path, 
                       training steps, and eval steps.

    Returns:
        Tuner: A component that tunes hyperparameters for the training process.
    """
    return Tuner(
        module_file=os.path.abspath(config["tuning_module"]),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(
            splits=["train"], num_steps=config["training_steps"]
        ),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=config["eval_steps"]),
    )


def component_trainer(
    transform: Transform, schema_gen: SchemaGen, tuner: Tuner, config: dict
) -> Trainer:
    """Creates a Trainer component for a TFX pipeline.

    Args:
        transform (Transform): The Transform component producing transformed examples.
        schema_gen (SchemaGen): The SchemaGen component producing schema.
        tuner (Tuner): The Tuner component producing best hyperparameters.
        config (dict): Configuration parameters including training module path, 
                       training steps, and eval steps.

    Returns:
        Trainer: A component that performs model training 
                 based on the provided module and hyperparameters.
    """
    return Trainer(
        module_file=os.path.abspath(config["training_module"]),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=tuner.outputs["best_hyperparameters"],
        train_args=trainer_pb2.TrainArgs(
            splits=["train"], num_steps=config["training_steps"]
        ),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=config["eval_steps"]),
    )


def component_model_resolver() -> Resolver:
    """Creates a ModelResolver component for a TFX pipeline.

    Returns:
        Resolver: A component that resolves the latest blessed model.
    """
    return Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("Latest_blessed_model_resolver")


def component_evaluator(
    transform: Transform, trainer: Trainer, model_resolver: Resolver
) -> Evaluator:
    """Creates an Evaluator component for a TFX pipeline.

    Args:
        transform (Transform): The Transform component producing transformed examples.
        trainer (Trainer): The Trainer component producing trained model.
        model_resolver (Resolver): The Resolver component resolving the latest blessed model.

    Returns:
        Evaluator: A component that evaluates the trained model.
    """
    eval_config = tfma.EvalConfig(
        model_specs=[tfma.ModelSpec(label_key="is_allergen_xf")],
        slicing_specs=[
            tfma.SlicingSpec(),
            tfma.SlicingSpec(
                feature_keys=["Main_Ingredient", "Fat_Oil", "Seasoning", "Allergens"]
            ),
        ],
        metrics_specs=[
            tfma.MetricsSpec(
                metrics=[
                    tfma.MetricConfig(class_name="AUC"),
                    tfma.MetricConfig(class_name="Precision", config='{"top_k": 1}'),
                    tfma.MetricConfig(class_name="Recall", config='{"top_k": 1}'),
                    tfma.MetricConfig(class_name="ExampleCount"),
                    tfma.MetricConfig(
                        class_name="BinaryAccuracy",
                        threshold=tfma.MetricThreshold(
                            value_threshold=tfma.GenericValueThreshold(
                                lower_bound={"value": 0.5}
                            ),
                            change_threshold=tfma.GenericChangeThreshold(
                                direction=tfma.MetricDirection.HIGHER_IS_BETTER,
                                absolute={"value": 0.0001},
                            ),
                        ),
                    ),
                ]
            )
        ],
    )
    return Evaluator(
        examples=transform.outputs["transformed_examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )


def component_pusher(
    trainer: Trainer, evaluator: Evaluator, serving_model_dir: str
) -> Pusher:
    """Creates a Pusher component for a TFX pipeline.

    Args:
        trainer (Trainer): The Trainer component producing trained model.
        evaluator (Evaluator): The Evaluator component producing evaluation results.
        serving_model_dir (str): The directory where the serving model will be stored.

    Returns:
        Pusher: A component that pushes the trained model for serving.
    """
    infra_validator = InfraValidator(
        model=trainer.outputs["model"],
        serving_spec=ServingSpec(
            tensorflow_serving=TensorFlowServing(tags=["latest"]),
            local_docker=LocalDockerConfig(),
        ),
    )

    return Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        infra_blessing=infra_validator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )
