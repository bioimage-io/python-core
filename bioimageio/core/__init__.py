"""
.. include:: ../../README.md
"""

from bioimageio.spec import (
    build_description,
    dump_description,
    load_dataset_description,
    load_description,
    load_description_and_validate_format_only,
    load_model_description,
    save_bioimageio_package,
    save_bioimageio_package_as_folder,
    save_bioimageio_yaml_only,
    validate_format,
)

from . import (
    axis,
    block_meta,
    cli,
    commands,
    common,
    digest_spec,
    io,
    model_adapters,
    prediction,
    proc_ops,
    proc_setup,
    sample,
    stat_calculators,
    stat_measures,
    tensor,
)
from ._prediction_pipeline import PredictionPipeline, create_prediction_pipeline
from ._resource_tests import (
    enable_determinism,
    load_description_and_test,
    test_description,
    test_model,
)
from ._settings import settings
from .axis import Axis, AxisId
from .backends import create_model_adapter
from .block_meta import BlockMeta
from .common import MemberId
from .prediction import predict, predict_many
from .sample import Sample
from .stat_calculators import compute_dataset_measures
from .stat_measures import Stat
from .tensor import Tensor
from .utils import VERSION
from .weight_converters import add_weights

__version__ = VERSION


# aliases
test_resource = test_description
"""alias of `test_description`"""
load_resource = load_description
"""alias of `load_description`"""
load_model = load_model_description
"""alias of `load_model_description`"""

__all__ = [
    "__version__",
    "add_weights",
    "axis",
    "Axis",
    "AxisId",
    "block_meta",
    "BlockMeta",
    "build_description",
    "cli",
    "commands",
    "common",
    "compute_dataset_measures",
    "create_model_adapter",
    "create_prediction_pipeline",
    "digest_spec",
    "dump_description",
    "enable_determinism",
    "io",
    "load_dataset_description",
    "load_description_and_test",
    "load_description_and_validate_format_only",
    "load_description",
    "load_model_description",
    "load_model",
    "load_resource",
    "MemberId",
    "model_adapters",
    "predict_many",
    "predict",
    "prediction",
    "PredictionPipeline",
    "proc_ops",
    "proc_setup",
    "sample",
    "Sample",
    "save_bioimageio_package_as_folder",
    "save_bioimageio_package",
    "save_bioimageio_yaml_only",
    "settings",
    "stat_calculators",
    "stat_measures",
    "Stat",
    "tensor",
    "Tensor",
    "test_description",
    "test_model",
    "test_resource",
    "validate_format",
]
