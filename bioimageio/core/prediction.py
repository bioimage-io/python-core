import collections.abc
from pathlib import Path
from typing import (
    Hashable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    Union,
)

from loguru import logger
from tqdm import tqdm

from bioimageio.spec import load_description
from bioimageio.spec.common import PermissiveFileSource
from bioimageio.spec.model import v0_4, v0_5

from ._prediction_pipeline import PredictionPipeline, create_prediction_pipeline
from .axis import AxisId
from .common import BlocksizeParameter, MemberId, PerMember
from .digest_spec import TensorSource, create_sample_for_model, get_member_id
from .io import save_sample
from .sample import Sample


def predict(
    *,
    model: Union[
        PermissiveFileSource, v0_4.ModelDescr, v0_5.ModelDescr, PredictionPipeline
    ],
    inputs: Union[Sample, PerMember[TensorSource], TensorSource],
    sample_id: Hashable = "sample",
    blocksize_parameter: Optional[BlocksizeParameter] = None,
    input_block_shape: Optional[Mapping[MemberId, Mapping[AxisId, int]]] = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    save_output_path: Optional[Union[Path, str]] = None,
) -> Sample:
    """Run prediction for a single set of input(s) with a bioimage.io model

    Args:
        model: Model to predict with.
            May be given as RDF source, model description or prediction pipeline.
        inputs: the input sample or the named input(s) for this model as a dictionary
        sample_id: the sample id.
            The **sample_id** is used to format **save_output_path**
            and to distinguish sample specific log messages.
        blocksize_parameter: (optional) Tile the input into blocks parametrized by
            **blocksize_parameter** according to any parametrized axis sizes defined
            by the **model**.
            See `bioimageio.spec.model.v0_5.ParameterizedSize` for details.
            Note: For a predetermined, fixed block shape use **input_block_shape**.
        input_block_shape: (optional) Tile the input sample tensors into blocks.
            Note: Use **blocksize_parameter** for a parameterized block shape to
                run prediction independent of the exact block shape.
        skip_preprocessing: Flag to skip the model's preprocessing.
        skip_postprocessing: Flag to skip the model's postprocessing.
        save_output_path: A path with to save the output to. M
            Must contain:
            - `{output_id}` (or `{member_id}`) if the model has multiple output tensors
            May contain:
            - `{sample_id}` to avoid overwriting recurrent calls
    """
    if isinstance(model, PredictionPipeline):
        pp = model
        model = pp.model_description
    else:
        if not isinstance(model, (v0_4.ModelDescr, v0_5.ModelDescr)):
            loaded = load_description(model)
            if not isinstance(loaded, (v0_4.ModelDescr, v0_5.ModelDescr)):
                raise ValueError(f"expected model description, but got {loaded}")
            model = loaded

        pp = create_prediction_pipeline(model)

    if save_output_path is not None:
        if (
            "{output_id}" not in str(save_output_path)
            and "{member_id}" not in str(save_output_path)
            and len(model.outputs) > 1
        ):
            raise ValueError(
                f"Missing `{{output_id}}` in save_output_path={save_output_path} to "
                + "distinguish model outputs "
                + str([get_member_id(d) for d in model.outputs])
            )

    if isinstance(inputs, Sample):
        sample = inputs
    else:
        sample = create_sample_for_model(
            pp.model_description, inputs=inputs, sample_id=sample_id
        )

    if input_block_shape is not None:
        if blocksize_parameter is not None:
            logger.warning(
                "ignoring blocksize_parameter={} in favor of input_block_shape={}",
                blocksize_parameter,
                input_block_shape,
            )

        output = pp.predict_sample_with_fixed_blocking(
            sample,
            input_block_shape=input_block_shape,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
        )
    elif blocksize_parameter is not None:
        output = pp.predict_sample_with_blocking(
            sample,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
            ns=blocksize_parameter,
        )
    else:
        output = pp.predict_sample_without_blocking(
            sample,
            skip_preprocessing=skip_preprocessing,
            skip_postprocessing=skip_postprocessing,
        )
    if save_output_path:
        save_sample(save_output_path, output)

    return output


def predict_many(
    *,
    model: Union[
        PermissiveFileSource, v0_4.ModelDescr, v0_5.ModelDescr, PredictionPipeline
    ],
    inputs: Union[Iterable[PerMember[TensorSource]], Iterable[TensorSource]],
    sample_id: str = "sample{i:03}",
    blocksize_parameter: Optional[
        Union[
            v0_5.ParameterizedSize_N,
            Mapping[Tuple[MemberId, AxisId], v0_5.ParameterizedSize_N],
        ]
    ] = None,
    skip_preprocessing: bool = False,
    skip_postprocessing: bool = False,
    save_output_path: Optional[Union[Path, str]] = None,
) -> Iterator[Sample]:
    """Run prediction for a multiple sets of inputs with a bioimage.io model

    Args:
        model: Model to predict with.
            May be given as RDF source, model description or prediction pipeline.
        inputs: An iterable of the named input(s) for this model as a dictionary.
        sample_id: The sample id.
            note: `{i}` will be formatted as the i-th sample.
            If `{i}` (or `{i:`) is not present and `inputs` is not an iterable `{i:03}`
            is appended.
        blocksize_parameter: (optional) Tile the input into blocks parametrized by
            blocksize according to any parametrized axis sizes defined in the model RDF.
        skip_preprocessing: Flag to skip the model's preprocessing.
        skip_postprocessing: Flag to skip the model's postprocessing.
        save_output_path: A path to save the output to.
            Must contain:
            - `{sample_id}` to differentiate predicted samples
            - `{output_id}` (or `{member_id}`) if the model has multiple outputs
    """
    if save_output_path is not None and "{sample_id}" not in str(save_output_path):
        raise ValueError(
            f"Missing `{{sample_id}}` in save_output_path={save_output_path}"
            + " to differentiate predicted samples."
        )

    if isinstance(model, PredictionPipeline):
        pp = model
    else:
        if not isinstance(model, (v0_4.ModelDescr, v0_5.ModelDescr)):
            loaded = load_description(model)
            if not isinstance(loaded, (v0_4.ModelDescr, v0_5.ModelDescr)):
                raise ValueError(f"expected model description, but got {loaded}")
            model = loaded

        pp = create_prediction_pipeline(model)

    if not isinstance(inputs, collections.abc.Mapping):
        if "{i}" not in sample_id and "{i:" not in sample_id:
            sample_id += "{i:03}"

        total = len(inputs) if isinstance(inputs, collections.abc.Sized) else None

        for i, ipts in tqdm(enumerate(inputs), total=total):
            yield predict(
                model=pp,
                inputs=ipts,
                sample_id=sample_id.format(i=i),
                blocksize_parameter=blocksize_parameter,
                skip_preprocessing=skip_preprocessing,
                skip_postprocessing=skip_postprocessing,
                save_output_path=save_output_path,
            )
