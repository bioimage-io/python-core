import traceback
from typing import Optional

from loguru import logger
from pydantic import DirectoryPath

from bioimageio.spec import (
    InvalidDescr,
    load_model_description,
    save_bioimageio_package_as_folder,
)
from bioimageio.spec.model.v0_5 import ModelDescr, WeightsFormat

from .._resource_tests import test_model


def add_weights(
    model_descr: ModelDescr,
    *,
    output_path: DirectoryPath,
    source_format: Optional[WeightsFormat] = None,
    target_format: Optional[WeightsFormat] = None,
    verbose: bool = False,
) -> Optional[ModelDescr]:
    """Convert model weights to other formats and add them to the model description

    Args:
        output_path: Path to save updated model package to.
        source_format: convert from a specific weights format.
                       Default: choose automatically from any available.
        target_format: convert to a specific weights format.
                       Default: attempt to convert to any missing format.
        devices: Devices that may be used during conversion.
        verbose: log more (error) output

    Returns:
        - An updated model description if any converted weights were added.
        - `None` if no conversion was possible.
    """
    if not isinstance(model_descr, ModelDescr):
        if model_descr.type == "model" and not isinstance(model_descr, InvalidDescr):
            raise TypeError(
                f"Model format {model_descr.format} is not supported, please update"
                + f" model to format {ModelDescr.implemented_format_version} first."
            )

        raise TypeError(type(model_descr))

    # save model to local folder
    output_path = save_bioimageio_package_as_folder(
        model_descr, output_path=output_path
    )
    # reload from local folder to make sure we do not edit the given model
    _model_descr = load_model_description(output_path, perform_io_checks=False)
    assert isinstance(_model_descr, ModelDescr)
    model_descr = _model_descr
    del _model_descr

    if source_format is None:
        available = set(model_descr.weights.available_formats)
    else:
        available = {source_format}

    if target_format is None:
        missing = set(model_descr.weights.missing_formats)
    else:
        missing = {target_format}

    originally_missing = set(missing)

    if "pytorch_state_dict" in available and "torchscript" in missing:
        logger.info(
            "Attempting to convert 'pytorch_state_dict' weights to 'torchscript'."
        )
        from .pytorch_to_torchscript import convert

        try:
            torchscript_weights_path = output_path / "weights_torchscript.pt"
            model_descr.weights.torchscript = convert(
                model_descr,
                output_path=torchscript_weights_path,
                use_tracing=False,
            )
        except Exception as e:
            if verbose:
                traceback.print_exception(e)

            logger.error(e)
        else:
            available.add("torchscript")
            missing.discard("torchscript")

    if "pytorch_state_dict" in available and "torchscript" in missing:
        logger.info(
            "Attempting to convert 'pytorch_state_dict' weights to 'torchscript' by tracing."
        )
        from .pytorch_to_torchscript import convert

        try:
            torchscript_weights_path = output_path / "weights_torchscript_traced.pt"

            model_descr.weights.torchscript = convert(
                model_descr,
                output_path=torchscript_weights_path,
                use_tracing=True,
            )
        except Exception as e:
            if verbose:
                traceback.print_exception(e)

            logger.error(e)
        else:
            available.add("torchscript")
            missing.discard("torchscript")

    if "torchscript" in available and "onnx" in missing:
        logger.info("Attempting to convert 'torchscript' weights to 'onnx'.")
        from .torchscript_to_onnx import convert

        try:
            onnx_weights_path = output_path / "weights.onnx"
            model_descr.weights.onnx = convert(
                model_descr,
                output_path=onnx_weights_path,
            )
        except Exception as e:
            if verbose:
                traceback.print_exception(e)

            logger.error(e)
        else:
            available.add("onnx")
            missing.discard("onnx")

    if "pytorch_state_dict" in available and "onnx" in missing:
        logger.info("Attempting to convert 'pytorch_state_dict' weights to 'onnx'.")
        from .pytorch_to_onnx import convert

        try:
            onnx_weights_path = output_path / "weights.onnx"

            model_descr.weights.onnx = convert(
                model_descr,
                output_path=onnx_weights_path,
                verbose=verbose,
            )
        except Exception as e:
            if verbose:
                traceback.print_exception(e)

            logger.error(e)
        else:
            available.add("onnx")
            missing.discard("onnx")

    if missing:
        logger.warning(
            f"Converting from any of the available weights formats {available} to any"
            + f" of {missing} failed or is not yet implemented. Please create an issue"
            + " at https://github.com/bioimage-io/core-bioimage-io-python/issues/new/choose"
            + " if you would like bioimageio.core to support a particular conversion."
        )

    if originally_missing == missing:
        logger.warning("failed to add any converted weights")
        return None
    else:
        logger.info("added weights formats {}", originally_missing - missing)
        # resave model with updated rdf.yaml
        _ = save_bioimageio_package_as_folder(model_descr, output_path=output_path)
        _ = test_model(model_descr)
        return model_descr
