"""These functions implement the logic of the bioimageio command line interface
defined in `bioimageio.core.cli`."""

from pathlib import Path
from typing import Optional, Sequence, Union

from typing_extensions import Literal

from bioimageio.core.common import SupportedWeightsFormat
from bioimageio.spec import (
    InvalidDescr,
    ResourceDescr,
    save_bioimageio_package,
    save_bioimageio_package_as_folder,
)
from bioimageio.spec import update_format as update_format

from ._resource_tests import test_description

WeightFormatArgAll = Literal[SupportedWeightsFormat, "all"]
WeightFormatArgAny = Literal[SupportedWeightsFormat, "any"]


def test(
    descr: Union[ResourceDescr, InvalidDescr],
    *,
    weight_format: WeightFormatArgAll = "all",
    devices: Optional[Union[str, Sequence[str]]] = None,
    summary: Union[Path, Sequence[Path]] = (),
    runtime_env: Union[
        Literal["currently-active", "as-described"], Path
    ] = "currently-active",
    determinism: Literal["seed_only", "full"] = "seed_only",
) -> int:
    """Test a bioimageio resource.

    Arguments as described in `bioimageio.core.cli.TestCmd`
    """
    if isinstance(descr, InvalidDescr):
        descr.validation_summary.display()
        return 1

    test_summary = test_description(
        descr,
        weight_format=None if weight_format == "all" else weight_format,
        devices=[devices] if isinstance(devices, str) else devices,
        runtime_env=runtime_env,
        determinism=determinism,
    )
    _ = test_summary.log(summary)
    return 0 if test_summary.status == "passed" else 1


def validate_format(
    descr: Union[ResourceDescr, InvalidDescr],
    summary: Union[Path, Sequence[Path]] = (),
):
    """DEPRECATED; Access the existing `validation_summary` attribute instead.
    validate the meta data format of a bioimageio resource

    Args:
        descr: a bioimageio resource description
    """
    _ = descr.validation_summary.log(summary)
    return 0 if descr.validation_summary.status == "passed" else 1


def package(
    descr: ResourceDescr,
    path: Path,
    *,
    weight_format: WeightFormatArgAll = "all",
):
    """Save a resource's metadata with its associated files.

    Note: If `path` does not have a `.zip` suffix this command will save the
          package as an unzipped folder instead.

    Args:
        descr: a bioimageio resource description
        path: output path
        weight-format: include only this single weight-format (if not 'all').
    """
    if isinstance(descr, InvalidDescr):
        logged = descr.validation_summary.log()
        msg = f"Invalid {descr.type} description."
        if logged:
            msg += f" Details saved to {logged}."

        raise ValueError(msg)

    if weight_format == "all":
        weights_priority_order = None
    else:
        weights_priority_order = (weight_format,)

    if path.suffix == ".zip":
        _ = save_bioimageio_package(
            descr,
            output_path=path,
            weights_priority_order=weights_priority_order,
        )
    else:
        _ = save_bioimageio_package_as_folder(
            descr,
            output_path=path,
            weights_priority_order=weights_priority_order,
        )
    return 0
