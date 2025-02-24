import collections.abc
import warnings
import zipfile
from io import TextIOWrapper
from pathlib import Path, PurePosixPath
from shutil import copyfileobj
from typing import (
    Any,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import h5py
import numpy as np
from imageio.v3 import imread, imwrite  # type: ignore
from loguru import logger
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, TypeAdapter
from typing_extensions import assert_never

from bioimageio.spec._internal.io import interprete_file_source
from bioimageio.spec.common import (
    HttpUrl,
    PermissiveFileSource,
    RelativeFilePath,
    ZipPath,
)
from bioimageio.spec.utils import download, load_array, save_array

from .axis import AxisLike
from .common import PerMember
from .sample import Sample
from .stat_measures import DatasetMeasure, MeasureValue
from .tensor import Tensor

DEFAULT_H5_DATASET_PATH = "data"


SUFFIXES_WITH_DATAPATH = (".h5", ".hdf", ".hdf5")


def load_image(
    source: PermissiveFileSource, is_volume: Optional[bool] = None
) -> NDArray[Any]:
    """load a single image as numpy array

    Args:
        source: image source
        is_volume: deprecated
    """
    if is_volume is not None:
        warnings.warn("**is_volume** is deprecated and will be removed soon.")

    parsed_source = interprete_file_source(source)

    if isinstance(parsed_source, RelativeFilePath):
        src = parsed_source.absolute()
    else:
        src = parsed_source

    # FIXME: why is pyright complaining about giving the union to _split_dataset_path?
    if isinstance(src, Path):
        file_source, subpath = _split_dataset_path(src)
    elif isinstance(src, HttpUrl):
        file_source, subpath = _split_dataset_path(src)
    elif isinstance(src, ZipPath):
        file_source, subpath = _split_dataset_path(src)
    else:
        assert_never(src)

    path = download(file_source).path

    if path.suffix == ".npy":
        if subpath is not None:
            raise ValueError(f"Unexpected subpath {subpath} for .npy path {path}")
        return load_array(path)
    elif path.suffix in SUFFIXES_WITH_DATAPATH:
        if subpath is None:
            dataset_path = DEFAULT_H5_DATASET_PATH
        else:
            dataset_path = str(subpath)

        with h5py.File(path, "r") as f:
            h5_dataset = f.get(  # pyright: ignore[reportUnknownVariableType]
                dataset_path
            )
            if not isinstance(h5_dataset, h5py.Dataset):
                raise ValueError(
                    f"{path} is not of type {h5py.Dataset}, but has type "
                    + str(
                        type(h5_dataset)  # pyright: ignore[reportUnknownArgumentType]
                    )
                )
            image: NDArray[Any]
            image = h5_dataset[:]  # pyright: ignore[reportUnknownVariableType]
            assert isinstance(image, np.ndarray), type(
                image  # pyright: ignore[reportUnknownArgumentType]
            )
            return image  # pyright: ignore[reportUnknownVariableType]
    elif isinstance(path, ZipPath):
        return imread(
            path.read_bytes(), extension=path.suffix
        )  # pyright: ignore[reportUnknownVariableType]
    else:
        return imread(path)  # pyright: ignore[reportUnknownVariableType]


def load_tensor(
    path: Union[Path, str], axes: Optional[Sequence[AxisLike]] = None
) -> Tensor:
    # TODO: load axis meta data
    array = load_image(path)

    return Tensor.from_numpy(array, dims=axes)


_SourceT = TypeVar("_SourceT", Path, HttpUrl, ZipPath)


def _split_dataset_path(
    source: _SourceT,
) -> Tuple[_SourceT, Optional[PurePosixPath]]:
    """Split off subpath (e.g. internal  h5 dataset path)
    from a file path following a file extension.

    Examples:
        >>> _split_dataset_path(Path("my_file.h5/dataset"))
        (...Path('my_file.h5'), PurePosixPath('dataset'))

        >>> _split_dataset_path(Path("my_plain_file"))
        (...Path('my_plain_file'), None)

    """
    if isinstance(source, RelativeFilePath):
        src = source.absolute()
    else:
        src = source

    del source

    def separate_pure_path(path: PurePosixPath):
        for p in path.parents:
            if p.suffix in SUFFIXES_WITH_DATAPATH:
                return p, PurePosixPath(path.relative_to(p))

        return path, None

    if isinstance(src, HttpUrl):
        file_path, data_path = separate_pure_path(PurePosixPath(src.path or ""))

        if data_path is None:
            return src, None

        return (
            HttpUrl(str(file_path).replace(f"/{data_path}", "")),
            data_path,
        )

    if isinstance(src, ZipPath):
        file_path, data_path = separate_pure_path(PurePosixPath(str(src)))

        if data_path is None:
            return src, None

        return (
            ZipPath(str(file_path).replace(f"/{data_path}", "")),
            data_path,
        )

    file_path, data_path = separate_pure_path(PurePosixPath(src))
    return Path(file_path), data_path


def save_tensor(path: Union[Path, str], tensor: Tensor) -> None:
    # TODO: save axis meta data

    data: NDArray[Any] = tensor.data.to_numpy()
    file_path, subpath = _split_dataset_path(Path(path))
    if not file_path.suffix:
        raise ValueError(f"No suffix (needed to decide file format) found in {path}")

    file_path.parent.mkdir(exist_ok=True, parents=True)
    if file_path.suffix == ".npy":
        if subpath is not None:
            raise ValueError(f"Unexpected subpath {subpath} found in .npy path {path}")
        save_array(file_path, data)
    elif file_path.suffix in (".h5", ".hdf", ".hdf5"):
        if subpath is None:
            dataset_path = DEFAULT_H5_DATASET_PATH
        else:
            dataset_path = str(subpath)

        with h5py.File(file_path, "a") as f:
            if dataset_path in f:
                del f[dataset_path]

            _ = f.create_dataset(dataset_path, data=data, chunks=True)
    else:
        # if singleton_axes := [a for a, s in tensor.tagged_shape.items() if s == 1]:
        #     tensor = tensor[{a: 0 for a in singleton_axes}]
        #     singleton_axes_msg = f"(without singleton axes {singleton_axes}) "
        # else:
        singleton_axes_msg = ""

        logger.debug(
            "writing tensor {} {}to {}",
            dict(tensor.tagged_shape),
            singleton_axes_msg,
            path,
        )
        imwrite(path, data)


def save_sample(
    path: Union[Path, str, PerMember[Union[Path, str]]], sample: Sample
) -> None:
    """Save a **sample** to a **path** pattern
    or all sample members in the **path** mapping.

    If **path** is a pathlib.Path or a string and the **sample** has multiple members,
    **path** it must contain `{member_id}` (or `{input_id}` or `{output_id}`).

    (Each) **path** may contain `{sample_id}` to be formatted with the **sample** object.
    """
    if not isinstance(path, collections.abc.Mapping):
        if len(sample.members) < 2 or any(
            m in str(path) for m in ("{member_id}", "{input_id}", "{output_id}")
        ):
            path = {m: path for m in sample.members}
        else:
            raise ValueError(
                f"path {path} must contain '{{member_id}}' for sample with multiple members {list(sample.members)}."
            )

    for m, p in path.items():
        t = sample.members[m]
        p_formatted = Path(
            str(p).format(sample_id=sample.id, member_id=m, input_id=m, output_id=m)
        )
        save_tensor(p_formatted, t)


class _SerializedDatasetStatsEntry(
    BaseModel, frozen=True, arbitrary_types_allowed=True
):
    measure: DatasetMeasure
    value: MeasureValue


_stat_adapter = TypeAdapter(
    Sequence[_SerializedDatasetStatsEntry],
    config=ConfigDict(arbitrary_types_allowed=True),
)


def save_dataset_stat(stat: Mapping[DatasetMeasure, MeasureValue], path: Path):
    serializable = [
        _SerializedDatasetStatsEntry(measure=k, value=v) for k, v in stat.items()
    ]
    _ = path.write_bytes(_stat_adapter.dump_json(serializable))


def load_dataset_stat(path: Path):
    seq = _stat_adapter.validate_json(path.read_bytes())
    return {e.measure: e.value for e in seq}


def ensure_unzipped(source: Union[PermissiveFileSource, ZipPath], folder: Path):
    """unzip a (downloaded) **source** to a file in **folder** if source is a zip archive.
    Always returns the path to the unzipped source (maybe source itself)"""
    local_weights_file = download(source).path
    if isinstance(local_weights_file, ZipPath):
        # source is inside a zip archive
        out_path = folder / local_weights_file.filename
        with local_weights_file.open("rb") as src, out_path.open("wb") as dst:
            assert not isinstance(src, TextIOWrapper)
            copyfileobj(src, dst)

        local_weights_file = out_path

    if zipfile.is_zipfile(local_weights_file):
        # source itself is a zipfile
        out_path = folder / local_weights_file.with_suffix(".unzipped").name
        with zipfile.ZipFile(local_weights_file, "r") as f:
            f.extractall(out_path)

        return out_path
    else:
        return local_weights_file
