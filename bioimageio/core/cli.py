"""bioimageio CLI

Note: Some docstrings use a hair space ' '
      to place the added '(default: ...)' on a new line.
"""

import json
import shutil
import subprocess
import sys
from argparse import RawTextHelpFormatter
from difflib import SequenceMatcher
from functools import cached_property
from io import StringIO
from pathlib import Path
from pprint import pformat, pprint
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import rich.markdown
from loguru import logger
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    CliPositionalArg,
    CliSettingsSource,
    CliSubCommand,
    JsonConfigSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)
from ruyaml import YAML
from tqdm import tqdm
from typing_extensions import assert_never

from bioimageio.spec import (
    AnyModelDescr,
    InvalidDescr,
    ResourceDescr,
    load_description,
    settings,
)
from bioimageio.spec._internal.io_basics import ZipPath
from bioimageio.spec._internal.io_utils import write_yaml
from bioimageio.spec._internal.types import NotEmpty
from bioimageio.spec.dataset import DatasetDescr
from bioimageio.spec.model import ModelDescr, v0_4, v0_5
from bioimageio.spec.notebook import NotebookDescr
from bioimageio.spec.utils import download, ensure_description_is_model

from .commands import (
    WeightFormatArgAll,
    WeightFormatArgAny,
    package,
    test,
    update_format,
)
from .common import MemberId, SampleId, SupportedWeightsFormat
from .digest_spec import get_member_ids, load_sample_for_model
from .io import load_dataset_stat, save_dataset_stat, save_sample
from .prediction import create_prediction_pipeline
from .proc_setup import (
    DatasetMeasure,
    Measure,
    MeasureValue,
    StatsCalculator,
    get_required_dataset_measures,
)
from .sample import Sample
from .stat_measures import Stat
from .utils import VERSION
from .weight_converters._add_weights import add_weights

yaml = YAML(typ="safe")


class CmdBase(BaseModel, use_attribute_docstrings=True, cli_implicit_flags=True):
    pass


class ArgMixin(BaseModel, use_attribute_docstrings=True, cli_implicit_flags=True):
    pass


class WithSummaryLogging(ArgMixin):
    summary: Union[Path, Sequence[Path]] = Field(
        (), examples=[Path("summary.md"), Path("bioimageio_summaries/")]
    )
    """Save the validation summary as JSON, Markdown or HTML.
    The format is chosen based on the suffix: `.json`, `.md`, `.html`.
    If a folder is given (path w/o suffix) the summary is saved in all formats.
    """

    def log(self, descr: Union[ResourceDescr, InvalidDescr]):
        _ = descr.validation_summary.log(self.summary)


class WithSource(ArgMixin):
    source: CliPositionalArg[str]
    """Url/path to a bioimageio.yaml/rdf.yaml file
    or a bioimage.io resource identifier, e.g. 'affable-shark'"""

    @cached_property
    def descr(self):
        return load_description(self.source)

    @property
    def descr_id(self) -> str:
        """a more user-friendly description id
        (replacing legacy ids with their nicknames)
        """
        if isinstance(self.descr, InvalidDescr):
            return str(getattr(self.descr, "id", getattr(self.descr, "name")))

        nickname = None
        if (
            isinstance(self.descr.config, v0_5.Config)
            and (bio_config := self.descr.config.bioimageio)
            and bio_config.model_extra is not None
        ):
            nickname = bio_config.model_extra.get("nickname")

        return str(nickname or self.descr.id or self.descr.name)


class ValidateFormatCmd(CmdBase, WithSource, WithSummaryLogging):
    """Validate the meta data format of a bioimageio resource."""

    perform_io_checks: bool = Field(
        settings.perform_io_checks, alias="perform-io-checks"
    )
    """Wether or not to perform validations that requires downloading remote files.
    Note: Default value is set by `BIOIMAGEIO_PERFORM_IO_CHECKS` environment variable.
    """

    @cached_property
    def descr(self):
        return load_description(self.source, perform_io_checks=self.perform_io_checks)

    def run(self):
        self.log(self.descr)
        sys.exit(0 if self.descr.validation_summary.status == "passed" else 1)


class TestCmd(CmdBase, WithSource, WithSummaryLogging):
    """Test a bioimageio resource (beyond meta data formatting)."""

    weight_format: WeightFormatArgAll = "all"
    """The weight format to limit testing to.

    (only relevant for model resources)"""

    devices: Optional[Union[str, Sequence[str]]] = None
    """Device(s) to use for testing"""

    runtime_env: Union[Literal["currently-active", "as-described"], Path] = Field(
        "currently-active", alias="runtime-env"
    )
    """The python environment to run the tests in
        - `"currently-active"`: use active Python interpreter
        - `"as-described"`: generate a conda environment YAML file based on the model
            weights description.
        - A path to a conda environment YAML.
          Note: The `bioimageio.core` dependency will be added automatically if not present.
    """

    determinism: Literal["seed_only", "full"] = "seed_only"
    """Modes to improve reproducibility of test outputs."""

    def run(self):
        sys.exit(
            test(
                self.descr,
                weight_format=self.weight_format,
                devices=self.devices,
                summary=self.summary,
                runtime_env=self.runtime_env,
                determinism=self.determinism,
            )
        )


class PackageCmd(CmdBase, WithSource):
    """Save a resource's metadata with its associated files."""

    path: CliPositionalArg[Path]
    """The path to write the (zipped) package to.
    If it does not have a `.zip` suffix
    this command will save the package as an unzipped folder instead."""

    weight_format: WeightFormatArgAll = "all"
    """The weight format to include in the package (for model descriptions only)."""

    def run(self):
        if isinstance(self.descr, InvalidDescr):
            paths = self.descr.validation_summary.log()
            raise ValueError(
                f"Invalid {self.descr.type} description. Logged details to {paths}"
            )

        sys.exit(
            package(
                self.descr,
                self.path,
                weight_format=self.weight_format,
            )
        )


def _get_stat(
    model_descr: AnyModelDescr,
    dataset: Iterable[Sample],
    dataset_length: int,
    stats_path: Path,
) -> Mapping[DatasetMeasure, MeasureValue]:
    req_dataset_meas, _ = get_required_dataset_measures(model_descr)
    if not req_dataset_meas:
        return {}

    req_dataset_meas, _ = get_required_dataset_measures(model_descr)

    if stats_path.exists():
        logger.info("loading precomputed dataset measures from {}", stats_path)
        stat = load_dataset_stat(stats_path)
        for m in req_dataset_meas:
            if m not in stat:
                raise ValueError(f"Missing {m} in {stats_path}")

        return stat

    stats_calc = StatsCalculator(req_dataset_meas)

    for sample in tqdm(
        dataset, total=dataset_length, desc="precomputing dataset stats", unit="sample"
    ):
        stats_calc.update(sample)

    stat = stats_calc.finalize()
    save_dataset_stat(stat, stats_path)

    return stat


class UpdateFormatCmd(CmdBase, WithSource):
    """Update the metadata format"""

    output: Optional[Path] = None
    """Save updated bioimageio.yaml to this file.

    Updated bioimageio.yaml is rendered to the terminal if the output is None.
    """

    def run(self):
        updated = update_format(self.descr, output_path=self.output)
        updated_stream = StringIO()
        write_yaml(updated, updated_stream)
        updated_md = f"```yaml\n{updated_stream.getvalue()}\n```"

        rich_markdown = rich.markdown.Markdown(updated_md)
        console = rich.console.Console()
        console.print(rich_markdown)


class PredictCmd(CmdBase, WithSource):
    """Run inference on your data with a bioimage.io model."""

    inputs: NotEmpty[Sequence[Union[str, NotEmpty[Tuple[str, ...]]]]] = (
        "{input_id}/001.tif",
    )
    """Model input sample paths (for each input tensor)

    The input paths are expected to have shape...
     - (n_samples,) or (n_samples,1) for models expecting a single input tensor
     - (n_samples,) containing the substring '{input_id}', or
     - (n_samples, n_model_inputs) to provide each input tensor path explicitly.

    All substrings that are replaced by metadata from the model description:
    - '{model_id}'
    - '{input_id}'

    Example inputs to process sample 'a' and 'b'
    for a model expecting a 'raw' and a 'mask' input tensor:
    --inputs="[[\"a_raw.tif\",\"a_mask.tif\"],[\"b_raw.tif\",\"b_mask.tif\"]]"
    (Note that JSON double quotes need to be escaped.)

    Alternatively a `bioimageio-cli.yaml` (or `bioimageio-cli.json`) file
    may provide the arguments, e.g.:
    ```yaml
    inputs:
    - [a_raw.tif, a_mask.tif]
    - [b_raw.tif, b_mask.tif]
    ```

    `.npy` and any file extension supported by imageio are supported.
     Aavailable formats are listed at
    https://imageio.readthedocs.io/en/stable/formats/index.html#all-formats.
    Some formats have additional dependencies.

     
    """

    outputs: Union[str, NotEmpty[Tuple[str, ...]]] = (
        "outputs_{model_id}/{output_id}/{sample_id}.tif"
    )
    """Model output path pattern (per output tensor)

    All substrings that are replaced:
    - '{model_id}' (from model description)
    - '{output_id}' (from model description)
    - '{sample_id}' (extracted from input paths)

     
    """

    overwrite: bool = False
    """allow overwriting existing output files"""

    blockwise: bool = False
    """process inputs blockwise"""

    stats: Path = Path("dataset_statistics.json")
    """path to dataset statistics
    (will be written if it does not exist,
    but the model requires statistical dataset measures)
     """

    preview: bool = False
    """preview which files would be processed
    and what outputs would be generated."""

    weight_format: WeightFormatArgAny = "any"
    """The weight format to use."""

    example: bool = False
    """generate and run an example

    1. downloads example model inputs
    2. creates a `{model_id}_example` folder
    3. writes input arguments to `{model_id}_example/bioimageio-cli.yaml`
    4. executes a preview dry-run
    5. executes prediction with example input

     
    """

    def _example(self):
        model_descr = ensure_description_is_model(self.descr)
        input_ids = get_member_ids(model_descr.inputs)
        example_inputs = (
            model_descr.sample_inputs
            if isinstance(model_descr, v0_4.ModelDescr)
            else [ipt.sample_tensor or ipt.test_tensor for ipt in model_descr.inputs]
        )
        if not example_inputs:
            raise ValueError(f"{self.descr_id} does not specify any example inputs.")

        inputs001: List[str] = []
        example_path = Path(f"{self.descr_id}_example")
        example_path.mkdir(exist_ok=True)

        for t, src in zip(input_ids, example_inputs):
            local = download(src).path
            dst = Path(f"{example_path}/{t}/001{''.join(local.suffixes)}")
            dst.parent.mkdir(parents=True, exist_ok=True)
            inputs001.append(dst.as_posix())
            if isinstance(local, Path):
                shutil.copy(local, dst)
            elif isinstance(local, ZipPath):
                _ = local.root.extract(local.at, path=dst)
            else:
                assert_never(local)

        inputs = [tuple(inputs001)]
        output_pattern = f"{example_path}/outputs/{{output_id}}/{{sample_id}}.tif"

        bioimageio_cli_path = example_path / YAML_FILE
        stats_file = "dataset_statistics.json"
        stats = (example_path / stats_file).as_posix()
        yaml.dump(
            dict(
                inputs=inputs,
                outputs=output_pattern,
                stats=stats_file,
                blockwise=self.blockwise,
            ),
            bioimageio_cli_path,
        )

        yaml_file_content = None

        # escaped double quotes
        inputs_json = json.dumps(inputs)
        inputs_escaped = inputs_json.replace('"', r"\"")
        source_escaped = self.source.replace('"', r"\"")

        def get_example_command(preview: bool, escape: bool = False):
            q: str = '"' if escape else ""

            return [
                "bioimageio",
                "predict",
                # --no-preview not supported for py=3.8
                *(["--preview"] if preview else []),
                "--overwrite",
                *(["--blockwise"] if self.blockwise else []),
                f"--stats={q}{stats}{q}",
                f"--inputs={q}{inputs_escaped if escape else inputs_json}{q}",
                f"--outputs={q}{output_pattern}{q}",
                f"{q}{source_escaped if escape else self.source}{q}",
            ]

        if Path(YAML_FILE).exists():
            logger.info(
                "temporarily removing '{}' to execute example prediction", YAML_FILE
            )
            yaml_file_content = Path(YAML_FILE).read_bytes()
            Path(YAML_FILE).unlink()

        try:
            _ = subprocess.run(get_example_command(True), check=True)
            _ = subprocess.run(get_example_command(False), check=True)
        finally:
            if yaml_file_content is not None:
                _ = Path(YAML_FILE).write_bytes(yaml_file_content)
                logger.debug("restored '{}'", YAML_FILE)

        print(
            "🎉 Sucessfully ran example prediction!\n"
            + "To predict the example input using the CLI example config file"
            + f" {example_path/YAML_FILE}, execute `bioimageio predict` from {example_path}:\n"
            + f"$ cd {str(example_path)}\n"
            + f'$ bioimageio predict "{source_escaped}"\n\n'
            + "Alternatively run the following command"
            + " in the current workind directory, not the example folder:\n$ "
            + " ".join(get_example_command(False, escape=True))
            + f"\n(note that a local '{JSON_FILE}' or '{YAML_FILE}' may interfere with this)"
        )

    def run(self):
        if self.example:
            return self._example()

        model_descr = ensure_description_is_model(self.descr)

        input_ids = get_member_ids(model_descr.inputs)
        output_ids = get_member_ids(model_descr.outputs)

        minimum_input_ids = tuple(
            str(ipt.id) if isinstance(ipt, v0_5.InputTensorDescr) else str(ipt.name)
            for ipt in model_descr.inputs
            if not isinstance(ipt, v0_5.InputTensorDescr) or not ipt.optional
        )
        maximum_input_ids = tuple(
            str(ipt.id) if isinstance(ipt, v0_5.InputTensorDescr) else str(ipt.name)
            for ipt in model_descr.inputs
        )

        def expand_inputs(i: int, ipt: Union[str, Tuple[str, ...]]) -> Tuple[str, ...]:
            if isinstance(ipt, str):
                ipts = tuple(
                    ipt.format(model_id=self.descr_id, input_id=t) for t in input_ids
                )
            else:
                ipts = tuple(
                    p.format(model_id=self.descr_id, input_id=t)
                    for t, p in zip(input_ids, ipt)
                )

            if len(set(ipts)) < len(ipts):
                if len(minimum_input_ids) == len(maximum_input_ids):
                    n = len(minimum_input_ids)
                else:
                    n = f"{len(minimum_input_ids)}-{len(maximum_input_ids)}"

                raise ValueError(
                    f"[input sample #{i}] Include '{{input_id}}' in path pattern or explicitly specify {n} distinct input paths (got {ipt})"
                )

            if len(ipts) < len(minimum_input_ids):
                raise ValueError(
                    f"[input sample #{i}] Expected at least {len(minimum_input_ids)} inputs {minimum_input_ids}, got {ipts}"
                )

            if len(ipts) > len(maximum_input_ids):
                raise ValueError(
                    f"Expected at most {len(maximum_input_ids)} inputs {maximum_input_ids}, got {ipts}"
                )

            return ipts

        inputs = [expand_inputs(i, ipt) for i, ipt in enumerate(self.inputs, start=1)]

        sample_paths_in = [
            {t: Path(p) for t, p in zip(input_ids, ipts)} for ipts in inputs
        ]

        sample_ids = _get_sample_ids(sample_paths_in)

        def expand_outputs():
            if isinstance(self.outputs, str):
                outputs = [
                    tuple(
                        Path(
                            self.outputs.format(
                                model_id=self.descr_id, output_id=t, sample_id=s
                            )
                        )
                        for t in output_ids
                    )
                    for s in sample_ids
                ]
            else:
                outputs = [
                    tuple(
                        Path(p.format(model_id=self.descr_id, output_id=t, sample_id=s))
                        for t, p in zip(output_ids, self.outputs)
                    )
                    for s in sample_ids
                ]

            for i, out in enumerate(outputs, start=1):
                if len(set(out)) < len(out):
                    raise ValueError(
                        f"[output sample #{i}] Include '{{output_id}}' in path pattern or explicitly specify {len(output_ids)} distinct output paths (got {out})"
                    )

                if len(out) != len(output_ids):
                    raise ValueError(
                        f"[output sample #{i}] Expected {len(output_ids)} outputs {output_ids}, got {out}"
                    )

            return outputs

        outputs = expand_outputs()

        sample_paths_out = [
            {MemberId(t): Path(p) for t, p in zip(output_ids, out)} for out in outputs
        ]

        if not self.overwrite:
            for sample_paths in sample_paths_out:
                for p in sample_paths.values():
                    if p.exists():
                        raise FileExistsError(
                            f"{p} already exists. use --overwrite to (re-)write outputs anyway."
                        )
        if self.preview:
            print("🛈 bioimageio prediction preview structure:")
            pprint(
                {
                    "{sample_id}": dict(
                        inputs={"{input_id}": "<input path>"},
                        outputs={"{output_id}": "<output path>"},
                    )
                }
            )
            print("🔎 bioimageio prediction preview output:")
            pprint(
                {
                    s: dict(
                        inputs={t: p.as_posix() for t, p in sp_in.items()},
                        outputs={t: p.as_posix() for t, p in sp_out.items()},
                    )
                    for s, sp_in, sp_out in zip(
                        sample_ids, sample_paths_in, sample_paths_out
                    )
                }
            )
            return

        def input_dataset(stat: Stat):
            for s, sp_in in zip(sample_ids, sample_paths_in):
                yield load_sample_for_model(
                    model=model_descr,
                    paths=sp_in,
                    stat=stat,
                    sample_id=s,
                )

        stat: Dict[Measure, MeasureValue] = dict(
            _get_stat(
                model_descr, input_dataset({}), len(sample_ids), self.stats
            ).items()
        )

        pp = create_prediction_pipeline(
            model_descr,
            weight_format=None if self.weight_format == "any" else self.weight_format,
        )
        predict_method = (
            pp.predict_sample_with_blocking
            if self.blockwise
            else pp.predict_sample_without_blocking
        )

        for sample_in, sp_out in tqdm(
            zip(input_dataset(dict(stat)), sample_paths_out),
            total=len(inputs),
            desc=f"predict with {self.descr_id}",
            unit="sample",
        ):
            sample_out = predict_method(sample_in)
            save_sample(sp_out, sample_out)


class ConvertWeightsCmd(CmdBase, WithSource):
    output: CliPositionalArg[Path]
    """The path to write the updated model package to."""

    source_format: Optional[SupportedWeightsFormat] = Field(None, alias="source-format")
    """Exclusively use these weights to convert to other formats."""

    target_format: Optional[SupportedWeightsFormat] = Field(None, alias="target-format")
    """Exclusively add this weight format."""

    verbose: bool = False
    """Log more (error) output."""

    def run(self):
        model_descr = ensure_description_is_model(self.descr)
        if isinstance(model_descr, v0_4.ModelDescr):
            raise TypeError(
                f"model format {model_descr.format_version} not supported."
                + " Please update the model first."
            )
        updated_model_descr = add_weights(
            model_descr,
            output_path=self.output,
            source_format=self.source_format,
            target_format=self.target_format,
            verbose=self.verbose,
        )
        if updated_model_descr is None:
            return

        _ = updated_model_descr.validation_summary.log()


JSON_FILE = "bioimageio-cli.json"
YAML_FILE = "bioimageio-cli.yaml"


class Bioimageio(
    BaseSettings,
    cli_implicit_flags=True,
    cli_parse_args=True,
    cli_prog_name="bioimageio",
    cli_use_class_docs_for_groups=True,
    use_attribute_docstrings=True,
):
    """bioimageio - CLI for bioimage.io resources 🦒"""

    model_config = SettingsConfigDict(
        json_file=JSON_FILE,
        yaml_file=YAML_FILE,
    )

    validate_format: CliSubCommand[ValidateFormatCmd] = Field(alias="validate-format")
    "Check a resource's metadata format"

    test: CliSubCommand[TestCmd]
    "Test a bioimageio resource (beyond meta data formatting)"

    package: CliSubCommand[PackageCmd]
    "Package a resource"

    predict: CliSubCommand[PredictCmd]
    "Predict with a model resource"

    update_format: CliSubCommand[UpdateFormatCmd] = Field(alias="update-format")
    """Update the metadata format"""

    add_weights: CliSubCommand[ConvertWeightsCmd] = Field(alias="add-weights")
    """Add additional weights to the model descriptions converted from available
    formats to improve deployability."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: Type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> Tuple[PydanticBaseSettingsSource, ...]:
        cli: CliSettingsSource[BaseSettings] = CliSettingsSource(
            settings_cls,
            cli_parse_args=True,
            formatter_class=RawTextHelpFormatter,
        )
        sys_args = pformat(sys.argv)
        logger.info("starting CLI with arguments:\n{}", sys_args)
        return (
            cli,
            init_settings,
            YamlConfigSettingsSource(settings_cls),
            JsonConfigSettingsSource(settings_cls),
        )

    @model_validator(mode="before")
    @classmethod
    def _log(cls, data: Any):
        logger.info(
            "loaded CLI input:\n{}",
            pformat({k: v for k, v in data.items() if v is not None}),
        )
        return data

    def run(self):
        logger.info(
            "executing CLI command:\n{}",
            pformat({k: v for k, v in self.model_dump().items() if v is not None}),
        )
        cmd = (
            self.validate_format
            or self.test
            or self.package
            or self.predict
            or self.update_format
            or self.add_weights
        )
        assert cmd is not None
        cmd.run()


assert isinstance(Bioimageio.__doc__, str)
Bioimageio.__doc__ += f"""

library versions:
  bioimageio.core {VERSION}
  bioimageio.spec {VERSION}

spec format versions:
        model RDF {ModelDescr.implemented_format_version}
      dataset RDF {DatasetDescr.implemented_format_version}
     notebook RDF {NotebookDescr.implemented_format_version}

"""


def _get_sample_ids(
    input_paths: Sequence[Mapping[MemberId, Path]],
) -> Sequence[SampleId]:
    """Get sample ids for given input paths, based on the common path per sample.

    Falls back to sample01, samle02, etc..."""

    matcher = SequenceMatcher()

    def get_common_seq(seqs: Sequence[Sequence[str]]) -> Sequence[str]:
        """extract a common sequence from multiple sequences
        (order sensitive; strips whitespace and slashes)
        """
        common = seqs[0]

        for seq in seqs[1:]:
            if not seq:
                continue
            matcher.set_seqs(common, seq)
            i, _, size = matcher.find_longest_match()
            common = common[i : i + size]

        if isinstance(common, str):
            common = common.strip().strip("/")
        else:
            common = [cs for c in common if (cs := c.strip().strip("/"))]

        if not common:
            raise ValueError(f"failed to find common sequence for {seqs}")

        return common

    def get_shorter_diff(seqs: Sequence[Sequence[str]]) -> List[Sequence[str]]:
        """get a shorter sequence whose entries are still unique
        (order sensitive, not minimal sequence)
        """
        min_seq_len = min(len(s) for s in seqs)
        # cut from the start
        for start in range(min_seq_len - 1, -1, -1):
            shortened = [s[start:] for s in seqs]
            if len(set(shortened)) == len(seqs):
                min_seq_len -= start
                break
        else:
            seen: Set[Sequence[str]] = set()
            dupes = [s for s in seqs if s in seen or seen.add(s)]
            raise ValueError(f"Found duplicate entries {dupes}")

        # cut from the end
        for end in range(min_seq_len - 1, 1, -1):
            shortened = [s[:end] for s in shortened]
            if len(set(shortened)) == len(seqs):
                break

        return shortened

    full_tensor_ids = [
        sorted(
            p.resolve().with_suffix("").as_posix() for p in input_sample_paths.values()
        )
        for input_sample_paths in input_paths
    ]
    try:
        long_sample_ids = [get_common_seq(t) for t in full_tensor_ids]
        sample_ids = get_shorter_diff(long_sample_ids)
    except ValueError as e:
        raise ValueError(f"failed to extract sample ids: {e}")

    return sample_ids
