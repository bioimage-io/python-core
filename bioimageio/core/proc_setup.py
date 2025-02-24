from typing import (
    Callable,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Union,
)

from typing_extensions import assert_never

from bioimageio.core.digest_spec import get_member_id
from bioimageio.spec.model import AnyModelDescr, v0_4, v0_5

from .proc_ops import (
    AddKnownDatasetStats,
    EnsureDtype,
    Processing,
    UpdateStats,
    get_proc,
)
from .sample import Sample
from .stat_calculators import StatsCalculator
from .stat_measures import (
    DatasetMeasure,
    DatasetMeasureBase,
    Measure,
    MeasureValue,
    SampleMeasure,
    SampleMeasureBase,
)

TensorDescr = Union[
    v0_4.InputTensorDescr,
    v0_4.OutputTensorDescr,
    v0_5.InputTensorDescr,
    v0_5.OutputTensorDescr,
]


class PreAndPostprocessing(NamedTuple):
    pre: List[Processing]
    post: List[Processing]


class _ProcessingCallables(NamedTuple):
    pre: Callable[[Sample], None]
    post: Callable[[Sample], None]


class _SetupProcessing(NamedTuple):
    pre: List[Processing]
    post: List[Processing]
    pre_measures: Set[Measure]
    post_measures: Set[Measure]


class _ApplyProcs:
    def __init__(self, procs: Sequence[Processing]):
        super().__init__()
        self._procs = procs

    def __call__(self, sample: Sample) -> None:
        for op in self._procs:
            op(sample)


def get_pre_and_postprocessing(
    model: AnyModelDescr,
    *,
    dataset_for_initial_statistics: Iterable[Sample],
    keep_updating_initial_dataset_stats: bool = False,
    fixed_dataset_stats: Optional[Mapping[DatasetMeasure, MeasureValue]] = None,
) -> _ProcessingCallables:
    """Creates callables to apply pre- and postprocessing in-place to a sample"""

    setup = setup_pre_and_postprocessing(
        model=model,
        dataset_for_initial_statistics=dataset_for_initial_statistics,
        keep_updating_initial_dataset_stats=keep_updating_initial_dataset_stats,
        fixed_dataset_stats=fixed_dataset_stats,
    )
    return _ProcessingCallables(_ApplyProcs(setup.pre), _ApplyProcs(setup.post))


def setup_pre_and_postprocessing(
    model: AnyModelDescr,
    dataset_for_initial_statistics: Iterable[Sample],
    keep_updating_initial_dataset_stats: bool = False,
    fixed_dataset_stats: Optional[Mapping[DatasetMeasure, MeasureValue]] = None,
) -> PreAndPostprocessing:
    """
    Get pre- and postprocessing operators for a `model` description.
    Used in `bioimageio.core.create_prediction_pipeline"""
    prep, post, prep_meas, post_meas = _prepare_setup_pre_and_postprocessing(model)

    missing_dataset_stats = {
        m
        for m in prep_meas | post_meas
        if fixed_dataset_stats is None or m not in fixed_dataset_stats
    }
    if missing_dataset_stats:
        initial_stats_calc = StatsCalculator(missing_dataset_stats)
        for sample in dataset_for_initial_statistics:
            initial_stats_calc.update(sample)

        initial_stats = initial_stats_calc.finalize()
    else:
        initial_stats = {}

    prep.insert(
        0,
        UpdateStats(
            StatsCalculator(prep_meas, initial_stats),
            keep_updating_initial_dataset_stats=keep_updating_initial_dataset_stats,
        ),
    )
    if post_meas:
        post.insert(
            0,
            UpdateStats(
                StatsCalculator(post_meas, initial_stats),
                keep_updating_initial_dataset_stats=keep_updating_initial_dataset_stats,
            ),
        )

    if fixed_dataset_stats:
        prep.insert(0, AddKnownDatasetStats(fixed_dataset_stats))
        post.insert(0, AddKnownDatasetStats(fixed_dataset_stats))

    return PreAndPostprocessing(prep, post)


class RequiredMeasures(NamedTuple):
    pre: Set[Measure]
    post: Set[Measure]


class RequiredDatasetMeasures(NamedTuple):
    pre: Set[DatasetMeasure]
    post: Set[DatasetMeasure]


class RequiredSampleMeasures(NamedTuple):
    pre: Set[SampleMeasure]
    post: Set[SampleMeasure]


def get_requried_measures(model: AnyModelDescr) -> RequiredMeasures:
    s = _prepare_setup_pre_and_postprocessing(model)
    return RequiredMeasures(s.pre_measures, s.post_measures)


def get_required_dataset_measures(model: AnyModelDescr) -> RequiredDatasetMeasures:
    s = _prepare_setup_pre_and_postprocessing(model)
    return RequiredDatasetMeasures(
        {m for m in s.pre_measures if isinstance(m, DatasetMeasureBase)},
        {m for m in s.post_measures if isinstance(m, DatasetMeasureBase)},
    )


def get_requried_sample_measures(model: AnyModelDescr) -> RequiredSampleMeasures:
    s = _prepare_setup_pre_and_postprocessing(model)
    return RequiredSampleMeasures(
        {m for m in s.pre_measures if isinstance(m, SampleMeasureBase)},
        {m for m in s.post_measures if isinstance(m, SampleMeasureBase)},
    )


def _prepare_procs(
    tensor_descrs: Union[
        Sequence[v0_4.InputTensorDescr],
        Sequence[v0_5.InputTensorDescr],
        Sequence[v0_4.OutputTensorDescr],
        Sequence[v0_5.OutputTensorDescr],
    ],
) -> List[Processing]:
    procs: List[Processing] = []
    for t_descr in tensor_descrs:
        if isinstance(t_descr, (v0_4.InputTensorDescr, v0_4.OutputTensorDescr)):
            member_id = get_member_id(t_descr)
            procs.append(
                EnsureDtype(input=member_id, output=member_id, dtype=t_descr.data_type)
            )

        if isinstance(t_descr, (v0_4.InputTensorDescr, v0_5.InputTensorDescr)):
            for proc_d in t_descr.preprocessing:
                procs.append(get_proc(proc_d, t_descr))
        elif isinstance(t_descr, (v0_4.OutputTensorDescr, v0_5.OutputTensorDescr)):
            for proc_d in t_descr.postprocessing:
                procs.append(get_proc(proc_d, t_descr))
        else:
            assert_never(t_descr)

        if isinstance(
            t_descr,
            (v0_4.InputTensorDescr, (v0_4.InputTensorDescr, v0_4.OutputTensorDescr)),
        ) and not isinstance(procs[-1], EnsureDtype):
            member_id = get_member_id(t_descr)
            procs.append(
                EnsureDtype(input=member_id, output=member_id, dtype=t_descr.data_type)
            )

    return procs


def _prepare_setup_pre_and_postprocessing(model: AnyModelDescr) -> _SetupProcessing:
    if isinstance(model, v0_4.ModelDescr):
        pre = _prepare_procs(model.inputs)
        post = _prepare_procs(model.outputs)
    elif isinstance(model, v0_5.ModelDescr):
        pre = _prepare_procs(model.inputs)
        post = _prepare_procs(model.outputs)
    else:
        assert_never(model)

    return _SetupProcessing(
        pre=pre,
        post=post,
        pre_measures={m for proc in pre for m in proc.required_measures},
        post_measures={m for proc in post for m in proc.required_measures},
    )
