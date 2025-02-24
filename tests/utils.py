"""utils to test bioimageio.core"""

import os
from typing import Any, Protocol, Sequence

import pytest


class ParameterSet(Protocol):
    def __init__(self, values: Sequence[Any], marks: Any, id: str) -> None:
        super().__init__()


class test_func(Protocol):
    def __call__(*args: Any, **kwargs: Any): ...


expensive_test = pytest.mark.skipif(
    os.getenv("RUN_EXPENSIVE_TESTS") != "true",
    reason="Skipping expensive test (enable by RUN_EXPENSIVE_TESTS='true')",
)
