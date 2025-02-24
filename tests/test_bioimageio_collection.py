from typing import Any, Collection, Dict, Iterable, Mapping, Tuple

import pytest
import requests
from pydantic import HttpUrl

from bioimageio.spec import InvalidDescr
from bioimageio.spec.common import Sha256
from tests.utils import ParameterSet, expensive_test

BASE_URL = "https://uk1s3.embassy.ebi.ac.uk/public-datasets/bioimage.io/"


def _get_latest_rdf_sources():
    entries: Any = requests.get(BASE_URL + "all_versions.json").json()["entries"]
    ret: Dict[str, Tuple[HttpUrl, Sha256]] = {}
    for entry in entries:
        version = entry["versions"][0]
        ret[f"{entry['concept']}/{version['v']}"] = (
            HttpUrl(version["source"]),  # pyright: ignore[reportCallIssue]
            Sha256(version["sha256"]),
        )

    return ret


ALL_LATEST_RDF_SOURCES: Mapping[str, Tuple[HttpUrl, Sha256]] = _get_latest_rdf_sources()


def yield_bioimageio_yaml_urls() -> Iterable[ParameterSet]:
    for descr_url, sha in ALL_LATEST_RDF_SOURCES.values():
        key = (
            str(descr_url)
            .replace(BASE_URL, "")
            .replace("/files/rdf.yaml", "")
            .replace("/files/bioimageio.yaml", "")
        )
        yield pytest.param(descr_url, sha, key, id=key)


KNOWN_INVALID: Collection[str] = set()


@expensive_test
@pytest.mark.parametrize("descr_url,sha,key", list(yield_bioimageio_yaml_urls()))
def test_rdf(
    descr_url: HttpUrl,
    sha: Sha256,
    key: str,
):
    if key in KNOWN_INVALID:
        pytest.skip("known failure")

    from bioimageio.core import load_description_and_test

    descr = load_description_and_test(descr_url, sha256=sha)
    assert not isinstance(descr, InvalidDescr)
    assert (
        descr.validation_summary.status == "passed"
    ), descr.validation_summary.format()
