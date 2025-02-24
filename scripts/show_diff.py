import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import pooch

from bioimageio.core import load_description, save_bioimageio_yaml_only

if __name__ == "__main__":
    rdf_source = "https://raw.githubusercontent.com/bioimage-io/spec-bioimage-io/main/example_descriptions/models/unet2d_nuclei_broad/v0_4_9.bioimageio.yaml"

    local_source = Path(pooch.retrieve(rdf_source, None))
    model_as_is = load_description(rdf_source, format_version="discover")
    model_latest = load_description(rdf_source, format_version="latest")
    print(model_latest.validation_summary)

    with TemporaryDirectory() as tmp:
        as_is = Path(tmp) / "as_is.bioimageio.yaml"

        save_bioimageio_yaml_only(
            model_as_is, file=as_is
        )  # write out as is to avoid sorting diff
        latest = Path(tmp) / "latest.bioimageio.yaml"
        save_bioimageio_yaml_only(model_latest, file=latest)

        _ = subprocess.run(f"git diff --no-index --ignore-all-space {as_is} {latest}")
