from bioimageio.spec import load_description_and_validate_format_only


def test_model(any_model: str):
    summary = load_description_and_validate_format_only(any_model)
    assert summary.status == "passed", summary.display()
