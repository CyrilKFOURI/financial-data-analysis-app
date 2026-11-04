import fda_contracts


def test_version_is_exposed() -> None:
    assert fda_contracts.__version__ == "0.1.0"
