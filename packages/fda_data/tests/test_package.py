import fda_data


def test_version_is_exposed() -> None:
    assert fda_data.__version__ == "0.1.0"
