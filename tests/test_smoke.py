import re

import survivor_tagging


def test_package_imports_and_has_version():
    assert isinstance(survivor_tagging.__version__, str)
    assert re.fullmatch(r"\d+\.\d+\.\d+", survivor_tagging.__version__)
