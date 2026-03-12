from pathlib import Path
from bayara.cli import cmd_check


def test_check_smoke():
    path = Path('examples/basic_classification.bay')
    assert cmd_check(path) == 0
