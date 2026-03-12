from dataclasses import dataclass
from typing import List


@dataclass
class SourceLine:
    line_no: int
    text: str


def lex_source(source: str) -> List[SourceLine]:
    lines: List[SourceLine] = []
    for idx, raw in enumerate(source.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith('#'):
            continue
        lines.append(SourceLine(idx, stripped))
    return lines
