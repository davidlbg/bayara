class BayaraError(Exception):
    pass


class BayaraSyntaxError(BayaraError):
    def __init__(self, message: str, line: int | None = None, column: int | None = None):
        prefix = "[Syntax Error]"
        location = ""
        if line is not None:
            location += f" line {line}"
        if column is not None:
            location += f", column {column}"
        super().__init__(f"{prefix}{location}: {message}")


class BayaraSemanticError(BayaraError):
    def __init__(self, message: str, line: int | None = None):
        prefix = "[Semantic Error]"
        location = f" line {line}" if line is not None else ""
        super().__init__(f"{prefix}{location}: {message}")
