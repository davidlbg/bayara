class BayaraError(Exception):
    """Base Bayara exception."""


class BayaraSyntaxError(BayaraError):
    def __init__(self, line_no: int, message: str):
        super().__init__(f"Syntax error on line {line_no}: {message}")
        self.line_no = line_no
        self.message = message


class BayaraSemanticError(BayaraError):
    def __init__(self, line_no: int, message: str):
        super().__init__(f"Semantic error on line {line_no}: {message}")
        self.line_no = line_no
        self.message = message
