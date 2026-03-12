from .errors import BayaraSyntaxError
from .tokens import Token


SYMBOLS = {
    '{': 'LBRACE',
    '}': 'RBRACE',
    ',': 'COMMA',
}


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.length = len(source)
        self.pos = 0
        self.line = 1
        self.column = 1

    def current(self) -> str | None:
        if self.pos >= self.length:
            return None
        return self.source[self.pos]

    def peek(self, offset: int = 1) -> str | None:
        idx = self.pos + offset
        if idx >= self.length:
            return None
        return self.source[idx]

    def advance(self) -> str | None:
        ch = self.current()
        if ch is None:
            return None
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def lex(self) -> list[Token]:
        tokens: list[Token] = []
        while (ch := self.current()) is not None:
            start_line, start_col = self.line, self.column

            if ch in ' \t\r':
                self.advance()
                continue

            if ch == '\n':
                tokens.append(Token('NEWLINE', '\n', start_line, start_col))
                self.advance()
                continue

            if ch == '#':
                self._skip_single_line_comment()
                continue

            if ch == '/' and self.peek() == '*':
                self._skip_multiline_comment()
                continue

            if ch == '-' and self.peek() == '>':
                self.advance(); self.advance()
                tokens.append(Token('ARROW', '->', start_line, start_col))
                continue

            if ch in SYMBOLS:
                self.advance()
                tokens.append(Token(SYMBOLS[ch], ch, start_line, start_col))
                continue

            if ch == '"':
                tokens.append(self._lex_string())
                continue

            if ch.isdigit() or (ch == '.' and (self.peek() or '').isdigit()):
                tokens.append(self._lex_number())
                continue

            if ch.isalpha() or ch == '_':
                tokens.append(self._lex_ident())
                continue

            raise BayaraSyntaxError(f"unexpected character '{ch}'", start_line, start_col)

        tokens.append(Token('EOF', '', self.line, self.column))
        return self._compress_newlines(tokens)

    def _skip_single_line_comment(self) -> None:
        while (ch := self.current()) is not None and ch != '\n':
            self.advance()

    def _skip_multiline_comment(self) -> None:
        start_line, start_col = self.line, self.column
        self.advance(); self.advance()  # skip /*
        while True:
            ch = self.current()
            if ch is None:
                raise BayaraSyntaxError('unterminated multiline comment', start_line, start_col)
            if ch == '*' and self.peek() == '/':
                self.advance(); self.advance()
                return
            self.advance()

    def _lex_string(self) -> Token:
        start_line, start_col = self.line, self.column
        self.advance()  # opening quote
        chars: list[str] = []
        while True:
            ch = self.current()
            if ch is None:
                raise BayaraSyntaxError('unterminated string literal', start_line, start_col)
            if ch == '"':
                self.advance()
                return Token('STRING', ''.join(chars), start_line, start_col)
            if ch == '\\':
                self.advance()
                escaped = self.current()
                if escaped is None:
                    raise BayaraSyntaxError('unterminated escape sequence', self.line, self.column)
                mapping = {'n': '\n', 't': '\t', '"': '"', '\\': '\\'}
                chars.append(mapping.get(escaped, escaped))
                self.advance()
                continue
            chars.append(ch)
            self.advance()

    def _lex_number(self) -> Token:
        start_line, start_col = self.line, self.column
        chars: list[str] = []
        dot_count = 0
        while (ch := self.current()) is not None and (ch.isdigit() or ch == '.'):
            if ch == '.':
                dot_count += 1
                if dot_count > 1:
                    raise BayaraSyntaxError('invalid numeric literal', start_line, start_col)
            chars.append(ch)
            self.advance()
        return Token('NUMBER', ''.join(chars), start_line, start_col)

    def _lex_ident(self) -> Token:
        start_line, start_col = self.line, self.column
        chars: list[str] = []
        while (ch := self.current()) is not None and (ch.isalnum() or ch in {'_', '.'}):
            chars.append(ch)
            self.advance()
        return Token('IDENT', ''.join(chars), start_line, start_col)

    def _compress_newlines(self, tokens: list[Token]) -> list[Token]:
        compressed: list[Token] = []
        prev_newline = True
        for token in tokens:
            if token.type == 'NEWLINE':
                if prev_newline:
                    continue
                prev_newline = True
                compressed.append(token)
            else:
                prev_newline = False
                compressed.append(token)
        if compressed and compressed[0].type == 'NEWLINE':
            compressed.pop(0)
        return compressed
