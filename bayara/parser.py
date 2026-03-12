from .ast_nodes import (
    DatasetStmt,
    EvaluateStmt,
    ExportStmt,
    FeaturesStmt,
    ModelStmt,
    PrepareDropCmd,
    PrepareDropNullsCmd,
    PrepareFillNullsCmd,
    PrepareOneHotCmd,
    PrepareScaleCmd,
    PrepareStmt,
    Program,
    SaveStmt,
    SimpleDatasetStmt,
    SplitStmt,
    TargetStmt,
    TrainStmt,
)
from .errors import BayaraSyntaxError
from .tokens import Token


class Parser:
    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def current(self) -> Token:
        return self.tokens[self.pos]

    def peek(self, offset: int = 1) -> Token:
        idx = min(self.pos + offset, len(self.tokens) - 1)
        return self.tokens[idx]

    def advance(self) -> Token:
        tok = self.current()
        if tok.type != 'EOF':
            self.pos += 1
        return tok

    def match(self, *types: str) -> Token | None:
        if self.current().type in types:
            return self.advance()
        return None

    def expect(self, token_type: str, value: str | None = None, message: str | None = None) -> Token:
        tok = self.current()
        if tok.type != token_type or (value is not None and tok.value != value):
            if message is None:
                expected = value if value is not None else token_type
                message = f"expected {expected}"
            raise BayaraSyntaxError(message, tok.line, tok.column)
        return self.advance()

    def expect_keyword(self, value: str) -> Token:
        tok = self.current()
        if tok.type != 'IDENT' or tok.value != value:
            raise BayaraSyntaxError(f"expected '{value}'", tok.line, tok.column)
        return self.advance()

    def parse(self) -> Program:
        statements = []
        while self.current().type != 'EOF':
            while self.match('NEWLINE'):
                pass
            if self.current().type == 'EOF':
                break
            statements.append(self.parse_statement())
            while self.match('NEWLINE'):
                pass
        return Program(statements)

    def parse_statement(self):
        tok = self.current()
        if tok.type != 'IDENT':
            raise BayaraSyntaxError('expected statement', tok.line, tok.column)
        kw = tok.value
        if kw == 'dataset':
            return self.parse_dataset()
        if kw in {'show', 'describe', 'columns', 'shape', 'inspect'}:
            return self.parse_simple_dataset_stmt()
        if kw == 'prepare':
            return self.parse_prepare()
        if kw == 'target':
            return self.parse_target()
        if kw == 'features':
            return self.parse_features()
        if kw == 'split':
            return self.parse_split()
        if kw == 'model':
            return self.parse_model()
        if kw == 'train':
            return self.parse_train()
        if kw == 'evaluate':
            return self.parse_evaluate()
        if kw == 'save':
            return self.parse_save()
        if kw == 'export':
            return self.parse_export()
        raise BayaraSyntaxError(f"unknown statement '{kw}'", tok.line, tok.column)

    def parse_dataset(self) -> DatasetStmt:
        start = self.expect_keyword('dataset')
        name = self.expect('IDENT', message='expected dataset name').value
        self.expect_keyword('from')
        path = self.expect('STRING', message='expected CSV path string').value
        return DatasetStmt(name=name, path=path, line=start.line)

    def parse_simple_dataset_stmt(self) -> SimpleDatasetStmt:
        start = self.advance()
        dataset = self.expect('IDENT', message='expected dataset name').value
        return SimpleDatasetStmt(command=start.value, dataset=dataset, line=start.line)

    def parse_prepare(self) -> PrepareStmt:
        start = self.expect_keyword('prepare')
        dataset = self.expect('IDENT', message='expected dataset name').value
        self.expect('LBRACE', message="expected '{' to open prepare block")
        while self.match('NEWLINE'):
            pass
        commands = []
        while self.current().type != 'RBRACE':
            commands.append(self.parse_prepare_command())
            while self.match('NEWLINE'):
                pass
            if self.current().type == 'EOF':
                raise BayaraSyntaxError("expected '}' to close prepare block", self.current().line, self.current().column)
        self.expect('RBRACE')
        return PrepareStmt(dataset=dataset, commands=commands, line=start.line)

    def parse_prepare_command(self):
        tok = self.current()
        if tok.type != 'IDENT':
            raise BayaraSyntaxError('expected prepare command', tok.line, tok.column)
        if tok.value == 'drop':
            start = self.advance()
            if self.current().type == 'IDENT' and self.current().value == 'nulls':
                self.advance()
                return PrepareDropNullsCmd(line=start.line)
            columns = self.parse_ident_list()
            return PrepareDropCmd(columns=columns, line=start.line)
        if tok.value == 'fill':
            start = self.advance()
            self.expect_keyword('nulls')
            column = self.expect('IDENT', message='expected column after fill nulls').value
            self.expect_keyword('with')
            strategy_token = self.current()
            if strategy_token.type == 'IDENT' and strategy_token.value in {'mean', 'median', 'mode'}:
                strategy = self.advance().value
            elif strategy_token.type == 'STRING':
                strategy = self.advance().value
            elif strategy_token.type == 'NUMBER':
                raw = self.advance().value
                strategy = float(raw) if '.' in raw else int(raw)
            else:
                raise BayaraSyntaxError("expected mean, median, mode, string or number after 'with'", strategy_token.line, strategy_token.column)
            return PrepareFillNullsCmd(column=column, strategy=strategy, line=start.line)
        if tok.value == 'onehot':
            start = self.advance()
            columns = self.parse_ident_list()
            return PrepareOneHotCmd(columns=columns, line=start.line)
        if tok.value in {'standardize', 'normalize'}:
            start = self.advance()
            columns = self.parse_ident_list()
            return PrepareScaleCmd(kind=start.value, columns=columns, line=start.line)
        raise BayaraSyntaxError(f"unknown prepare command '{tok.value}'", tok.line, tok.column)

    def parse_target(self) -> TargetStmt:
        start = self.expect_keyword('target')
        dataset = self.expect('IDENT', message='expected dataset name').value
        self.expect('ARROW', message="expected '->' after dataset in target statement")
        column = self.expect('IDENT', message='expected target column').value
        return TargetStmt(dataset=dataset, column=column, line=start.line)

    def parse_features(self) -> FeaturesStmt:
        start = self.expect_keyword('features')
        dataset = self.expect('IDENT', message='expected dataset name').value
        self.expect('ARROW', message="expected '->' after dataset in features statement")
        columns = self.parse_ident_list()
        return FeaturesStmt(dataset=dataset, columns=columns, line=start.line)

    def parse_split(self) -> SplitStmt:
        start = self.expect_keyword('split')
        dataset = self.expect('IDENT', message='expected dataset name').value
        self.expect_keyword('test')
        number = self.expect('NUMBER', message="expected test size number after 'test'")
        return SplitStmt(dataset=dataset, test_size=float(number.value), line=start.line)

    def parse_model(self) -> ModelStmt:
        start = self.expect_keyword('model')
        name = self.expect('IDENT', message='expected model name').value
        self.expect_keyword('as')
        model_type = self.expect('IDENT', message='expected model type').value
        return ModelStmt(name=name, model_type=model_type, line=start.line)

    def parse_train(self) -> TrainStmt:
        start = self.expect_keyword('train')
        model = self.expect('IDENT', message='expected model name').value
        self.expect_keyword('with')
        dataset = self.expect('IDENT', message='expected dataset name').value
        return TrainStmt(model=model, dataset=dataset, line=start.line)

    def parse_evaluate(self) -> EvaluateStmt:
        start = self.expect_keyword('evaluate')
        model = self.expect('IDENT', message='expected model name').value
        self.expect_keyword('with')
        metrics = self.parse_ident_list()
        return EvaluateStmt(model=model, metrics=metrics, line=start.line)

    def parse_save(self) -> SaveStmt:
        start = self.expect_keyword('save')
        model = self.expect('IDENT', message='expected model name').value
        self.expect_keyword('to')
        path = self.expect('STRING', message='expected output path string').value
        return SaveStmt(model=model, path=path, line=start.line)

    def parse_export(self) -> ExportStmt:
        start = self.expect_keyword('export')
        dataset = self.expect('IDENT', message='expected dataset name').value
        self.expect_keyword('to')
        path = self.expect('STRING', message='expected output path string').value
        return ExportStmt(dataset=dataset, path=path, line=start.line)

    def parse_ident_list(self) -> list[str]:
        items = [self.expect('IDENT', message='expected identifier').value]
        while self.match('COMMA'):
            items.append(self.expect('IDENT', message='expected identifier after comma').value)
        return items
