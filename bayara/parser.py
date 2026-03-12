import re
from typing import List

from .ast_nodes import (
    ColumnsStmt,
    DatasetStmt,
    DescribeStmt,
    EvaluateStmt,
    ExportStmt,
    FeaturesStmt,
    ModelStmt,
    PredictStmt,
    PrepareCommand,
    PrepareStmt,
    Program,
    SaveStmt,
    ShapeStmt,
    ShowStmt,
    SplitStmt,
    TargetStmt,
    TrainStmt,
)
from .errors import BayaraSyntaxError
from .lexer import SourceLine, lex_source


DATASET_RE = re.compile(r'^dataset\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+from\s+"(?P<path>.+)"$')
SIMPLE_NAME_RE = re.compile(r'^(show|describe|columns|shape)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)$')
TARGET_RE = re.compile(r'^target\s+(?P<dataset>[A-Za-z_][A-Za-z0-9_]*)\s*->\s*(?P<column>[A-Za-z_][A-Za-z0-9_]*)$')
FEATURES_RE = re.compile(r'^features\s+(?P<dataset>[A-Za-z_][A-Za-z0-9_]*)\s*->\s*(?P<columns>.+)$')
SPLIT_RE = re.compile(r'^split\s+(?P<dataset>[A-Za-z_][A-Za-z0-9_]*)\s+test\s+(?P<test_size>0(?:\.\d+)?|1(?:\.0+)?)$')
MODEL_RE = re.compile(r'^model\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s+as\s+(?P<model_type>[A-Za-z_][A-Za-z0-9_]*)$')
TRAIN_RE = re.compile(r'^train\s+(?P<model>[A-Za-z_][A-Za-z0-9_]*)\s+with\s+(?P<dataset>[A-Za-z_][A-Za-z0-9_]*)$')
EVAL_RE = re.compile(r'^evaluate\s+(?P<model>[A-Za-z_][A-Za-z0-9_]*)\s+with\s+(?P<metrics>.+)$')
SAVE_RE = re.compile(r'^save\s+(?P<model>[A-Za-z_][A-Za-z0-9_]*)\s+to\s+"(?P<path>.+)"$')
EXPORT_RE = re.compile(r'^export\s+(?P<dataset>[A-Za-z_][A-Za-z0-9_]*)\s+to\s+"(?P<path>.+)"$')
PREDICT_RE = re.compile(r'^predict\s+(?P<model>[A-Za-z_][A-Za-z0-9_]*)\s+on\s+(?P<dataset>[A-Za-z_][A-Za-z0-9_]*)$')
PREPARE_OPEN_RE = re.compile(r'^prepare\s+(?P<dataset>[A-Za-z_][A-Za-z0-9_]*)\s*\{$')


VALID_PREPARE_COMMANDS = {"drop nulls", "onehot", "standardize", "normalize"}


def parse_identifier_list(value: str, line_no: int) -> List[str]:
    parts = [item.strip() for item in value.split(',') if item.strip()]
    if not parts:
        raise BayaraSyntaxError(line_no, 'expected at least one identifier')
    for part in parts:
        if not re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', part):
            raise BayaraSyntaxError(line_no, f"invalid identifier '{part}'")
    return parts


def parse_prepare_command(source_line: SourceLine) -> PrepareCommand:
    text = source_line.text
    if text == 'drop nulls':
        return PrepareCommand(line_no=source_line.line_no, command='drop nulls')

    for prefix in ('onehot ', 'standardize ', 'normalize '):
        if text.startswith(prefix):
            cols = parse_identifier_list(text[len(prefix):], source_line.line_no)
            return PrepareCommand(line_no=source_line.line_no, command=prefix.strip(), columns=cols)

    raise BayaraSyntaxError(source_line.line_no, f"invalid prepare command: '{text}'")


def parse_source(source: str) -> Program:
    lines = lex_source(source)
    statements = []
    i = 0

    while i < len(lines):
        line = lines[i]
        text = line.text

        m = DATASET_RE.match(text)
        if m:
            statements.append(DatasetStmt(line_no=line.line_no, name=m.group('name'), path=m.group('path')))
            i += 1
            continue

        m = SIMPLE_NAME_RE.match(text)
        if m:
            cmd = text.split()[0]
            name = m.group('name')
            stmt_map = {
                'show': ShowStmt,
                'describe': DescribeStmt,
                'columns': ColumnsStmt,
                'shape': ShapeStmt,
            }
            statements.append(stmt_map[cmd](line_no=line.line_no, name=name))
            i += 1
            continue

        m = TARGET_RE.match(text)
        if m:
            statements.append(TargetStmt(line_no=line.line_no, dataset=m.group('dataset'), column=m.group('column')))
            i += 1
            continue

        m = FEATURES_RE.match(text)
        if m:
            columns = parse_identifier_list(m.group('columns'), line.line_no)
            statements.append(FeaturesStmt(line_no=line.line_no, dataset=m.group('dataset'), columns=columns))
            i += 1
            continue

        m = SPLIT_RE.match(text)
        if m:
            statements.append(SplitStmt(line_no=line.line_no, dataset=m.group('dataset'), test_size=float(m.group('test_size'))))
            i += 1
            continue

        m = MODEL_RE.match(text)
        if m:
            statements.append(ModelStmt(line_no=line.line_no, name=m.group('name'), model_type=m.group('model_type')))
            i += 1
            continue

        m = TRAIN_RE.match(text)
        if m:
            statements.append(TrainStmt(line_no=line.line_no, model=m.group('model'), dataset=m.group('dataset')))
            i += 1
            continue

        m = EVAL_RE.match(text)
        if m:
            metrics = parse_identifier_list(m.group('metrics'), line.line_no)
            statements.append(EvaluateStmt(line_no=line.line_no, model=m.group('model'), metrics=metrics))
            i += 1
            continue

        m = SAVE_RE.match(text)
        if m:
            statements.append(SaveStmt(line_no=line.line_no, model=m.group('model'), path=m.group('path')))
            i += 1
            continue

        m = EXPORT_RE.match(text)
        if m:
            statements.append(ExportStmt(line_no=line.line_no, dataset=m.group('dataset'), path=m.group('path')))
            i += 1
            continue

        m = PREDICT_RE.match(text)
        if m:
            statements.append(PredictStmt(line_no=line.line_no, model=m.group('model'), dataset=m.group('dataset')))
            i += 1
            continue

        m = PREPARE_OPEN_RE.match(text)
        if m:
            dataset = m.group('dataset')
            commands = []
            i += 1
            while i < len(lines) and lines[i].text != '}':
                commands.append(parse_prepare_command(lines[i]))
                i += 1
            if i >= len(lines):
                raise BayaraSyntaxError(line.line_no, "prepare block was not closed with '}'")
            statements.append(PrepareStmt(line_no=line.line_no, dataset=dataset, commands=commands))
            i += 1
            continue

        if text == '}':
            raise BayaraSyntaxError(line.line_no, "unexpected closing '}'")

        raise BayaraSyntaxError(line.line_no, f"could not parse statement: '{text}'")

    return Program(line_no=1, statements=statements)
