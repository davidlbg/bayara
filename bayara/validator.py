from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .ast_nodes import (
    DatasetStmt,
    EvaluateStmt,
    ExportStmt,
    FeaturesStmt,
    ModelStmt,
    PrepareDropCmd,
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
from .errors import BayaraSemanticError


SUPPORTED_MODELS = {
    'random_forest',
    'logistic_regression',
    'decision_tree',
    'knn',
    'naive_bayes',
    'linear_regression',
}

SUPPORTED_METRICS = {
    'accuracy', 'precision', 'recall', 'f1', 'mae', 'mse', 'r2'
}


@dataclass
class DatasetState:
    path: str
    sample_df: pd.DataFrame | None = None
    target: str | None = None
    features: list[str] | None = None
    split: float | None = None
    prepare_commands: list[Any] = field(default_factory=list)


@dataclass
class ValidationContext:
    datasets: dict[str, DatasetState] = field(default_factory=dict)
    models: dict[str, str] = field(default_factory=dict)
    model_dataset: dict[str, str] = field(default_factory=dict)


def _load_sample_csv(path: str, line: int) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise BayaraSemanticError(f"CSV file '{path}' does not exist", line)
    try:
        return pd.read_csv(p, nrows=50)
    except Exception as exc:
        raise BayaraSemanticError(f"could not read CSV '{path}': {exc}", line) from exc


def _ensure_columns_exist(df: pd.DataFrame, dataset: str, columns: list[str], line: int) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise BayaraSemanticError(
            f"dataset '{dataset}' is missing column(s): {', '.join(missing)}", line
        )


def _apply_prepare_sample(df: pd.DataFrame, dataset: str, prepare_stmt: PrepareStmt) -> pd.DataFrame:
    result = df.copy()
    for cmd in prepare_stmt.commands:
        if isinstance(cmd, PrepareDropCmd):
            _ensure_columns_exist(result, dataset, cmd.columns, cmd.line)
            result = result.drop(columns=cmd.columns)
        elif isinstance(cmd, PrepareFillNullsCmd):
            _ensure_columns_exist(result, dataset, [cmd.column], cmd.line)
            if cmd.strategy == 'mean':
                result[cmd.column] = result[cmd.column].fillna(result[cmd.column].mean())
            elif cmd.strategy == 'median':
                result[cmd.column] = result[cmd.column].fillna(result[cmd.column].median())
            elif cmd.strategy == 'mode':
                mode = result[cmd.column].mode(dropna=True)
                fill = mode.iloc[0] if not mode.empty else None
                result[cmd.column] = result[cmd.column].fillna(fill)
            else:
                result[cmd.column] = result[cmd.column].fillna(cmd.strategy)
        elif isinstance(cmd, PrepareOneHotCmd):
            _ensure_columns_exist(result, dataset, cmd.columns, cmd.line)
            result = pd.get_dummies(result, columns=cmd.columns)
        elif isinstance(cmd, PrepareScaleCmd):
            _ensure_columns_exist(result, dataset, cmd.columns, cmd.line)
        else:
            raise BayaraSemanticError('unknown prepare command during validation', prepare_stmt.line)
    return result


def validate(program: Program) -> ValidationContext:
    ctx = ValidationContext()

    for stmt in program.statements:
        if isinstance(stmt, DatasetStmt):
            if stmt.name in ctx.datasets:
                raise BayaraSemanticError(f"dataset '{stmt.name}' already defined", stmt.line)
            sample_df = _load_sample_csv(stmt.path, stmt.line)
            ctx.datasets[stmt.name] = DatasetState(path=stmt.path, sample_df=sample_df)

        elif isinstance(stmt, SimpleDatasetStmt):
            if stmt.dataset not in ctx.datasets:
                raise BayaraSemanticError(f"dataset '{stmt.dataset}' was not defined", stmt.line)

        elif isinstance(stmt, PrepareStmt):
            if stmt.dataset not in ctx.datasets:
                raise BayaraSemanticError(f"dataset '{stmt.dataset}' was not defined", stmt.line)
            ds = ctx.datasets[stmt.dataset]
            ds.sample_df = _apply_prepare_sample(ds.sample_df, stmt.dataset, stmt)
            ds.prepare_commands.extend(stmt.commands)

        elif isinstance(stmt, TargetStmt):
            if stmt.dataset not in ctx.datasets:
                raise BayaraSemanticError(f"dataset '{stmt.dataset}' was not defined", stmt.line)
            ds = ctx.datasets[stmt.dataset]
            _ensure_columns_exist(ds.sample_df, stmt.dataset, [stmt.column], stmt.line)
            ds.target = stmt.column

        elif isinstance(stmt, FeaturesStmt):
            if stmt.dataset not in ctx.datasets:
                raise BayaraSemanticError(f"dataset '{stmt.dataset}' was not defined", stmt.line)
            ds = ctx.datasets[stmt.dataset]
            _ensure_columns_exist(ds.sample_df, stmt.dataset, stmt.columns, stmt.line)
            ds.features = stmt.columns

        elif isinstance(stmt, SplitStmt):
            if stmt.dataset not in ctx.datasets:
                raise BayaraSemanticError(f"dataset '{stmt.dataset}' was not defined", stmt.line)
            if not (0 < stmt.test_size < 1):
                raise BayaraSemanticError('split test size must be between 0 and 1', stmt.line)
            ctx.datasets[stmt.dataset].split = stmt.test_size

        elif isinstance(stmt, ModelStmt):
            if stmt.name in ctx.models:
                raise BayaraSemanticError(f"model '{stmt.name}' already defined", stmt.line)
            if stmt.model_type not in SUPPORTED_MODELS:
                raise BayaraSemanticError(f"unsupported model '{stmt.model_type}'", stmt.line)
            ctx.models[stmt.name] = stmt.model_type

        elif isinstance(stmt, TrainStmt):
            if stmt.model not in ctx.models:
                raise BayaraSemanticError(f"model '{stmt.model}' was not defined", stmt.line)
            if stmt.dataset not in ctx.datasets:
                raise BayaraSemanticError(f"dataset '{stmt.dataset}' was not defined", stmt.line)
            ds = ctx.datasets[stmt.dataset]
            if ds.target is None:
                raise BayaraSemanticError(f"dataset '{stmt.dataset}' has no target defined", stmt.line)
            if ds.features is None:
                raise BayaraSemanticError(f"dataset '{stmt.dataset}' has no features defined", stmt.line)
            ctx.model_dataset[stmt.model] = stmt.dataset

        elif isinstance(stmt, EvaluateStmt):
            if stmt.model not in ctx.models:
                raise BayaraSemanticError(f"model '{stmt.model}' was not defined", stmt.line)
            if stmt.model not in ctx.model_dataset:
                raise BayaraSemanticError(f"model '{stmt.model}' has not been trained", stmt.line)
            unsupported = [m for m in stmt.metrics if m not in SUPPORTED_METRICS]
            if unsupported:
                raise BayaraSemanticError(f"unsupported metric(s): {', '.join(unsupported)}", stmt.line)

        elif isinstance(stmt, SaveStmt):
            if stmt.model not in ctx.models:
                raise BayaraSemanticError(f"model '{stmt.model}' was not defined", stmt.line)

        elif isinstance(stmt, ExportStmt):
            if stmt.dataset not in ctx.datasets:
                raise BayaraSemanticError(f"dataset '{stmt.dataset}' was not defined", stmt.line)

    return ctx
