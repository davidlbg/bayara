from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set

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
from .errors import BayaraSemanticError


CLASSIFICATION_MODELS = {
    'random_forest': 'RandomForestClassifier(random_state=42)',
    'logistic_regression': 'LogisticRegression(max_iter=1000, random_state=42)',
    'decision_tree': 'DecisionTreeClassifier(random_state=42)',
    'knn': 'KNeighborsClassifier()',
    'naive_bayes': 'GaussianNB()',
}
REGRESSION_MODELS = {
    'linear_regression': 'LinearRegression()',
}
ALL_MODELS = {**CLASSIFICATION_MODELS, **REGRESSION_MODELS}
CLASSIFICATION_METRICS = {'accuracy', 'precision', 'recall', 'f1'}
REGRESSION_METRICS = {'mae', 'mse', 'r2'}


@dataclass
class DatasetState:
    target: Optional[str] = None
    features: Optional[List[str]] = None
    split_defined: bool = False
    split_names: Dict[str, str] = field(default_factory=dict)


@dataclass
class CompileContext:
    datasets: Dict[str, DatasetState] = field(default_factory=dict)
    models: Dict[str, str] = field(default_factory=dict)
    trained_models: Set[str] = field(default_factory=set)


HEADER_LINES = [
    'import os',
    'import pandas as pd',
    'import joblib',
    'from sklearn.model_selection import train_test_split',
    'from sklearn.preprocessing import StandardScaler, MinMaxScaler',
    'from sklearn.ensemble import RandomForestClassifier',
    'from sklearn.linear_model import LogisticRegression, LinearRegression',
    'from sklearn.tree import DecisionTreeClassifier',
    'from sklearn.neighbors import KNeighborsClassifier',
    'from sklearn.naive_bayes import GaussianNB',
    'from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score',
    '',
]


def ensure_dataset_exists(ctx: CompileContext, name: str, line_no: int) -> DatasetState:
    if name not in ctx.datasets:
        raise BayaraSemanticError(line_no, f"dataset '{name}' was not defined")
    return ctx.datasets[name]


def transpile_program(program: Program) -> str:
    ctx = CompileContext()
    python_lines: List[str] = HEADER_LINES.copy()

    for stmt in program.statements:
        if isinstance(stmt, DatasetStmt):
            ctx.datasets[stmt.name] = DatasetState()
            python_lines.extend(_transpile_dataset(stmt))
        elif isinstance(stmt, ShowStmt):
            ensure_dataset_exists(ctx, stmt.name, stmt.line_no)
            python_lines.append(f'print({stmt.name}.head())')
        elif isinstance(stmt, DescribeStmt):
            ensure_dataset_exists(ctx, stmt.name, stmt.line_no)
            python_lines.append(f'print({stmt.name}.describe(include="all"))')
        elif isinstance(stmt, ColumnsStmt):
            ensure_dataset_exists(ctx, stmt.name, stmt.line_no)
            python_lines.append(f'print(list({stmt.name}.columns))')
        elif isinstance(stmt, ShapeStmt):
            ensure_dataset_exists(ctx, stmt.name, stmt.line_no)
            python_lines.append(f'print({stmt.name}.shape)')
        elif isinstance(stmt, TargetStmt):
            state = ensure_dataset_exists(ctx, stmt.dataset, stmt.line_no)
            state.target = stmt.column
        elif isinstance(stmt, FeaturesStmt):
            state = ensure_dataset_exists(ctx, stmt.dataset, stmt.line_no)
            state.features = stmt.columns
        elif isinstance(stmt, PrepareStmt):
            ensure_dataset_exists(ctx, stmt.dataset, stmt.line_no)
            python_lines.extend(_transpile_prepare(stmt))
        elif isinstance(stmt, SplitStmt):
            state = ensure_dataset_exists(ctx, stmt.dataset, stmt.line_no)
            if not state.target:
                raise BayaraSemanticError(stmt.line_no, f"target was not defined for dataset '{stmt.dataset}'")
            if not state.features:
                raise BayaraSemanticError(stmt.line_no, f"features were not defined for dataset '{stmt.dataset}'")
            state.split_defined = True
            state.split_names = {
                'X_train': f'{stmt.dataset}_X_train',
                'X_test': f'{stmt.dataset}_X_test',
                'y_train': f'{stmt.dataset}_y_train',
                'y_test': f'{stmt.dataset}_y_test',
            }
            python_lines.extend(_transpile_split(stmt, state))
        elif isinstance(stmt, ModelStmt):
            if stmt.model_type not in ALL_MODELS:
                raise BayaraSemanticError(stmt.line_no, f"model '{stmt.model_type}' is not supported in Bayara 1.0")
            ctx.models[stmt.name] = stmt.model_type
            python_lines.append(f'{stmt.name} = {ALL_MODELS[stmt.model_type]}')
        elif isinstance(stmt, TrainStmt):
            if stmt.model not in ctx.models:
                raise BayaraSemanticError(stmt.line_no, f"model '{stmt.model}' was not defined")
            state = ensure_dataset_exists(ctx, stmt.dataset, stmt.line_no)
            if not state.split_defined:
                raise BayaraSemanticError(stmt.line_no, f"dataset '{stmt.dataset}' must be split before training")
            split = state.split_names
            python_lines.append(f"{stmt.model}.fit({split['X_train']}, {split['y_train']})")
            ctx.trained_models.add(stmt.model)
        elif isinstance(stmt, EvaluateStmt):
            if stmt.model not in ctx.models:
                raise BayaraSemanticError(stmt.line_no, f"model '{stmt.model}' was not defined")
            if stmt.model not in ctx.trained_models:
                raise BayaraSemanticError(stmt.line_no, f"model '{stmt.model}' must be trained before evaluation")
            dataset_name = _find_dataset_for_model(program, stmt.model, stmt.line_no)
            state = ensure_dataset_exists(ctx, dataset_name, stmt.line_no)
            python_lines.extend(_transpile_evaluate(stmt, state, ctx.models[stmt.model]))
        elif isinstance(stmt, SaveStmt):
            if stmt.model not in ctx.models:
                raise BayaraSemanticError(stmt.line_no, f"model '{stmt.model}' was not defined")
            python_lines.extend([
                f'os.makedirs(os.path.dirname(r"{stmt.path}") or ".", exist_ok=True)',
                f'joblib.dump({stmt.model}, r"{stmt.path}")',
                f'print("saved model to:", r"{stmt.path}")',
            ])
        elif isinstance(stmt, ExportStmt):
            ensure_dataset_exists(ctx, stmt.dataset, stmt.line_no)
            python_lines.extend([
                f'os.makedirs(os.path.dirname(r"{stmt.path}") or ".", exist_ok=True)',
                f'{stmt.dataset}.to_csv(r"{stmt.path}", index=False)',
                f'print("exported dataset to:", r"{stmt.path}")',
            ])
        elif isinstance(stmt, PredictStmt):
            if stmt.model not in ctx.models:
                raise BayaraSemanticError(stmt.line_no, f"model '{stmt.model}' was not defined")
            if stmt.model not in ctx.trained_models:
                raise BayaraSemanticError(stmt.line_no, f"model '{stmt.model}' must be trained before prediction")
            ensure_dataset_exists(ctx, stmt.dataset, stmt.line_no)
            python_lines.extend([
                f'predictions = {stmt.model}.predict({stmt.dataset})',
                'print("predictions:", list(predictions))',
            ])
        else:
            raise BayaraSemanticError(stmt.line_no, f'unsupported AST node: {type(stmt).__name__}')

    return '\n'.join(python_lines) + '\n'


def _transpile_dataset(stmt: DatasetStmt) -> List[str]:
    suffix = Path(stmt.path).suffix.lower()
    if suffix == '.csv':
        reader = 'pd.read_csv'
    elif suffix == '.json':
        reader = 'pd.read_json'
    elif suffix == '.parquet':
        reader = 'pd.read_parquet'
    else:
        raise BayaraSemanticError(stmt.line_no, f"unsupported dataset format '{suffix or 'unknown'}'")
    return [f'{stmt.name} = {reader}(r"{stmt.path}")']


def _transpile_prepare(stmt: PrepareStmt) -> List[str]:
    lines: List[str] = []
    for idx, command in enumerate(stmt.commands, start=1):
        if command.command == 'drop nulls':
            lines.append(f'{stmt.dataset} = {stmt.dataset}.dropna()')
        elif command.command == 'onehot':
            cols = repr(command.columns)
            lines.append(f'{stmt.dataset} = pd.get_dummies({stmt.dataset}, columns={cols})')
        elif command.command == 'standardize':
            scaler_name = f'_{stmt.dataset}_std_{idx}'
            cols = repr(command.columns)
            lines.append(f'{scaler_name} = StandardScaler()')
            lines.append(f'{stmt.dataset}[{cols}] = {scaler_name}.fit_transform({stmt.dataset}[{cols}])')
        elif command.command == 'normalize':
            scaler_name = f'_{stmt.dataset}_norm_{idx}'
            cols = repr(command.columns)
            lines.append(f'{scaler_name} = MinMaxScaler()')
            lines.append(f'{stmt.dataset}[{cols}] = {scaler_name}.fit_transform({stmt.dataset}[{cols}])')
        else:
            raise BayaraSemanticError(command.line_no, f"unknown prepare command '{command.command}'")
    return lines


def _transpile_split(stmt: SplitStmt, state: DatasetState) -> List[str]:
    features_repr = repr(state.features)
    target = state.target
    names = state.split_names
    return [
        f'{stmt.dataset}_X = {stmt.dataset}[{features_repr}]',
        f'{stmt.dataset}_y = {stmt.dataset}["{target}"]',
        f"{names['X_train']}, {names['X_test']}, {names['y_train']}, {names['y_test']} = train_test_split({stmt.dataset}_X, {stmt.dataset}_y, test_size={stmt.test_size}, random_state=42)",
    ]


def _transpile_evaluate(stmt: EvaluateStmt, state: DatasetState, model_type: str) -> List[str]:
    lines: List[str] = []
    split = state.split_names
    lines.append(f'{stmt.model}_preds = {stmt.model}.predict({split["X_test"]})')

    for metric in stmt.metrics:
        if metric in CLASSIFICATION_METRICS and model_type in REGRESSION_MODELS:
            raise BayaraSemanticError(stmt.line_no, f"metric '{metric}' is for classification, but model '{model_type}' is regression")
        if metric in REGRESSION_METRICS and model_type in CLASSIFICATION_MODELS:
            raise BayaraSemanticError(stmt.line_no, f"metric '{metric}' is for regression, but model '{model_type}' is classification")
        if metric not in CLASSIFICATION_METRICS and metric not in REGRESSION_METRICS:
            raise BayaraSemanticError(stmt.line_no, f"metric '{metric}' is not supported in Bayara 1.0")

        if metric == 'accuracy':
            lines.append(f'print("accuracy:", accuracy_score({split["y_test"]}, {stmt.model}_preds))')
        elif metric == 'precision':
            lines.append(f'print("precision:", precision_score({split["y_test"]}, {stmt.model}_preds, zero_division=0))')
        elif metric == 'recall':
            lines.append(f'print("recall:", recall_score({split["y_test"]}, {stmt.model}_preds, zero_division=0))')
        elif metric == 'f1':
            lines.append(f'print("f1:", f1_score({split["y_test"]}, {stmt.model}_preds, zero_division=0))')
        elif metric == 'mae':
            lines.append(f'print("mae:", mean_absolute_error({split["y_test"]}, {stmt.model}_preds))')
        elif metric == 'mse':
            lines.append(f'print("mse:", mean_squared_error({split["y_test"]}, {stmt.model}_preds))')
        elif metric == 'r2':
            lines.append(f'print("r2:", r2_score({split["y_test"]}, {stmt.model}_preds))')
    return lines


def _find_dataset_for_model(program: Program, model_name: str, line_no: int) -> str:
    for stmt in program.statements:
        if isinstance(stmt, TrainStmt) and stmt.model == model_name:
            return stmt.dataset
    raise BayaraSemanticError(line_no, f"could not infer dataset used by model '{model_name}'")
