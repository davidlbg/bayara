from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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
from .validator import validate


MODEL_IMPORTS = {
    'random_forest': ('from sklearn.ensemble import RandomForestClassifier', 'RandomForestClassifier(random_state=42)'),
    'logistic_regression': ('from sklearn.linear_model import LogisticRegression', 'LogisticRegression(max_iter=1000, random_state=42)'),
    'decision_tree': ('from sklearn.tree import DecisionTreeClassifier', 'DecisionTreeClassifier(random_state=42)'),
    'knn': ('from sklearn.neighbors import KNeighborsClassifier', 'KNeighborsClassifier()'),
    'naive_bayes': ('from sklearn.naive_bayes import GaussianNB', 'GaussianNB()'),
    'linear_regression': ('from sklearn.linear_model import LinearRegression', 'LinearRegression()'),
}

METRIC_IMPORTS = {
    'accuracy': 'accuracy_score',
    'precision': 'precision_score',
    'recall': 'recall_score',
    'f1': 'f1_score',
    'mae': 'mean_absolute_error',
    'mse': 'mean_squared_error',
    'r2': 'r2_score',
}


@dataclass
class DatasetRuntimeState:
    path: str
    target: str | None = None
    features: list[str] | None = None
    split: float | None = None
    pre_split_lines: list[str] = field(default_factory=list)
    post_split_scalers: list[tuple[str, list[str]]] = field(default_factory=list)
    split_emitted: bool = False


@dataclass
class ModelRuntimeState:
    model_type: str
    dataset: str | None = None
    trained: bool = False


class Transpiler:
    def __init__(self, program: Program):
        self.program = program
        self.validation = validate(program)
        self.datasets: dict[str, DatasetRuntimeState] = {}
        self.models: dict[str, ModelRuntimeState] = {}
        self.lines: list[str] = []
        self.imports: set[str] = {
            'import pandas as pd',
            'import joblib',
            'from sklearn.model_selection import train_test_split',
            'from sklearn.preprocessing import StandardScaler, MinMaxScaler',
        }
        self.metric_imports: set[str] = set()
        self.auto_split_noted: set[str] = set()

    def transpile(self) -> str:
        for stmt in self.program.statements:
            self.emit_statement(stmt)
        header = sorted(self.imports)
        if self.metric_imports:
            header.append(f"from sklearn.metrics import {', '.join(sorted(self.metric_imports))}")
        header.append('')
        return '\n'.join(header + self.lines) + '\n'

    def emit_statement(self, stmt: Any) -> None:
        if isinstance(stmt, DatasetStmt):
            self.datasets[stmt.name] = DatasetRuntimeState(path=stmt.path)
            self.lines.append(f'{stmt.name} = pd.read_csv(r"{stmt.path}")')
            return

        if isinstance(stmt, SimpleDatasetStmt):
            if stmt.command == 'show':
                self.lines.append(f'print({stmt.dataset}.head())')
            elif stmt.command == 'describe':
                self.lines.append(f'print({stmt.dataset}.describe(include="all"))')
            elif stmt.command == 'columns':
                self.lines.append(f'print(list({stmt.dataset}.columns))')
            elif stmt.command == 'shape':
                self.lines.append(f'print({stmt.dataset}.shape)')
            elif stmt.command == 'inspect':
                self.lines.extend([
                    f'print("shape:", {stmt.dataset}.shape)',
                    f'print("columns:", list({stmt.dataset}.columns))',
                    'print("dtypes:")',
                    f'print({stmt.dataset}.dtypes)',
                    'print("missing values:")',
                    f'print({stmt.dataset}.isnull().sum())',
                ])
            return

        if isinstance(stmt, PrepareStmt):
            ds = self.datasets[stmt.dataset]
            for cmd in stmt.commands:
                if isinstance(cmd, PrepareDropCmd):
                    ds.pre_split_lines.append(f'{stmt.dataset} = {stmt.dataset}.drop(columns={cmd.columns!r})')
                elif isinstance(cmd, PrepareFillNullsCmd):
                    strat = cmd.strategy
                    if strat == 'mean':
                        ds.pre_split_lines.append(
                            f'{stmt.dataset}["{cmd.column}"] = {stmt.dataset}["{cmd.column}"].fillna({stmt.dataset}["{cmd.column}"].mean())'
                        )
                    elif strat == 'median':
                        ds.pre_split_lines.append(
                            f'{stmt.dataset}["{cmd.column}"] = {stmt.dataset}["{cmd.column}"].fillna({stmt.dataset}["{cmd.column}"].median())'
                        )
                    elif strat == 'mode':
                        mode_var = f'_mode_value_{stmt.dataset}_{cmd.column}'
                        ds.pre_split_lines.append(
                            f'{mode_var} = {stmt.dataset}["{cmd.column}"].mode(dropna=True).iloc[0] if not {stmt.dataset}["{cmd.column}"].mode(dropna=True).empty else None'
                        )
                        ds.pre_split_lines.append(
                            f'{stmt.dataset}["{cmd.column}"] = {stmt.dataset}["{cmd.column}"].fillna({mode_var})'
                        )
                    elif isinstance(strat, str):
                        ds.pre_split_lines.append(f'{stmt.dataset}["{cmd.column}"] = {stmt.dataset}["{cmd.column}"].fillna({strat!r})')
                    else:
                        ds.pre_split_lines.append(f'{stmt.dataset}["{cmd.column}"] = {stmt.dataset}["{cmd.column}"].fillna({strat})')
                elif isinstance(cmd, PrepareOneHotCmd):
                    ds.pre_split_lines.append(f'{stmt.dataset} = pd.get_dummies({stmt.dataset}, columns={cmd.columns!r})')
                elif isinstance(cmd, PrepareScaleCmd):
                    ds.post_split_scalers.append((cmd.kind, cmd.columns))
            return

        if isinstance(stmt, TargetStmt):
            self.datasets[stmt.dataset].target = stmt.column
            return

        if isinstance(stmt, FeaturesStmt):
            self.datasets[stmt.dataset].features = stmt.columns
            return

        if isinstance(stmt, SplitStmt):
            self.datasets[stmt.dataset].split = stmt.test_size
            self._emit_split_if_needed(stmt.dataset)
            return

        if isinstance(stmt, ModelStmt):
            import_stmt, constructor = MODEL_IMPORTS[stmt.model_type]
            self.imports.add(import_stmt)
            self.models[stmt.name] = ModelRuntimeState(model_type=stmt.model_type)
            self.lines.append(f'{stmt.name} = {constructor}')
            return

        if isinstance(stmt, TrainStmt):
            self._ensure_split_for_dataset(stmt.dataset)
            self.models[stmt.model].dataset = stmt.dataset
            self.models[stmt.model].trained = True
            self.lines.append(f'{stmt.model}.fit(X_train, y_train)')
            return

        if isinstance(stmt, EvaluateStmt):
            model_state = self.models[stmt.model]
            if model_state.dataset is None:
                raise BayaraSemanticError(f"model '{stmt.model}' has no associated dataset", stmt.line)
            self._ensure_split_for_dataset(model_state.dataset)
            self.lines.append(f'preds = {stmt.model}.predict(X_test)')
            for metric in stmt.metrics:
                self.metric_imports.add(METRIC_IMPORTS[metric])
                func = METRIC_IMPORTS[metric]
                if metric in {'precision', 'recall', 'f1'}:
                    self.lines.append(f'print("{metric}:", {func}(y_test, preds, zero_division=0))')
                else:
                    self.lines.append(f'print("{metric}:", {func}(y_test, preds))')
            return

        if isinstance(stmt, SaveStmt):
            self.imports.add('from pathlib import Path')
            self.lines.append(f'Path(r"{stmt.path}").parent.mkdir(parents=True, exist_ok=True)')
            self.lines.append(f'joblib.dump({stmt.model}, r"{stmt.path}")')
            return

        if isinstance(stmt, ExportStmt):
            self.imports.add('from pathlib import Path')
            self.lines.append(f'Path(r"{stmt.path}").parent.mkdir(parents=True, exist_ok=True)')
            self.lines.append(f'{stmt.dataset}.to_csv(r"{stmt.path}", index=False)')
            return

    def _ensure_split_for_dataset(self, dataset: str) -> None:
        ds = self.datasets[dataset]
        if ds.split is None:
            ds.split = 0.2
            if dataset not in self.auto_split_noted:
                self.lines.append(f'print("[Bayara] No split defined for {dataset}; using default test size 0.2")')
                self.auto_split_noted.add(dataset)
        self._emit_split_if_needed(dataset)

    def _emit_split_if_needed(self, dataset: str) -> None:
        ds = self.datasets[dataset]
        if ds.split_emitted:
            return
        for line in ds.pre_split_lines:
            self.lines.append(line)
        self.lines.append(f'X = {dataset}[{ds.features!r}]')
        self.lines.append(f'y = {dataset}["{ds.target}"]')
        self.lines.append(
            f'X_train, X_test, y_train, y_test = train_test_split(X, y, test_size={ds.split}, random_state=42)'
        )
        for index, (kind, columns) in enumerate(ds.post_split_scalers, start=1):
            scaler_name = f'_{kind}_scaler_{dataset}_{index}'
            scaler_class = 'StandardScaler' if kind == 'standardize' else 'MinMaxScaler'
            self.lines.append(f'{scaler_name} = {scaler_class}()')
            self.lines.append(f'X_train[{columns!r}] = {scaler_name}.fit_transform(X_train[{columns!r}])')
            self.lines.append(f'X_test[{columns!r}] = {scaler_name}.transform(X_test[{columns!r}])')
        ds.split_emitted = True


def transpile(program: Program) -> str:
    return Transpiler(program).transpile()
