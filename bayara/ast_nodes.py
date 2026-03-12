from dataclasses import dataclass
from typing import List, Union


@dataclass
class Program:
    statements: list


@dataclass
class DatasetStmt:
    name: str
    path: str
    line: int


@dataclass
class SimpleDatasetStmt:
    command: str
    dataset: str
    line: int


@dataclass
class PrepareDropCmd:
    columns: List[str]
    line: int


@dataclass
class PrepareDropNullsCmd:
    line: int


@dataclass
class PrepareFillNullsCmd:
    column: str
    strategy: Union[str, float, int]
    line: int


@dataclass
class PrepareOneHotCmd:
    columns: List[str]
    line: int


@dataclass
class PrepareScaleCmd:
    kind: str
    columns: List[str]
    line: int


@dataclass
class PrepareStmt:
    dataset: str
    commands: List[
        Union[
            PrepareDropCmd,
            PrepareDropNullsCmd,
            PrepareFillNullsCmd,
            PrepareOneHotCmd,
            PrepareScaleCmd,
        ]
    ]
    line: int


@dataclass
class TargetStmt:
    dataset: str
    column: str
    line: int


@dataclass
class FeaturesStmt:
    dataset: str
    columns: List[str]
    line: int


@dataclass
class SplitStmt:
    dataset: str
    test_size: float
    line: int


@dataclass
class ModelStmt:
    name: str
    model_type: str
    line: int


@dataclass
class TrainStmt:
    model: str
    dataset: str
    line: int


@dataclass
class EvaluateStmt:
    model: str
    metrics: List[str]
    line: int


@dataclass
class SaveStmt:
    model: str
    path: str
    line: int


@dataclass
class ExportStmt:
    dataset: str
    path: str
    line: int
