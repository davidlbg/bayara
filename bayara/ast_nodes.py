from dataclasses import dataclass, field
from typing import List


@dataclass
class Node:
    line_no: int


@dataclass
class Program(Node):
    statements: List[Node] = field(default_factory=list)


@dataclass
class DatasetStmt(Node):
    name: str
    path: str


@dataclass
class ShowStmt(Node):
    name: str


@dataclass
class DescribeStmt(Node):
    name: str


@dataclass
class ColumnsStmt(Node):
    name: str


@dataclass
class ShapeStmt(Node):
    name: str


@dataclass
class TargetStmt(Node):
    dataset: str
    column: str


@dataclass
class FeaturesStmt(Node):
    dataset: str
    columns: List[str]


@dataclass
class SplitStmt(Node):
    dataset: str
    test_size: float


@dataclass
class ModelStmt(Node):
    name: str
    model_type: str


@dataclass
class TrainStmt(Node):
    model: str
    dataset: str


@dataclass
class EvaluateStmt(Node):
    model: str
    metrics: List[str]


@dataclass
class SaveStmt(Node):
    model: str
    path: str


@dataclass
class ExportStmt(Node):
    dataset: str
    path: str


@dataclass
class PredictStmt(Node):
    model: str
    dataset: str


@dataclass
class PrepareCommand(Node):
    command: str
    columns: List[str] = field(default_factory=list)


@dataclass
class PrepareStmt(Node):
    dataset: str
    commands: List[PrepareCommand] = field(default_factory=list)
