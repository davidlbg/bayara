# Bayara 1.0.0

Bayara is a DSL for tabular data analysis and classical machine learning pipelines.

It compiles concise `.bay` scripts into Python powered by **pandas** and **scikit-learn**.

## What Bayara 1.0 supports

### Data
- `dataset <name> from "<path>"`
- `show <dataset>`
- `describe <dataset>`
- `columns <dataset>`
- `shape <dataset>`

### Preparation
```bayara
prepare churn {
    drop nulls
    onehot geography, gender
    standardize age, balance, salary
    normalize age, balance, salary
}
```

Supported prepare commands:
- `drop nulls`
- `onehot <col1>, <col2>, ...`
- `standardize <col1>, <col2>, ...`
- `normalize <col1>, <col2>, ...`

### ML pipeline
- `target <dataset> -> <column>`
- `features <dataset> -> <col1>, <col2>, ...`
- `split <dataset> test <number>`
- `model <name> as <model_type>`
- `train <model> with <dataset>`
- `evaluate <model> with <metric1>, <metric2>, ...`
- `save <model> to "<path>"`
- `export <dataset> to "<path>"`
- `predict <model> on <dataset>`

### Models
- `random_forest`
- `logistic_regression`
- `decision_tree`
- `knn`
- `naive_bayes`
- `linear_regression`

### Metrics
Classification:
- `accuracy`
- `precision`
- `recall`
- `f1`

Regression:
- `mae`
- `mse`
- `r2`

## CLI

```bash
python bayara.py compile examples/basic_classification.bay output.py
python bayara.py run examples/basic_classification.bay
python bayara.py check examples/basic_classification.bay
python bayara.py version
```

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Example

```bayara
dataset churn from "data/churn.csv"

shape churn
columns churn
show churn
describe churn

prepare churn {
    drop nulls
    standardize age, balance, salary
}

target churn -> exited
features churn -> age, balance, salary
split churn test 0.2

model clf as random_forest
train clf with churn

evaluate clf with accuracy, precision, recall, f1
save clf to "models/churn_rf.pkl"
```

Run it:

```bash
python bayara.py run examples/basic_classification.bay
```

## Project structure

```text
Bayara/
├── bayara.py
├── bayara/
│   ├── __init__.py
│   ├── ast_nodes.py
│   ├── cli.py
│   ├── errors.py
│   ├── lexer.py
│   ├── parser.py
│   └── transpiler.py
├── data/
├── examples/
├── tests/
├── requirements.txt
├── LICENSE
└── .gitignore
```

## Notes

- Bayara 1.0 is intentionally small.
- The parser is statement-based and line-oriented, with explicit syntax errors and semantic validation.
- This release is meant to be stable enough to publish as a first GitHub version.

## License

MIT
