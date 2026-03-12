# Bayara

Bayara is a small DSL for tabular data and classical machine learning pipelines.

This package is a **1.0.1-oriented foundation update**. The main goals are:

- a real tokenizer
- a recursive descent parser
- clearer syntax and semantic errors
- safer preprocessing for scaling steps
- runtime column validation during `check`
- small ergonomic improvements such as multiline comments and default split

## What's new in 1.0.1

- Replaced line-regex parsing with **Lexer + Recursive Descent Parser**
- Added `inspect <dataset>`
- Added `fill nulls <column> with ...`
- Added `drop <col1>, <col2>` inside `prepare`
- Added `/* multiline comments */`
- Added automatic `split ... test 0.2` when omitted
- Fixed **data leakage** for `standardize` and `normalize`

## Commands supported

- `dataset <name> from "path.csv"`
- `show <dataset>`
- `describe <dataset>`
- `columns <dataset>`
- `shape <dataset>`
- `inspect <dataset>`
- `prepare <dataset> { ... }`
- `target <dataset> -> <column>`
- `features <dataset> -> <col1>, <col2>, ...`
- `split <dataset> test <number>`
- `model <name> as <model_type>`
- `train <model> with <dataset>`
- `evaluate <model> with <metric1>, <metric2>, ...`
- `save <model> to "path.pkl"`
- `export <dataset> to "path.csv"`

## Prepare block commands

- `drop <col1>, <col2>`
- `fill nulls <column> with mean|median|mode|"text"|0`
- `onehot <col1>, <col2>`
- `standardize <col1>, <col2>`
- `normalize <col1>, <col2>`

## Example

```bayara
/* Bayara 1.0.1 example */

dataset churn from "data/churn.csv"
inspect churn

prepare churn {
    fill nulls balance with median
    onehot segment
    standardize age, balance, salary
}

target churn -> exited
features churn -> age, balance, salary, segment_enterprise, segment_retail

model clf as logistic_regression
train clf with churn

evaluate clf with accuracy, precision, recall, f1
save clf to "models/churn.pkl"
```

## CLI

```bash
python bayara.py check examples/churn_101.bay
python bayara.py compile examples/churn_101.bay output.py
python bayara.py run examples/churn_101.bay output.py
python bayara.py version
```
