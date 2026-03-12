# Bayara

**Bayara** is a small domain-specific language (DSL) designed for
**tabular data analysis and classical machine learning pipelines**.

It allows users to write concise `.bay` scripts that are compiled into
Python code using **pandas** and **scikit-learn**.

Bayara is not meant to compete with large frameworks such as MLFlow or
full ML platforms.\
Instead, it aims to provide a **simple, readable way to define tabular
ML workflows**.

------------------------------------------------------------------------

# Project Status

Bayara is an **early experimental language (v1.0.1)**.

The current version focuses on:

-   tabular dataset workflows
-   classical machine learning models
-   simple preprocessing pipelines
-   an easy-to-read DSL syntax

Future improvements are planned, but the goal of this version is to
provide a **clear and functional minimal language**.

------------------------------------------------------------------------

# Example

Example Bayara script:

``` bayara
dataset churn from "data/churn.csv"

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
```

This script:

1.  Loads a dataset
2.  Prepares the data
3.  Defines target and features
4.  Splits the dataset
5.  Trains a model
6.  Evaluates the results

------------------------------------------------------------------------

# Installation

Clone the repository:

``` bash
git clone https://github.com/davidlbg/bayara.git
cd bayara
```

Create a virtual environment:

``` bash
python -m venv .venv
```

Activate it:

Windows:

``` bash
.venv\Scripts\activate
```

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# CLI Usage

Run a script:

``` bash
python bayara.py run examples/basic_classification.bay
```

Compile only:

``` bash
python bayara.py compile examples/basic_classification.bay output.py
```

Check syntax:

``` bash
python bayara.py check examples/basic_classification.bay
```

Show version:

``` bash
python bayara.py version
```

------------------------------------------------------------------------

# Language Overview

## dataset

``` bayara
dataset churn from "data/churn.csv"
```

Compiles to:

``` python
churn = pd.read_csv("data/churn.csv")
```

------------------------------------------------------------------------

## show

``` bayara
show churn
```

Equivalent:

``` python
print(churn.head())
```

------------------------------------------------------------------------

## describe

``` bayara
describe churn
```

------------------------------------------------------------------------

## columns

``` bayara
columns churn
```

------------------------------------------------------------------------

## shape

``` bayara
shape churn
```

------------------------------------------------------------------------

# prepare

``` bayara
prepare churn {
    drop nulls
    onehot geography
    standardize age, balance
}
```

Supported operations:

### drop nulls

Removes rows containing missing values.

### onehot

One-hot encoding.

### standardize

Standard scaling.

### normalize

Min‑max scaling.

------------------------------------------------------------------------

# target

``` bayara
target churn -> exited
```

------------------------------------------------------------------------

# features

``` bayara
features churn -> age, balance, salary
```

------------------------------------------------------------------------

# split

``` bayara
split churn test 0.2
```

------------------------------------------------------------------------

# model

``` bayara
model clf as random_forest
```

Supported models:

-   random_forest
-   logistic_regression
-   decision_tree
-   knn
-   naive_bayes
-   linear_regression

------------------------------------------------------------------------

# train

``` bayara
train clf with churn
```

------------------------------------------------------------------------

# evaluate

``` bayara
evaluate clf with accuracy, precision, recall, f1
```

Supported metrics:

Classification: - accuracy - precision - recall - f1

Regression: - mae - mse - r2

------------------------------------------------------------------------

# save

``` bayara
save clf to "models/model.pkl"
```

------------------------------------------------------------------------

# export

``` bayara
export churn to "output.csv"
```

------------------------------------------------------------------------

# Project Structure

    bayara/
    │
    ├── bayara.py
    ├── bayara/
    │   ├── ast_nodes.py
    │   ├── cli.py
    │   ├── errors.py
    │   ├── lexer.py
    │   ├── parser.py
    │   ├── tokens.py
    │   ├── transpiler.py
    │   ├── validator.py
    │   └── version.py
    │
    ├── examples/
    ├── data/
    ├── models/
    ├── requirements.txt
    └── README.md

------------------------------------------------------------------------

# Roadmap

Future improvements may include:

-   improved parser and grammar
-   better error messages
-   dataset inspection tools
-   additional ML models
-   feature engineering helpers
-   improved CLI

------------------------------------------------------------------------

# Philosophy

Bayara aims to be:

-   simple
-   readable
-   focused on tabular ML
-   easy to experiment with

It is not intended to replace full ML frameworks, but to provide a
**small and understandable language for defining ML pipelines**.

------------------------------------------------------------------------

# Acknowledgements

This project was developed with the assistance of AI tools.

------------------------------------------------------------------------

# License

MIT License
