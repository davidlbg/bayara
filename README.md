# Bayara

**Bayara** is a small domain-specific language (DSL) designed for **tabular data analysis and classical machine learning pipelines**.

It allows users to write concise `.bay` scripts that are compiled into Python code using **pandas** and **scikit-learn**.

Bayara is not meant to compete with large frameworks such as MLFlow or full ML platforms.  
Instead, it aims to provide a **simple, readable way to define tabular ML workflows**.

---

# Project Status

Bayara is an **early experimental language (v1.0)**.

The current version focuses on:

- tabular dataset workflows
- classical machine learning models
- simple preprocessing pipelines
- an easy-to-read DSL syntax

Future improvements are planned, but the goal of this version is to provide a **clear and functional minimal language**.

---

# Example

Example Bayara script:

```bayara
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
```

---

# evaluate

Evaluates the trained model.

Example:

```bayara
evaluate clf with accuracy, precision, recall, f1
````

This script:

1. Loads a dataset
2. Prepares the data
3. Defines target and features
4. Splits the dataset
5. Trains a model
6. Evaluates the results

---

# Installation

Clone the repository:

```bash
git clone https://github.com/davidlbg/bayara.git
cd bayara
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it:

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# CLI Usage

Bayara provides a simple CLI.

Run a script:

```bash
python bayara.py run examples/basic_classification.bay
```

Compile only (generate Python code):

```bash
python bayara.py compile examples/basic_classification.bay output.py
```

Check syntax:

```bash
python bayara.py check examples/basic_classification.bay
```

Show version:

```bash
python bayara.py version
```

---

# Language Overview

This section documents the currently supported Bayara commands.

---

# dataset

Loads a dataset from a CSV file.

```bayara
dataset churn from "data/churn.csv"
```

This creates a dataset object named `churn`.

Internally this compiles to:

```python
churn = pd.read_csv("data/churn.csv")
```

---

# show

Displays the first rows of a dataset.

```bayara
show churn
```

Equivalent to:

```python
print(churn.head())
```

---

# describe

Shows dataset statistics.

```bayara
describe churn
```

Equivalent to:

```python
print(churn.describe())
```

---

# columns

Lists dataset columns.

```bayara
columns churn
```

---

# shape

Shows dataset dimensions.

```bayara
shape churn
```

Example output:

```
(40, 4)
```

---

# prepare block

Used to perform simple preprocessing steps.

Example:

```bayara
prepare churn {
    drop nulls
    onehot geography
    standardize age, balance
}
```

Supported operations:

### drop nulls

Removes rows containing missing values.

```
drop nulls
```

---

### onehot

Performs one-hot encoding.

```
onehot column_name
```

---

### standardize

Applies standard scaling (mean 0, variance 1).

```
standardize column1, column2
```

---

### normalize

Applies min-max scaling.

```
normalize column1, column2
```

---

# target

Defines the prediction target column.

```bayara
target churn -> exited
```

---

# features

Defines the input features.

```bayara
features churn -> age, balance, salary
```

---

# split

Splits the dataset into training and test sets.

```bayara
split churn test 0.2
```

This means **20% test data**.

---

# model

Defines the machine learning model.

Example:

```bayara
model clf as random_forest
```

Supported models:

* random_forest
* logistic_regression
* decision_tree
* knn
* naive_bayes
* linear_regression

---

# train

Trains a model.

```bayara
train clf with churn
```

---

# evaluate

Evaluates the trained model.

Example:

```bayara
evaluate clf with accuracy, precision, recall, f1
```

Supported metrics:

### Classification

* accuracy
* precision
* recall
* f1

### Regression

* mae
* mse
* r2

---

# save

Saves the trained model.

```bayara
save clf to "models/model.pkl"
```

---

# export

Exports a dataset to CSV.

```bayara
export churn to "output.csv"
```

---

# Project Structure

```
bayara/
│
├── bayara.py
├── bayara/
│   ├── cli.py
│   ├── lexer.py
│   ├── parser.py
│   ├── transpiler.py
│   └── ast_nodes.py
│
├── examples/
│
├── data/
│
├── models/
│
├── requirements.txt
└── README.md
```

---

# Roadmap

Future improvements may include:

* improved parser and grammar
* better error messages
* dataset inspection tools
* additional ML models
* feature engineering helpers
* improved CLI

---

# Philosophy

Bayara aims to be:

* simple
* readable
* focused on tabular ML
* easy to experiment with

It is not intended to replace full ML frameworks, but to provide a **small and understandable language for defining ML pipelines**.

---

# Acknowledgements

This project was developed with the assistance of AI tools.

---

# License

MIT License
