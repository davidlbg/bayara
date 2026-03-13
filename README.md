
<p align="center">
  <img src="https://raw.githubusercontent.com/davidlbg/bayara/main/assets/bayara-banner.png" alt="Bayara Banner" width="700"/>
</p>

<h1 align="center">Bayara</h1>

<p align="center">
  A small domain‑specific language for tabular machine learning pipelines
</p>

<p align="center">

![Version](https://img.shields.io/badge/version-1.0.2-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Status](https://img.shields.io/badge/status-experimental-orange)
![VS Code Extension](https://img.shields.io/badge/VSCode-extension-purple)

</p>

---

# What is Bayara?

**Bayara** is a small domain‑specific language (DSL) designed for **tabular data analysis and classical machine learning pipelines**.

It allows users to write simple `.bay` scripts that compile into **Python code using pandas and scikit‑learn**.

The goal of Bayara is not to replace large ML platforms, but to provide a **minimal and readable language for describing ML workflows**.

Bayara emphasizes:

• simplicity  
• readability  
• small learning curve  
• reproducible ML pipelines  

---

# Example

A complete machine learning pipeline in Bayara:

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

evaluate clf with accuracy, precision, recall, f1
```

This script:

1. Loads a dataset
2. Cleans the data
3. Defines target and features
4. Splits the dataset
5. Trains a model
6. Evaluates the results

Bayara compiles this script into **Python code using pandas and scikit‑learn**.

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

Install Bayara:

```bash
pip install -e .
```

---

# CLI Usage

Run a script

```bash
bayara run examples/basic_classification.bay
```

Compile to Python

```bash
bayara compile examples/basic_classification.bay output.py
```

Check syntax and semantic errors

```bash
bayara check examples/basic_classification.bay
```

Show version

```bash
bayara version
```

---

# Language Overview

## dataset

Loads a dataset from a CSV file.

```bayara
dataset churn from "data/churn.csv"
```

Compiles to:

```python
churn = pd.read_csv("data/churn.csv")
```

---

## prepare

Preprocessing block:

```bayara
prepare churn {
    drop nulls
    fill nulls age with median
    onehot geography
    standardize age, balance
}
```

Supported operations

• drop nulls  
• fill nulls  
• onehot  
• standardize  
• normalize  

---

## target

```bayara
target churn -> exited
```

---

## features

```bayara
features churn -> age, balance, salary
```

---

## split

```bayara
split churn test 0.2
```

If omitted, Bayara uses **0.2 as default**.

---

## model

```bayara
model clf as random_forest
```

Supported models:

• random_forest  
• logistic_regression  
• decision_tree  
• knn  
• naive_bayes  
• linear_regression  

---

## train

```bayara
train clf with churn
```

---

## evaluate

```bayara
evaluate clf with accuracy, precision, recall, f1
```

Metrics:

Classification:

• accuracy  
• precision  
• recall  
• f1  

Regression:

• mae  
• mse  
• r2  

---

# VS Code Extension

Bayara provides official **VS Code language support** including:

• syntax highlighting  
• autocomplete  
• snippets  
• hover documentation  

Search for:

**"Bayara Language Support"**

in the VS Code Marketplace.

---

# Project Structure

```
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
├── exports/
├── tests/
│
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

# Roadmap

Planned improvements

• improved grammar and parser  
• better error diagnostics  
• more ML models  
• plotting commands  
• pipeline abstractions  
• improved CLI features  

---

# Philosophy

Bayara aims to be

• simple  
• readable  
• focused on tabular ML  
• easy to experiment with  

It is intentionally **small and understandable**.

---

# Acknowledgements

This project was developed with the assistance of AI tools.

---

# License

MIT License
