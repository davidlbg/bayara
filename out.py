from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

churn = pd.read_csv(r"data/churn.csv")
print("shape:", churn.shape)
print("columns:", list(churn.columns))
print("dtypes:")
print(churn.dtypes)
print("missing values:")
print(churn.isnull().sum())
clf = LogisticRegression(max_iter=1000, random_state=42)
print("[Bayara] No split defined for churn; using default test size 0.2")
churn["balance"] = churn["balance"].fillna(churn["balance"].median())
churn = pd.get_dummies(churn, columns=['segment'])
X = churn[['age', 'balance', 'salary', 'segment_enterprise', 'segment_retail']]
y = churn["exited"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
_standardize_scaler_churn_1 = StandardScaler()
X_train[['age', 'balance', 'salary']] = _standardize_scaler_churn_1.fit_transform(X_train[['age', 'balance', 'salary']])
X_test[['age', 'balance', 'salary']] = _standardize_scaler_churn_1.transform(X_test[['age', 'balance', 'salary']])
clf.fit(X_train, y_train)
preds = clf.predict(X_test)
print("accuracy:", accuracy_score(y_test, preds))
print("precision:", precision_score(y_test, preds, zero_division=0))
print("recall:", recall_score(y_test, preds, zero_division=0))
print("f1:", f1_score(y_test, preds, zero_division=0))
Path(r"models/churn.pkl").parent.mkdir(parents=True, exist_ok=True)
joblib.dump(clf, r"models/churn.pkl")
