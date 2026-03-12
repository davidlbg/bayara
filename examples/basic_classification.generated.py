import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

churn = pd.read_csv(r"data/churn.csv")
print(churn.shape)
print(list(churn.columns))
print(churn.head())
print(churn.describe(include="all"))
churn = churn.dropna()
_churn_std_2 = StandardScaler()
churn[['age', 'balance', 'salary']] = _churn_std_2.fit_transform(churn[['age', 'balance', 'salary']])
churn_X = churn[['age', 'balance', 'salary']]
churn_y = churn["exited"]
churn_X_train, churn_X_test, churn_y_train, churn_y_test = train_test_split(churn_X, churn_y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(churn_X_train, churn_y_train)
clf_preds = clf.predict(churn_X_test)
print("accuracy:", accuracy_score(churn_y_test, clf_preds))
print("precision:", precision_score(churn_y_test, clf_preds, zero_division=0))
print("recall:", recall_score(churn_y_test, clf_preds, zero_division=0))
print("f1:", f1_score(churn_y_test, clf_preds, zero_division=0))
os.makedirs(os.path.dirname(r"models/churn_rf.pkl") or ".", exist_ok=True)
joblib.dump(clf, r"models/churn_rf.pkl")
print("saved model to:", r"models/churn_rf.pkl")
