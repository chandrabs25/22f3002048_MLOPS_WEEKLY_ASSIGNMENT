from pathlib import Path

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

data = pd.read_csv("iris.csv")

train, test = train_test_split(
    data, test_size=0.4, stratify=data["species"], random_state=42
)

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X_train = train[features]
y_train = train.species
X_test = test[features]
y_test = test.species

mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)

prediction = mod_dt.predict(X_test)
print("Accuracy:", metrics.accuracy_score(y_test, prediction))
print("Dataset rows:", len(data))

Path("artifacts").mkdir(exist_ok=True)
joblib.dump(mod_dt, "artifacts/model.joblib")
