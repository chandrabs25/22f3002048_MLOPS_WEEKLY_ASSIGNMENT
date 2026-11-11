from pathlib import Path

import joblib
import pandas as pd
from feast import FeatureStore
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

# Read only lookup keys, timestamps, and labels from the CSV.
entity_df = pd.read_csv(
    "iris_data_adapted_for_feast.csv",
    usecols=["iris_id", "event_timestamp", "species"],
)
entity_df["event_timestamp"] = pd.to_datetime(
    entity_df["event_timestamp"], utc=True
)

# Retrieve the four model input features through Feast.
store = FeatureStore(repo_path="feature_repo")
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=[f"iris_features:{name}" for name in features],
).to_df()

assert len(training_data) == len(entity_df), "Unexpected row count"
assert not training_data[features].isna().any().any(), "Missing features"

print("Rows retrieved from Feast:", len(training_data))
print(training_data.head())

train, test = train_test_split(
    training_data,
    test_size=0.4,
    stratify=training_data["species"],
    random_state=42,
)

model = DecisionTreeClassifier(max_depth=3, random_state=1)
model.fit(train[features], train["species"])

predictions = model.predict(test[features])
print("Test accuracy:", accuracy_score(test["species"], predictions))
print("Model classes:", model.classes_)

Path("artifacts").mkdir(exist_ok=True)
joblib.dump(model, "artifacts/iris_feast_model.joblib")
print("Saved artifacts/iris_feast_model.joblib")
