from datetime import timedelta

import numpy as np
import pandas as pd
from feast import FeatureStore

data = pd.read_parquet("feature_repo/data/iris.parquet")
store = FeatureStore(repo_path="feature_repo")

# Cover the dataset's actual date range.
store.materialize(
    start_date=data["event_timestamp"].min().to_pydatetime() - timedelta(seconds=1),
    end_date=data["event_timestamp"].max().to_pydatetime() + timedelta(seconds=1),
)

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
ids = sorted(data["iris_id"].unique().tolist())

online = store.get_online_features(
    features=[f"iris_features:{name}" for name in features],
    entity_rows=[{"iris_id": int(i)} for i in ids],
).to_df()

print("\nFeatures retrieved from SQLite:")
print(online)

# SQLite should contain the latest measurement for each iris ID.
expected = (
    data.sort_values("event_timestamp")
    .groupby("iris_id").tail(1)
    .set_index("iris_id").sort_index()
)
actual = online.set_index("iris_id").sort_index()

assert actual.index.equals(expected.index), "Iris IDs do not match"
np.testing.assert_allclose(
    actual[features].to_numpy(dtype=float),
    expected[features].to_numpy(dtype=float),
)

print("\nSUCCESS: online features match the latest source measurements.")
