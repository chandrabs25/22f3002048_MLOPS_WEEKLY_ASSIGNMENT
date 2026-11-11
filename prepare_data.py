from pathlib import Path
import pandas as pd

root = Path(__file__).resolve().parent
data = pd.read_csv(root / "iris_data_adapted_for_feast.csv")

for column in ["event_timestamp", "created_timestamp"]:
    data[column] = pd.to_datetime(data[column], utc=True)

data["iris_id"] = data["iris_id"].astype("int64")

output = root / "feature_repo/data/iris.parquet"
output.parent.mkdir(parents=True, exist_ok=True)
data.to_parquet(output, index=False)

print(f"Saved {len(data)} rows to {output}")
print(data.head())
