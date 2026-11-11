import numpy as np
import pandas as pd

data = pd.read_csv("iris.csv")
features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

extra = data.groupby("species").sample(
    n=5, replace=True, random_state=42
).copy()

noise = np.random.default_rng(42).normal(0, 0.05, (len(extra), 4))
extra[features] = (extra[features] + noise).clip(lower=0.01).round(2)

updated = pd.concat([data, extra], ignore_index=True)
updated.to_csv("iris.csv", index=False)

print(f"Dataset rows: {len(data)} -> {len(updated)}")
