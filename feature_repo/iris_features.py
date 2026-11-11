from datetime import timedelta
from pathlib import Path

from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float64
from feast.value_type import ValueType

iris = Entity(
    name="iris",
    join_keys=["iris_id"],
    value_type=ValueType.INT64,
)

iris_source = FileSource(
    name="iris_source",
    path=str(Path(__file__).resolve().parent / "data/iris.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

iris_features = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=0),
    schema=[
        Field(name="sepal_length", dtype=Float64),
        Field(name="sepal_width", dtype=Float64),
        Field(name="petal_length", dtype=Float64),
        Field(name="petal_width", dtype=Float64),
    ],
    online=True,
    source=iris_source,
)
