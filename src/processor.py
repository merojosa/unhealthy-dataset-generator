import pandas as pd
from typing import Any


def process_row(row: pd.Series, config: Any):
    date = row["fec"]
    tv_channel = row["can"]
    start_time = row["hin"]
    end_time = row["hfi"]

    # Check if video exists

    # Extract images based on start time and end time (one frame per second?)

    # Save those images in their respective directory following this guideline: https://docs.ultralytics.com/datasets/classify/
    print(row)
