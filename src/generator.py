import pandas as pd
from typing import Optional, Any
from src.processor import process_row
from src import non_ad_generator
import os
import shutil


def generate_dataset(config: Any) -> Optional[pd.DataFrame]:
    ads_dataframe = None

    try:
        # Read the first sheet of the Excel file
        ads_dataframe = pd.read_excel(
            f"{config.get("path").get("dataset")}/metadata.xlsx", sheet_name=0
        )
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

    result_path = f"{config.get("path").get("dataset")}/result"
    if os.path.exists(result_path):
        shutil.rmtree(result_path)

    # Accumulate the ad-frame count from process_row directly instead of
    # rescanning result/ad/ with glob at the end. With thousands of output
    # files glob.glob walks the whole directory; the running total is free.
    tip_values = config.get("tip_values")
    ad_count = 0
    for _, row in ads_dataframe.iterrows():
        ad_type = row["tip"]
        for tip_number in tip_values:
            if ad_type.startswith(f"{tip_number}="):  # Filter by tip_values
                ad_count += process_row(row, config)

    non_ad_generator.generate_non_ad_images(ads_dataframe, config, ad_count)

    return ads_dataframe
