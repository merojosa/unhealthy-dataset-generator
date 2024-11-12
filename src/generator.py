import pandas as pd
from typing import Optional, Any
from src.processor import process_row


def generate_dataset(config: Any) -> Optional[pd.DataFrame]:
    ads_dataframe = None

    try:
        # Read the first sheet of the Excel file
        ads_dataframe = pd.read_excel(
            f"{config.get("path")}/metadata.xlsx", sheet_name=0
        )
    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None

    tip_values = config.get("tip_values")
    for _, row in ads_dataframe.iterrows():
        ad_type = row["tip"]
        for tip_number in tip_values:
            if ad_type.startswith(f"{tip_number}="):  # Filter by tip_values
                process_row(row, config)

    return ads_dataframe
