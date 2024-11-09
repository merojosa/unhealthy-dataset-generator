import pandas as pd
from typing import List, Optional


def filter_excel_rows(
    file_path: str, filter_numbers: List[int]
) -> Optional[pd.DataFrame]:
    """
    Reads an Excel file and filters rows based on specified numbers in the tip column.

    Args:
        file_path (str): Path to the .xlsx file
        filter_numbers (list): List of numbers to filter for (e.g., [2, 3, 4] will keep rows starting with '2=', '3=', '4=')

    Returns:
        pandas.DataFrame: Filtered dataframe containing only rows matching the specified patterns
    """
    try:
        # Read the first sheet of the Excel file
        df = pd.read_excel(file_path, sheet_name=0)

        # Get the name of column J (tip column) from the header
        tip_column = df.columns[9]  # J is the 10th column (index 9)

        # Create filter condition for each number
        filter_conditions = pd.Series(False, index=df.index)
        for number in filter_numbers:
            filter_conditions |= df[tip_column].astype(str).str.startswith(f"{number}=")

        # Apply the filter
        filtered_df = df[filter_conditions]

        return filtered_df

    except Exception as e:
        print(f"Error processing file: {str(e)}")
        return None
