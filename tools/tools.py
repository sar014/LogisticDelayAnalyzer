from crewai.tools import tool
import pandas as pd
from io import StringIO

@tool
def inspect_csv(csv_content: str):
    """
    Inspect CSV structure, data types, missing values, and sample rows.
    Accepts CSV content as a string (not file path).
    """
    print("using inspect_csv")

    # Convert CSV string into DataFrame
    df = pd.read_csv(StringIO(csv_content))

    return {
        "total_rows": int(len(df)),
        "columns": list(df.columns),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "sample_rows": df.head(5).to_dict(orient="records")
    }
