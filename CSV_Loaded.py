from crewai.tools import BaseTool
import pandas as pd
class CSVLoaderTool(BaseTool):
    name: str = "csv_loader"
    description: str = "Loads a CSV file and returns schema and sample data"

    def _run(self, file_path: str):
        df = pd.read_csv(file_path)

        return {
            "columns": list(df.columns),
            "sample_rows": df.head(5).to_dict(orient="records"),
            "row_count": len(df)
        }
