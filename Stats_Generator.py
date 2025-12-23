from crewai.tools import BaseTool
import pandas as pd

class StatsTool(BaseTool):
    name: str = "stats_generator"
    description: str = "Computes statistics and aggregations from logistics data"

    def _run(self, file_path: str):
        df = pd.read_csv(file_path)

        stats = {
            "row_count": len(df),
            "numeric_summary": df.describe(include="number").to_dict(),
            "categorical_counts": {
                col: df[col].value_counts().head(10).to_dict()
                for col in df.select_dtypes(include="object").columns
            }
        }

        return stats
        
