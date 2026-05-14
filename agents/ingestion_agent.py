import pandas as pd
from tools.validation_tools import validate_dataset


class IngestionAgent:
    def run(self, file):
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        else:
            df = pd.read_excel(file)

        metadata = validate_dataset(df)

        return {
            "uploaded_df": df,
            "metadata": metadata
        }