import pandas as pd

df = None

def load_data(csv_path):
    global df
    df = pd.read_csv(csv_path)
    return df
