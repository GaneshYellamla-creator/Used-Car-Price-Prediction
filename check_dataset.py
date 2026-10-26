
import pandas as pd
from pathlib import Path

DATA_FILE = Path("data/used_cars.csv")

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA_FILE.resolve()}\n"
        "Place used_cars.csv inside the data folder."
    )

df = pd.read_csv(DATA_FILE)

print("\n========== DATASET INFORMATION ==========")
print("Dataset loaded successfully!")
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst five rows:")
print(df.head().to_string())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
