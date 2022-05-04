from pathlib import Path
from typing import Tuple
import pandas as pd


def get_datasets(
    dataset_path: Path = Path("data/train.csv"),
    fe_type: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    dataset = pd.read_csv(dataset_path)
    dataset.drop("Id", axis=1, inplace=True)
    print(f"Original dataset size: {dataset.shape}.")
    if fe_type == 0:
        print("No feature engineering techniques.")
    else:
        if fe_type == 1:
            print("Feature engineering techniques type 1.")
        elif fe_type == 2:
            print("Feature engineering techniques type 2.")
        print(f"Dataset size after fe: {dataset.shape}.")

    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]

    return (features, None, target, None)

if __name__ == '__main__':
    data = get_datasets(Path("data/train.csv"), 42, 0.2)
