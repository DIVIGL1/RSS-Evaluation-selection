from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split


def get_datasets(
    csv_path: Path = Path("data/train.csv"),
    random_state: int = 42,
    test_size: float = 0.0
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    dataset = pd.read_csv(csv_path)
    dataset.drop("Id", axis=1, inplace=True)
    print(f"Dataset shape: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]

    if test_size == 0.0:
        return (features, None, target, None)
    else:
        return(
            train_test_split(
                features,
                target,
                test_size=test_size,
                random_state=random_state
            )
        )

if __name__ == '__main__':
    data = get_datasets(Path("data/train.csv"), 42, 0.2)
