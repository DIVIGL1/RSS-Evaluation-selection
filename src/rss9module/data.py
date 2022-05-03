from pathlib import Path
from typing import Tuple
import pandas as pd
from sklearn.model_selection import train_test_split
import click


def get_datasets(
    csv_path: Path, random_state: int, test_split_ratio: float
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    dataset = pd.read_csv(csv_path)
    click.echo(f"Common Dataset shape is: {dataset.shape}.")
    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]
    
    features_train, features_val, target_train, target_val = \
        train_test_split(
            features, 
            target, 
            test_size=test_split_ratio, 
            random_state=random_state
        )

    click.echo(f"Train Dataset shape is: {features_train.shape}.")
    click.echo(f"Validate Dataset shape is: {features_val.shape}.")
    
    return features_train, features_val, target_train, target_val

if __name__ == '__main__':
    ppath = Path("data/train.csv")
    data = get_datasets(ppath, 42, 0.3)
