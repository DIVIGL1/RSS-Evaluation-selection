from pathlib import Path
from typing import Tuple
from sklearn.decomposition import PCA

import pandas as pd


def get_datasets(
    dataset_path: Path = Path("data/train.csv"),
    fe_type: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:

    dataset = pd.read_csv(dataset_path)
    dataset.drop("Id", axis=1, inplace=True)
    print(f"Original dataset size: {dataset.shape}.")
    if fe_type == 0:
        print(
'''
No feature engineering techniques.
    (Used original data)
'''
        )
    else:
        if fe_type == 1:
            print(
'''
Feature engineering techniques type 1.
    (Added degrees (2-4) and new columns)
'''
            )
            fe_by_hans(dataset, degree=4)

        elif fe_type == 2:
            print(
'''
Feature engineering techniques type 2.
    (Used original data and PCA with n_components=50)
'''
            )
            target = dataset["Cover_Type"]
            dataset.drop("Cover_Type", inplace=True, axis=1)

            dataset = pca_df(dataset, n_components=50)

            print(f"Dataset size after fe: {dataset.shape}.")
            return (dataset, target)

        elif fe_type == 3:
            print(
'''
Feature engineering techniques type 3.
    (Added degrees (2-3) and new columns
    and used PCA with n_components=50)
'''
            )
            fe_by_hans(dataset, degree=3, nodrop=True)

            target = dataset["Cover_Type"]
            dataset.drop("Cover_Type", inplace=True, axis=1)

            dataset = pca_df(dataset, n_components=50)

            print(f"Dataset size after fe: {dataset.shape}.")
            return (dataset, target)

        print(f"Dataset size after fe: {dataset.shape}.")

    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]

    return (features, target)

def pca_df(dataset: pd.DataFrame, n_components=2) -> None:
    pca = PCA(
        n_components=n_components,
        svd_solver='randomized',
        whiten=True
    ).fit(dataset)
    dataset = pd.DataFrame(pca.transform(dataset))
    return (dataset)
    
def fe_by_hans(dataset: pd.DataFrame, degree=4, nodrop=False) -> None:
    columns = [
        "Elevation",
        "Slope",
        "Horizontal_Distance_To_Hydrology",
        "Vertical_Distance_To_Hydrology",
        "Horizontal_Distance_To_Roadways",
        "Horizontal_Distance_To_Fire_Points",
    ]
    for one_col in columns:
        for num in range(2, degree + 1):
            dataset[one_col + "_" + str(num)] = dataset[one_col] ** num

    dataset["Dist"] =\
        dataset["Horizontal_Distance_To_Roadways"] +\
        dataset["Horizontal_Distance_To_Fire_Points"]

    dataset["Soil_Type_sum"] = 0
    for soil_num in range(1, 41):
        colname = "Soil_Type" + str(soil_num)
        dataset["Soil_Type_sum"] += 2 ** dataset[colname]
        if not nodrop:
            dataset.drop(colname, inplace=True, axis=1)
    

if __name__ == '__main__':
    data = get_datasets(Path("data/train.csv"), 42, 0.2)
