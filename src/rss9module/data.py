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
            print("Feature engineering techniques type 2.")
            columns = [
                "Elevation",
                "Slope",
                "Horizontal_Distance_To_Hydrology",
                "Vertical_Distance_To_Hydrology",
                "Horizontal_Distance_To_Roadways",
                "Horizontal_Distance_To_Fire_Points",
            ]
            for one_col in columns:
                dataset[one_col + "_2"] = dataset[one_col] ** 2
                dataset[one_col + "_3"] = dataset[one_col] ** 3
                dataset[one_col + "_4"] = dataset[one_col] ** 4

            dataset["Soil_Type_sum"] = 0
            for soil_num in range(1, 41):
                colname = "Soil_Type" + str(soil_num)
                dataset["Soil_Type_sum"] += dataset[colname]
                dataset.drop(colname, inplace=True, axis=1)

        elif fe_type == 2:
            '''
            Удаляем параметры, которые "pandas profiling" выдал:
             - как нулевые
             - как имеющие пары с сильной корреляцией
            '''
            cm = dataset.corr()[["Cover_Type"]]
            cm = cm[cm.index != "Cover_Type"]
            drop_columns = cm[cm > 0].dropna().index

            dataset.drop(drop_columns, axis=1, inplace=True)

            print("Feature engineering techniques type 1.")

        print(f"Dataset size after fe: {dataset.shape}.")

    features = dataset.drop("Cover_Type", axis=1)
    target = dataset["Cover_Type"]

    return (features, target)

if __name__ == '__main__':
    data = get_datasets(Path("data/train.csv"), 42, 0.2)
