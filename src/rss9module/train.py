from pathlib import Path

import click
import mlflow
import mlflow.sklearn
from joblib import dump
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             hamming_loss, matthews_corrcoef, precision_score)

from .data import get_datasets
from .pipeline import create_pipeline

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/unknown_model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option("--random-state", default=42, type=int, show_default=True)
@click.option("--test-size", default=0.2, type=float, show_default=True)
@click.option("--use-scaler", default=True, type=bool, show_default=True)
@click.option("--n-estimators", default=100, type=int, show_default=True)
@click.option("--criterion", default="gini", type=str, show_default=True)
@click.option("--max-depth", default=None, type=int, show_default=True)
@click.option("--max-features", default="auto", type=str, show_default=True)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_size: float,
    use_scaler: bool,
    n_estimators: int,
    criterion: str,
    max_depth: int,
    max_features: str
) -> None:
    compute_model(
        dataset_path=dataset_path,
        save_model_path=save_model_path,
        random_state=random_state,
        test_size=test_size,
        use_scaler=use_scaler,
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features
    )

def compute_model(
    dataset_path: Path = Path("data/train.csv"),
    save_model_path: Path = Path("data/unknown_model.joblib"),
    random_state: int = 42,
    test_size: float = 0.2,
    use_scaler: bool = True,
    n_estimators: int = 100,
    criterion: str = "gini",
    max_depth: int = None,
    max_features: str = "auto"
):
    # Получим наборы данных для обучения и валидации:
    features_train, features_val, target_train, target_val = get_datasets(
        dataset_path,
        random_state,
        test_size,
    )
    with mlflow.start_run():
        # Соберём параметры для передачи в функцию:
        params = {
            "random_state": random_state,
            "use_scaler": use_scaler,
            "n_estimators": n_estimators,
            "criterion": criterion,
            "max_depth": max_depth,
            "max_features": max_features,
        }
        # Запустим процедуры в соответствии с pipeline:
        pipeline = create_pipeline(**params)
        pipeline.fit(features_train, target_train)

        # на основании обсчитанно модели получим
        # предсказание для валидационной выборки:
        predict_val = pipeline.predict(features_val)

        # вычислим три разные метрики по результатам работы модели:
        acc = round(accuracy_score(target_val, predict_val), 4)
        balanced_acc = round(
            balanced_accuracy_score(target_val, predict_val), 4
        )
        m_corrcoef = round(matthews_corrcoef(target_val, predict_val), 4)
        precision_score_w = round(
            precision_score(target_val, predict_val, average="weighted"), 4
        )
        h_loss = round(hamming_loss(target_val, predict_val), 4)

        # Запишем данные в MLFlow:
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("criterion", criterion)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("max_features", max_features)

        # Выведем значения метрик:
        print("---------------------------------------------")
        print("Значение расчитанных метрик:")
        print_and_save_result("accuracy", acc)
        print_and_save_result("balanced accuracy", balanced_acc)
        print_and_save_result("Matthews correlation coefficient", m_corrcoef)
        print_and_save_result("precision score weighted", precision_score_w)
        print_and_save_result("Hamming loss", h_loss)
        print("---------------------------------------------")

        # Сохраним модель по переданному пути:
        dump(pipeline, save_model_path)
        print(f"Model is saved to {save_model_path}")

def print_and_save_result(title: str, value: float) -> None:
    print(f"    {title}: {value}")
    mlflow.log_metric(title, value)
