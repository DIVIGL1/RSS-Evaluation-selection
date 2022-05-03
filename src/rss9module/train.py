from pathlib import Path

import click
import mlflow
import mlflow.sklearn
from joblib import dump
from numpy import mean
from sklearn.model_selection import KFold, cross_val_score

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
    test_size: float = 0.0,
    use_scaler: bool = True,
    n_estimators: int = 100,
    criterion: str = "gini",
    max_depth: int = None,
    max_features: str = "auto"
):
    # Получим набор данных
    # (если передадит test_size=0.0, то разбиения не будет):
    x_train, _, y_train, _ = get_datasets(dataset_path)
#     with mlflow.start_run():
    # Соберём параметры для передачи в функцию:
    params = {
        "random_state": random_state,
        "use_scaler": use_scaler,
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
        "max_features": max_features,
    }
    # Запишем данные в MLFlow:
    mlflow.log_param("random_state", random_state)
    mlflow.log_param("use_scaler", use_scaler)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("criterion", criterion)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("max_features", max_features)

    # Запустим процедуры в соответствии с pipeline:
    model = create_pipeline(**params)
    model.fit(x_train, y_train)

    # Список названий оценок:
    scores_list = [
        "r2",
        "accuracy",
        "homogeneity_score",
        "neg_mean_absolute_error",
        "f1_macro",
        "roc_auc_ovr",
    ]

    # вычислим разные метрики по результатам cross_val_score
    # с использованием подготовленной модели model:
    cv = KFold(n_splits=10, random_state=random_state, shuffle=True)

    print("---------------------------------------------")
    print("Значение расчитанных метрик:")
    for one_score in scores_list:
        scores = cross_val_score(
            model, x_train, y_train, scoring=one_score, cv=cv, n_jobs=-1
        )
        print_and_save_result(one_score, round(mean(scores), 4))
    print("---------------------------------------------")

    # Сохраним модель по переданному пути:
    dump(model, save_model_path)
    print(f"Model is saved to {save_model_path}")

def print_and_save_result(title: str, value: float) -> None:
    print(f"    {title}: {value}")
    mlflow.log_metric(title, value)
