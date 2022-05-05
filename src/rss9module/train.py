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
@click.option("--model", default="rfc", type=str, show_default=True)
@click.option("--fe-type", default=0, type=int, show_default=True)
@click.option("--random-state", default=42, type=int, show_default=True)
@click.option("--use-scaler", default=True, type=bool, show_default=True)
@click.option("--n-estimators", default=None, type=int, show_default=True)
@click.option("--criterion", default=None, type=str, show_default=True)
@click.option("--max-depth", default=None, type=int, show_default=True)
@click.option("--max-features", default=None, type=str, show_default=True)
@click.option("--n-neighbors", default=None, type=int, show_default=True)
@click.option("--weights", default=None, type=str, show_default=True)
@click.option("--algorithm", default=None, type=str, show_default=True)
@click.option("--c-param", default=None, type=int, show_default=True)
@click.option("--kernel", default=None, type=str, show_default=True)
@click.option("--shrinking", default=None, type=bool, show_default=True)
@click.option("--tol", default=None, type=float, show_default=True)
def train(
    dataset_path: Path,
    save_model_path: Path,
    model: str,
    fe_type: int,
    random_state: int,
    use_scaler: bool,
    n_estimators: int,
    criterion: str,
    max_depth: int,
    max_features: str,
    n_neighbors: int,
    weights: str,
    algorithm: str,
    c_param: str,
    kernel: str,
    shrinking: bool,
    tol: float,
) -> None:

    runname = model + ": fe=" + str(fe_type)
    print(runname)
    mlflow.start_run(run_name=runname)

    # Соберём параметры в словарь:
    params = {
        "dataset_path": dataset_path,
        "save_model_path": save_model_path,
        "model": model,
        "fe_type": fe_type,
        "random_state": random_state,
        "use_scaler": use_scaler,
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
        "max_features": max_features,
        "n_neighbors": n_neighbors,
        "weights": weights,
        "algorithm": algorithm,
        "C": c_param,
        "kernel": kernel,
        "shrinking": shrinking,
        "tol": tol,
    }
    # Исключим из словаря не переданные значения (те кто = None):
    for key in list(params.keys()):
        if params[key] is None:
            params.pop(key)
    # Передадим их в качетве параметров
    compute_model(**params)
    mlflow.end_run()

def compute_model(**params):
    # Получим набор данных
    x_train, y_train = \
        get_datasets(
            dataset_path=params["dataset_path"],
            fe_type=params["fe_type"]
        )
    # Обработаем словарь с параметрами для дальнейшего обмена:
    save_model_path = params["save_model_path"]
    use_scaler = params["use_scaler"]
    random_state = params["random_state"]
    model = params["model"]
    params.pop("dataset_path")
    params.pop("fe_type")
    params.pop("save_model_path")
    params.pop("use_scaler")
    params.pop("random_state")
    params.pop("model")

    # Запишем все пароаметры модели в MLFlow:
    # 1. все вместе (будут видны в одном столбце):
    if len(params) == 0:
        mlflow.log_param("_all_params", "all default")
    else:
        mlflow.log_param("_all_params", params)
    # 2. по одному (будут видны каждый в своём столбце):
    for key in list(params.keys()):
        mlflow.log_param(key.lower(), params[key])

    # Запустим процедуры в соответствии с pipeline.
    # В словаре params остались только те параметры,
    # которые нужны для той или иной модели ML:
    model = create_pipeline(model, use_scaler, random_state, **params)

    model.fit(x_train, y_train)

    # Список названий оценок, которые будем вычислять:
    scores_list = [
        "r2",
        "accuracy",
        "homogeneity_score",
        "neg_mean_absolute_error",
    ]

    # вычислим разные метрики по результатам cross_val_score
    # с использованием подготовленной модели model:
    cv = KFold(n_splits=10, random_state=random_state, shuffle=True)

    print("---------------------------------------------")
    print("Значение расcчитанных метрик:")
    for one_score in scores_list:
        scores = cross_val_score(
            model,
            x_train,
            y_train,
            scoring=one_score,
            cv=cv,
            n_jobs=-1
        )
        print_and_save_result(one_score, round(mean(scores), 4))
    print("---------------------------------------------")

    # Сохраним модель по переданному пути:
    dump(model, save_model_path)
    print(f"Model is saved to {save_model_path}")

def print_and_save_result(title: str, value: float) -> None:
    print(f"    {title}: {value}")
    mlflow.log_metric(title, value)
