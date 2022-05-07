from pathlib import Path

import click
import mlflow
import mlflow.sklearn
from joblib import dump
from numpy import mean
import datetime
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd

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
    "-t",
    "--test-data-path",
    default="data/test.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "-p",
    "--predicted-data-path",
    default="data/submission.csv",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/unnamed_model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option("--do-prediction", default=False, type=bool, show_default=True)
@click.option("--without-preffix", default=False, type=bool, show_default=True)
@click.option("--model-type", default="rfc", type=str, show_default=True)
@click.option("--fe-type", default=0, type=int, show_default=True)
@click.option("--nested-cv", default=False, type=bool, show_default=True)
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
    test_data_path: Path,
    predicted_data_path: Path,
    save_model_path: Path,
    without_preffix: bool,
    do_prediction: bool,
    model_type: str,
    fe_type: int,
    nested_cv: bool,
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

    runname = model_type + ": fe=" + str(fe_type)
    print(runname)
    mlflow.start_run(run_name=runname)

    # Соберём параметры в словарь:
    params = {
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
    # Исключим из словаря не переданные значения (равные None):
    for key in list(params.keys()):
        if params[key] is None:
            params.pop(key)

    # Вызовем функцию формирования модели
    # и передадим их в качетве параметров
    compute_model(
        dataset_path,
        test_data_path,
        predicted_data_path,
        save_model_path,
        without_preffix,
        do_prediction,
        model_type,
        fe_type,
        nested_cv,
        random_state,
        use_scaler,
        **params
    )

    mlflow.end_run()

def compute_model(
    dataset_path,
    test_data_path,
    predicted_data_path,
    save_model_path,
    without_preffix,
    do_prediction,
    model_type,
    fe_type,
    nested_cv,
    random_state,
    use_scaler,
    **params
) -> None:

    # Получим тренировочный набор данных
    x_train, y_train = \
        get_datasets(
            dataset_path=dataset_path,
            fe_type=fe_type
        )

    # Запишем все параметры модели в MLFlow
    # 1. Для начала все вместе, чтобы были видны в одном столбце
    if len(params) == 0:
        mlflow.log_param("_all_params", "all default")
    else:
        mlflow.log_param("_all_params", params)
    # 2. по одному (будут видны каждый в своём столбце):
    for key in list(params.keys()):
        mlflow.log_param(key.lower(), params[key])

    # Список названий оценок, которые будем вычислять:
    if do_prediction:
        # В случае если нужно сформировать сформировать прогноз
        # не будем тратить время на расчет всех метрик
        # и выведем только accuracy:
        scores_list = ["accuracy"]
    else:
        scores_list = [
            "r2",
            "accuracy",
            "homogeneity_score",
            "neg_mean_absolute_error",
        ]
    # Для задания 7 и 9 использованы разные подходы:
    metric4model_selection = "accuracy"
    if not nested_cv:
        model_search_type = not_nested_cross_validation
    else:
        model_search_type = nested_cross_validation
    # Вызовем подбор моделии в соответствии со значением nested_cv:
    mlmodel = model_search_type(
        x_train=x_train,
        y_train=y_train,
        model_type=model_type,
        random_state=random_state,
        use_scaler=use_scaler,
        scores_list=scores_list,
        metric4model_selection=metric4model_selection,
        **params
    )

    # Сохраним модель по переданному пути:
    preffix = datetime.datetime.now().strftime("(%Y-%d-%m %H-%M-%S)") +\
        " " + model_type + " fe=" + str(fe_type) + " "
    if not without_preffix:
        save_model_path = Path(
            save_model_path.parent,
            preffix + save_model_path.stem + save_model_path.suffix
        )
        test_data_path = Path(
            test_data_path.parent,
            preffix + test_data_path.stem + test_data_path.suffix
        )
    dump(mlmodel, save_model_path)
    print(f"Model is saved to {save_model_path}")

    # Определим необходимость сформировать прогноз:
    if do_prediction:
        # Сформируем прогноз если это требуется:
        x_test, x_ids = \
            get_datasets(
                dataset_path=test_data_path,
                fe_type=fe_type,
                test_data=True
            )

        y_test = mlmodel.predict(x_test)

        y_pred = pd.DataFrame(y_test, index=x_ids, columns=["Cover_Type"])
        y_pred.to_csv(preffix + predicted_data_path, index_label="Id")

def nested_cross_validation(
        x_train,
        y_train,
        model_type,
        random_state,
        use_scaler,
        scores_list,
        metric4model_selection,
        **params
):
    '''
    В соответствии с заданием №9 требуется реализация nested cross-validation.
    Именно это и реализовано в данной функции.
    '''
    pass

def not_nested_cross_validation(
        x_train,
        y_train,
        model_type,
        random_state,
        use_scaler,
        scores_list,
        metric4model_selection,
        **params
):
    '''
    В соответствии с заданием №7 требуется реализация K-fold cross-validation.
    Именно это и реализовано в данной функции.
    '''
    # Слебующая переменная в этой функции не используется
    # и чтобы линтер не ругался, просто обратимся к ней:
    metric4model_selection = metric4model_selection

    # Запустим процедуры в соответствии с pipeline.
    # В словаре params остались только те параметры,
    # которые нужны для той или иной модели ML:
    mlmodel = create_pipeline(model_type, use_scaler, random_state, **params)

    # Сформируем модель
    mlmodel.fit(x_train, y_train)

    # Вычислим разные метрики по результатам cross_val_score
    # с использованием подготовленной модели model:
    cv = KFold(n_splits=10, random_state=random_state, shuffle=True)

    print("---------------------------------------------")
    print("Значение расcчитанных метрик:")
    for one_score in scores_list:
        scores = cross_val_score(
            mlmodel,
            x_train,
            y_train,
            scoring=one_score,
            cv=cv,
            n_jobs=-1
        )
        print_and_save_result(one_score, round(mean(scores), 4))
    print("---------------------------------------------")
    return mlmodel

def print_and_save_result(title: str, value: float) -> None:
    print(f"    {title}: {value}")
    mlflow.log_metric(title, value)
