from pathlib import Path

import click
import mlflow
import mlflow.sklearn
from joblib import dump
from numpy import mean
import datetime
from sklearn.model_selection import KFold, cross_val_score
import pandas as pd
from sklearn.model_selection import GridSearchCV
from typing import Dict
from sklearn.metrics import accuracy_score, r2_score
from sklearn.metrics import homogeneity_score
from sklearn.metrics import rand_score
from sklearn.pipeline import Pipeline

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
@click.option("--c-param", default=None, type=float, show_default=True)
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
) -> (None):

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
        **params,
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
    **params,
) -> (None):

    # Получим тренировочный набор данных
    x_train, y_train = get_datasets(dataset_path=dataset_path, fe_type=fe_type)

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
        scores_list = [["accuracy"], [accuracy_score], [True]]
    else:
        scores_list = [
            [
                "r2",
                "accuracy",
                "homogeneity_score",
                "rand_score",
            ],
            [
                r2_score,
                accuracy_score,
                homogeneity_score,
                rand_score,
            ],
            [
                False,
                True,
                False,
                False,
            ],
        ]

    # Запустим процедуры в соответствии с pipeline.
    # В словаре params остались только те параметры,
    # которые нужны для той или иной модели ML:
    mlmodel = create_pipeline(model_type, use_scaler, random_state, **params)

    # Для задания 7 и 9 использованы разные подходы,
    # они реализуются через следующую функию:
    mlmodel = nested_cross_validation(
        nested_cv=nested_cv,
        mlmodel=mlmodel,
        X=x_train,
        y=y_train,
        model_type=model_type,
        random_state=random_state,
        scores_list=scores_list,
        **params,
    )

    # Сохраним модель по переданному пути:
    preffix = datetime.datetime.now().strftime("(%Y-%d-%m %H-%M-%S)")
    preffix = preffix + " " + model_type + " fe=" + str(fe_type) + " "

    if not without_preffix:
        save_model_path = Path(
            save_model_path.parent,
            preffix + save_model_path.stem + save_model_path.suffix,
        )
        filename = preffix + test_data_path.stem + test_data_path.suffix
        test_data_path = Path(test_data_path.parent, filename)
    dump(mlmodel, save_model_path)
    print(f"Model is saved to {save_model_path}")

    # Определим необходимость сформировать прогноз:
    if do_prediction:
        # Сформируем прогноз если это требуется:
        x_test, x_ids = get_datasets(
            dataset_path=test_data_path, fe_type=fe_type, test_data=True
        )

        y_test = mlmodel.predict(x_test)

        y_pred = pd.DataFrame(y_test, index=x_ids, columns=["Cover_Type"])
        y_pred.to_csv(preffix + predicted_data_path, index_label="Id")


def nested_cross_validation(
    nested_cv, mlmodel, X, y, model_type, random_state, scores_list, **params
) -> (Pipeline):
    """
    В соответствии с заданием №9 требуется реализация nested cross-validation.
    Именно это и реализовано в данной функции.
    """
    if nested_cv:
        # В зависимости от типа модели определеяем
        # сетку параметров для выбора best_estimator_:
        if model_type == "rfc":
            grid_space = {
                "n_estimators": range(100, 201, 50),
                "max_features": ["sqrt", "log2", "auto"],
                "max_depth": [None, 10],
            }
        elif model_type == "knn":
            grid_space = {
                "n_neighbors": [2, 3, 4],
                "weights": ["distance", "uniform"],
                "algorithm": ["auto", "ball_tree"],
            }
        elif model_type == "svc":
            grid_space = {
                "C": [0.1, 1],
                "kernel": ["rbf", "poly"],
                "tol": [0.0001, 0.001],
                "shrinking": [True, False],
            }
        grid_space_pipe = rename_params(model_type, grid_space)

        # Создаём пустой список для хранения в нём оценок:
        outer_results = {key: [] for key in scores_list[0]}
        max_scorr = float("-inf")
        found_best_params = {}
        # Найдём какая метрика является контрольной:
        for one_score_name, p_control in zip(scores_list[0], scores_list[2]):
            if p_control:
                name_of_control_metric = one_score_name

        # Настраиваем внешнюю процедуру для кросс-валидации:

        cv_outer = KFold(
            n_splits=3,
            shuffle=True,
            random_state=random_state + 1
        )
        X_array = X.to_numpy()

        for train_ix, test_ix in cv_outer.split(X_array):
            # split data
            X_train, X_test = X_array[train_ix, :], X_array[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]
            # Настраиваем внутреннюю процедуру для кросс-валидации:
            cv_inner = KFold(
                n_splits=3,
                shuffle=True,
                random_state=random_state + 2
            )

            # Создадим GridSearchCV и выполним поиск по выбранной сетке:
            gscv = GridSearchCV(
                mlmodel,
                grid_space_pipe,
                scoring=name_of_control_metric,
                cv=cv_inner,
                refit=True,
            )
            result = gscv.fit(X_train, y_train)

            # Выберем лучшую модель
            inner_best_mlmodel = result.best_estimator_

            # Использую выбраную (лучшую) модель, сделаем
            # предсказание на отложенных (train) данных:
            y_best_model_predict = inner_best_mlmodel.predict(X_test)

            # Сделаем оценку по каждой из метрик и сохраним:
            for one_score_name, one_score_func, p_control in zip(
                scores_list[0], scores_list[1], scores_list[2]
            ):
                curr_score_value = one_score_func(y_test, y_best_model_predict)
                outer_results[one_score_name].append(curr_score_value)
                if p_control:
                    if max_scorr < curr_score_value:
                        max_scorr = curr_score_value
                        found_best_params = gscv.best_params_

        # summarize the estimated performance of the model

        mean_value = round(mean(outer_results[name_of_control_metric]), 4)
        max_scorr = round(max_scorr, 4)
        real_best_params = rename_params(
            model_type, found_best_params, use_for="mlmodel"
        )

        print("--------------------------------------------------------")
        print("-- Task No.9: Getting score on Nested Cross-Validation --")
        print("Used parameters for GridSearchCV:\n ", grid_space)
        print(f"Control metric name is '{name_of_control_metric}':")
        print(" this metric mean value:", mean_value)
        print(" found best parameters:", real_best_params)
        print(" metric value with best parameters on 'that' Folds:", max_scorr)
        if len(scores_list[0]) > 1:
            print()
            print("Other metrics computed during the circle (mean values):")
        for one_score_name, p_control in zip(scores_list[0], scores_list[2]):
            if not p_control:
                mean_value = round(mean(outer_results[one_score_name]), 4)
                print(f" {one_score_name}:", mean_value)
        print("--------------------------------------------------------")
        print("- Compute metrics on whole data using best model -")
        # Посчитаем метрики на всём датасете с её помощью best_model_:
        y_whole_prediction = inner_best_mlmodel.predict(X_array)
        for one_score_name, one_score_func in\
                zip(scores_list[0], scores_list[1]):
            scope_whole_value = one_score_func(y, y_whole_prediction)
            scope_whole_value = round(scope_whole_value, 4)
            print(f" {one_score_name}:", scope_whole_value)

        # Подготовим данные для другого
        # способа оценки (not_nested_cross_validation)
        # при условии использования best_model_:
        mlmodel = inner_best_mlmodel
        params = real_best_params.copy()

    else:
        pass
    # Вычисление оценок по требованиям задания №7:
    not_nested_cross_validation(
        mlmodel=mlmodel,
        X=X,
        y=y,
        random_state=random_state,
        scores_list=scores_list[0],
        **params,
    )

    return mlmodel


def not_nested_cross_validation(
    mlmodel, X, y, random_state, scores_list, **params
) -> (Pipeline):
    """
    В соответствии с заданием №7 требуется реализация K-fold cross-validation.
    Именно это и реализовано в данной функции.
    """
    # Вычислим разные метрики по результатам cross_val_score
    # с использованием подготовленной модели model:
    cv = KFold(n_splits=3, random_state=random_state, shuffle=True)

    print_parameters = "default parameters" if len(params) == 0 else params
    print("--------------------------------------------------------")
    print("-- Task No.7: Getting score through K-fold cross-validation --")
    print(f"Used parameters for model:\n    {print_parameters}")
    print("Value of selected metrics:")

    for one_score in scores_list:
        scores = cross_val_score(
            mlmodel,
            X,
            y,
            scoring=one_score,
            cv=cv,
            n_jobs=-1
        )
        print_and_save_result(one_score, round(mean(scores), 4))
    print("--------------------------------------------------------")
    return mlmodel


def print_and_save_result(title: str, value: float) -> (None):
    print(f"    {title}: {value}")
    mlflow.log_metric(title, value)


def rename_params(
    model_type: str,
    in_dict: Dict,
    use_for="gscv+pipe"
) -> (Dict):
    """
    Преобразование словаря с параметрами, для того чтобы его можно
    было использовать в GridSearchCV с pipeline, то есть к имени каждого
    параметра добавляется наименование имени элемента pipeline.
    И наоборот, параметры от GridSearchCV с pipeline в обычной моделе,
    то есть ранее добавленная приставка должна быть удалена.
    """
    out_dict = in_dict.copy()
    for one_key in list(out_dict.keys()):
        if use_for == "gscv+pipe":
            new_key = model_type + "__" + one_key
        else:
            new_key = one_key[len(model_type + "__"):]
        out_dict[new_key] = out_dict[one_key]
        out_dict.pop(one_key)

    return out_dict
