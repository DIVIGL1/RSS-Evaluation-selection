from click.testing import CliRunner
import pytest
import pandas as pd
import numpy as np

from rss9module.train import train


@pytest.fixture
def runner() -> CliRunner:
    """Fixture providing click runner."""
    return CliRunner()


def test_error_for_invalid_parameter_name(runner: CliRunner) -> (None):
    """
    Проверка на функциониование Click
    Будет формальная ошибка, отловленная Click.
    """
    result = runner.invoke(train, ["--unknown-parameter"])
    assert result.exit_code == 2
    assert "No such option: --unknown-parameter" in result.output


def test_random_state_bounds(runner: CliRunner) -> (None):
    """
    Проверка на функциониование Click: проверка корректности параметра
    Будет формальная ошибка, отловленная Click.
    """
    result = runner.invoke(train, ["--random-state", "-1"])
    assert result.exit_code == 2
    assert (
        "Error: Invalid value for '--random-state': -1 is not in the range x>=0."
        in result.output
    )


def test_data_file_exist(runner: CliRunner) -> (None):
    """
    Проверка на функциониование Click: проверка наличия файла.
    Будет формальная ошибка, отловленная Click.
    """
    filename = "dflksae;wke23023902834"
    result = runner.invoke(train, ["--dataset-path", filename])
    assert result.exit_code == 2
    assert (
        f"Error: Invalid value for '-d' / '--dataset-path': File '{filename}' does not exist."
        in result.output
    )

    result = runner.invoke(train, ["--test-data-path", filename])
    assert result.exit_code == 2
    assert (
        f"Error: Invalid value for '-t' / '--test-data-path': File '{filename}' does not exist."
        in result.output
    )


def test_enable_to_save_models(runner: CliRunner) -> (None):
    """
    Проверка возможности сохранить модель.
    """
    dirname = "z:/:/:///"
    # Прверка на наличие каталога:
    result = runner.invoke(train, ["--save-model-path", f"zd:/{dirname}/"])
    assert result.exit_code in [2, 5]
    assert "Missing directory to save models:" in result.output

    filename = "@<>!"
    # Прверка на наличие каталога:
    result = runner.invoke(train, ["--save-model-path", filename])
    assert result.exit_code in [2, 5]
    assert "Incorrect symbol in models's filename" in result.output


def test_enable_to_save_prediction(runner: CliRunner) -> (None):
    """
    Проверка возможности сохранить результат прогноза.
    """
    dirname = "z:/:/:///"
    # Прверка на наличие каталога:
    result = runner.invoke(train, ["--predicted-data-path", f"zd:/{dirname}/"])
    assert result.exit_code in [2, 5]
    assert "Missing directory to save predictions:" in result.output

    filename = "@<>!"
    # Прверка на наличие каталога:
    result = runner.invoke(train, ["--predicted-data-path", filename])
    assert result.exit_code in [2, 5]
    assert "Incorrect symbol in predictions's filename" in result.output


def test_model_prediction(runner: CliRunner) -> (None):
    """
    Проверка функционирования модели.
    """
    with runner.isolated_filesystem():
        # Создадим данные для обучения:
        columns = "Id,Elevation,Aspect,Slope,Horizontal_Distance_To_Hydrology,Vertical_Distance_To_Hydrology,Horizontal_Distance_To_Roadways,Hillshade_9am,Hillshade_Noon,Hillshade_3pm,Horizontal_Distance_To_Fire_Points,Wilderness_Area1,Wilderness_Area2,Wilderness_Area3,Wilderness_Area4,Soil_Type1,Soil_Type2,Soil_Type3,Soil_Type4,Soil_Type5,Soil_Type6,Soil_Type7,Soil_Type8,Soil_Type9,Soil_Type10,Soil_Type11,Soil_Type12,Soil_Type13,Soil_Type14,Soil_Type15,Soil_Type16,Soil_Type17,Soil_Type18,Soil_Type19,Soil_Type20,Soil_Type21,Soil_Type22,Soil_Type23,Soil_Type24,Soil_Type25,Soil_Type26,Soil_Type27,Soil_Type28,Soil_Type29,Soil_Type30,Soil_Type31,Soil_Type32,Soil_Type33,Soil_Type34,Soil_Type35,Soil_Type36,Soil_Type37,Soil_Type38,Soil_Type39,Soil_Type40,Cover_Type".split(
            ","
        )
        data = ""
        for num in range(11, 80):
            data += (
                ";"
                + str(num - 10)
                + " "
                + (str(num * 4.9 / 5) + " ") * 10
                + (str((num + 1) % 2) + " ") * 44
                + str(int(num // 10))
            )

        control_df = pd.DataFrame(np.matrix(data[1:]), columns=columns)
        control_df.to_csv("control_train.csv", index=False)

        # Создадим данные для тестирования:
        data = ""
        for num in range(11, 80):
            data += (
                ";"
                + str(num - 10)
                + " "
                + (str(num * 4.9 / 5) + " ") * 10
                + (str((num + 1) % 2) + " ") * 44
                + str(int(num // 10))
            )

        control_df = pd.DataFrame(np.matrix(data[1:]), columns=columns)
        control_df.drop(["Cover_Type"], axis=1, inplace=True)
        control_df.to_csv("control_test.csv", index=False)

        # Учим, и предсказываем:
        result = runner.invoke(
            train,
            [
                "--dataset-path",
                "control_train.csv",
                "--predicted-data-path",
                "prediction.csv",
                "--save-model-path",
                "model.save",
                "--test-data-path",
                "control_test.csv",
                "--do-prediction",
                "True",
                "--without-preffix",
                "True",
            ],
        )
        assert result.exit_code == 0

        df = pd.read_csv("prediction.csv", sep=",")
        assert round(df.Cover_Type.mean(), 7) == 4.0434783
        assert round(df.Cover_Type.std(), 7) == 1.9958397
