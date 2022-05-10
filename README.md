# Прогнозирование лесного покрова

## Содержание
1. <a href="#g1">Общее описание задачи</a>
1. <a href="#g2">Описание данных</a>
1. <a href="#g3">Установка пакета и получение данных</a>
1. <a href="#g4">Реализация командной строки</a>
1. <a href="#g5">Сведения об использованных моделях машинного обучения</a>
1. <a href="#g6">Сведения о реализованных подходах подбора и генерации признаков (feature selection)</a>
1. <a href="#g7">Метрики использованные для оценки моделей</a>
1. <a href="#g8">Применение MLFlow для сохранения результатов работы пакета</a>
1. <a href="#g9">Обычный и nested варианты Cross-validation и подбор гиперпараметров</a>
1. <a href="#g11">Реализованные тесты</a>
1. <a href="#g12">Форматирование кода в проекте</a>
1. <a href="#g13">Аннотация типов в коде</a>
1. <a href="#g14">Проверка всего подготовленного пакета одной командой</a>

***
<h3 id="g1">Общее описание задачи</h2>
Задача прогнозирования типа лесного покрова (Forest Cover Type Prediction), используемая в качестве соревнования с другими участниками, взята с площадке kaggle (описание здесь: https://www.kaggle.com/competitions/forest-cover-type-prediction/overview).

#### Представленный здесь пакет позволяет:

* проводить обучение нескольких алгоритмов машинного обучения;
* делать выбор из нескольких вариантов подготовки данных;
* на обученной модели делать расчет нескольких метрик;
* сохранять модель по результатам обучения и результаты оценок работы этой модели;
* формировать прогоноз.

Данные, на которых осуществляется обучение модели, расположенны по адресу: https://www.kaggle.com/competitions/forest-cover-type-prediction/data

***
<h3 id="g2">Описание данных</h2>
Область исследования включает в себя четыре зоны дикой природы, расположенные в Национальном лесу Рузвельта на севере Колорадо. Каждое наблюдение представляет собой участок размером 30 х 30 м. Вас просят предсказать целочисленную классификацию для типа лесного покрова. Семь типов:

1. Ель/Пихта
2. Сосна скрученная
3. Сосна жёлтая
4. Тополь/Ива
5. Осина
6. Дугласова пихта
7. Криволесье

Обучающая выборка (15120 наблюдений) содержит как признаки, так и целевой признак (Cover_Type). Тестовая выборка содержит только параметры наблюдений. Вы должны предсказать "Cover_Type" для каждой строки в тестовой выборке (565892 наблюдения).

#### Данные:
* Elevation - Высота в метрах
* Aspect - аспект в градусах азимута
* Slope - Уклон в градусах
* Horizontal_Distance_To_Hydrology - Горизонтальное расстояние до ближайших объектов поверхностных вод
* Vertical_Distance_To_Hydrology - Вертикальное расстояние до ближайших объектов поверхностных вод
* Horizontal_Distance_To_Roadways - Горизонтальное расстояние до ближайшей дороги
* Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
* Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
* Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
* Horizontal_Distance_To_Fire_Points - Горизонтальное расстояние до ближайших точек возгорания лесных пожаров
* Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Обозначение дикой местности
* Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Обозначение типа почвы
* Cover_Type (7 types, integers 1 to 7) - Обозначение типа лесного покрова

***
<h3 id="g3">Установка пакета и получение данных</h2>

#### Важно!

Для установки пакета в первую очередь требуется наличие предустановленного на Вашем компьютере <b>Poetry</b> - инструмента для управления зависимостями в Python проектах (аналог встроенного pip). Информацию по установке можно прочитать <a href="https://python-poetry.org/">на сайте</a> или <a href="https://habr.com/ru/post/593529/">в статье habr</a>.

#### Общие правила установки:

1. Необходимо клонировать данный репозиторий на Ваш компьютер. Для этого можно использовать следующую команду:
<b>git clone https://github.com/DIVIGL1/RSS-Evaluation-selection.git</b>
1. Как уже указывалось выше данные можно получить из соответствсующего соревнования на kaggle. Другим вариантом получения данных является клонирование отдельного репозитория в котором сохранены те самые данные на случай если на площадке kaggle их по какой-то причине поменяют. Выполнить это можно с помощью команды <b>git clone https://github.com/DIVIGL1/data.git</b>

#### Рекомендация по размещению данных
Рекомендуется данные расположить в дирректории <b>data</b>, которую нужно расположить внутри основного репозитория. Это позволит не указывать в командной строке каждый раз место расположения данных, так как пакет по умолчанию подставляет указанный путь к файлам train.csv и test.csv и использует для сохранения моделей и прогнозов.

Как результат, структура каталогов будет выглядить примерно следующим образом:

![DirTree](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/DirTree.PNG?raw=true)

#### Автоматизация установки
Для упрощения процесса установки можно создать файлы со скриптами, как это показано ниже.

* Пакетный файл Windows:

git clone https://github.com/DIVIGL1/RSS-Evaluation-selection.git

cd RSS-Evaluation-selection

call poetry install --no-dev

git clone https://github.com/DIVIGL1/data.git

* Скрипт для bash:

#!/bin/bash

git clone https://github.com/DIVIGL1/RSS-Evaluation-selection.git

cd RSS-Evaluation-selection

poetry install --no-dev

git clone https://github.com/DIVIGL1/data.git

#### Важно!
Обратите внимание, что предложенная выше в скриптах установка пакета использует параметр <b>--no-dev</b> что исключает установку компонент необходимых для разработчика.
***
<h3 id="g4">Реализация командной строки</h2>

Для организации command line interface (CLI) использована библиотека click, которая обладает рядом преимуществ:
* Автоматическое создание справки по параметрам командной строки.
* Поддерживает отложенную загрузку подкоманд во время выполнения.
* Меньшее количество кода по сравнению с argparse.
* argparse имеет встроенное поведение, которое пытается угадать, является ли что-то параметром или опцией. Такое поведение становится непредсказуемым при работе со сценариями, в которых не используется часть опций и/или параметров.
* argparse не поддерживает отключение перемежающихся аргументов. Без этой функции невозможно безопасно реализовать вложенный синтаксический анализ, например как в click.

***
<h3 id="g5">Сведения об использованных моделях машинного обучения</h2>

Использованы следующие модели (алгоритмы) машинного обучения:
1. RandomForestClassifier
1. KNeighborsClassifier
1. C-Support Vector Classification

***
<h3 id="g6">Сведения о реализованных подходах подбора и генерации признаков (feature selection)</h2>

Для обработки данных использовно четыре варианта, каждый из которых выбирается через параметр командной строки:

* Вариант 0: Исходные данные без каких-либо модификаций (--fe-type 0)
* Вариант 1: Созданы столбцы представляющие из себя степени от 2 до 4 относительно всех исходных и доволнительно создано несколько столбцов, представляющих собой суммы от исходных (--fe-type 1)
* Вариант 2: Произведено уменьшение размерности до 50 компонент с использованием PCA из sklearn.decomposition (--fe-type 2)
* Вариант 3: Комбинация вариантов 1 и 2 (--fe-type 3)

***
<h3 id="g7">Метрики использованные для оценки моделей</h2>

Для оценки результатов работы моделей использованы следующие метрики из библиотеки sklearn.metrics:
1. r2
1. accuracy (основная метрика),
1. homogeneity_score
1. rand_score

***
<h3 id="g8">Применение MLFlow для сохранения результатов работы пакета</h2>

Описываемый здесь пакет использует библиотеку MLFlow (устанавливается автоматически при инсталляции данного пакета). MLFlow позволяет сохранять и визуализировать результаты работы и применяемые параметры моделей машинного обучения.
На приведённом ниже скрине представлены результаты работы данного пакета, где использованы:
* Разные наборы гиперпараметров для каждой модели (представлены по четыре - один из них параметры по умолчанию).
* Разные методы разработки признаков для каждой модели (представлено 4 - один из них данные без обработки).
* Разные модели ML (представлены три: <b>rfc = RandomForestClassifier, knn = KNeighborsClassifier, svc = C-Support Vector Classification)</b>.

![MLFlow](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/Experiments.PNG?raw=true)

#### Обратите внимание: В данный пакет входят два файла для запуска из командной строки (examples.bat и examples.sh) позволяющие Вам получить представленные результаты после инсталляции пакета.

***
<h3 id="g9">Обычный и nested варианты Cross-validation и подбор гиперпараметров</h2>

Применяемый в предыдущем разделе <b>обычный подход к Cross-validation</b> требует ручной настройки гиперпараметров через соответствующие пареметры командной строки. Рассматриваемый тут пакет предполагает так же возможность автоматического подбора гиперпараметров для каждой модели по сетке с использованием <b>nested Cross-validation</b>.

Для раализации сказанного выше введён параметр командной строки --nested-cv, который по умолчанию равен <b>False</b> и в этом случае используется используется функциональность <b>обычного варианты Cross-validation</b>. В случае если этот параметр установлен в значение <b>True</b>, то используется <b>nested Cross-validation</b> с подбором гиперпараметров (в этом случае все заданные в ручную параметры игнорируются).

При этом в результате работы этого блока кода:
1. для каждой из трёх моделей используется своя сетка подбора параметров с использованием GridSearchCV (отображается на экране во время исполнения кода);
1. производится расчет каждой из выбранных метрик (отображается на экране);
1. по сетке вычисляется лучшая модель (на экране отображаются параметры подобранные для лучшей модели);
1. используя лучшую модель, формируется предсказание на всём наборе данных и для сравнения расчитываются те же выбранные метрики (отображается на экране);
1. используя лучшую модель, производится расчет по принципам задания №7 (значения вычестленных метрик выводятся на экран, а также производится сохранение результатов в MLFlow.

#### Обратите внимание: В данный пакет входят два файла для запуска из командной строки (examples-nested.bat и examples-nested.sh) позволяющие Вам получить представленные ниже результаты после инсталляции пакета.

Ниже представлены изображения экранов отработавших алгоритмов:


#### RandomForestClassifier:
![RandomForestClassifier](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/rfc.PNG?raw=true)
***

#### KNeighborsClassifier:
![KNeighborsClassifier](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/knn.PNG?raw=true)
***

#### C-Support Vector Classification:
![C-Support Vector Classification](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/svc.PNG?raw=true)

<h3 id="g11">Реализованные тесты</h2>
Для обеспечения проверки результатов функцинирования кода после внесения изменений необходимо проводить проверку его на тестовых примерах. В данном случае это выполнено на библиотеке pytest c реализацией следующих проверок:

* проверка на фиктивный параметр;
* проверка на граничные значения random_state;
* проверка на запуск с отсутствующими данными;
* проверка на возможность сохранения модели;
* проверка на возможность сохранения прогноза;
* проверка правилоности прогноза на тестовом примере.

Стоит отметить, что кроме этих тестов все параметры, применяемые в командной строке, контролируются на правильность их ввода с помощью библиотеки Click. Так, например, <b>--c-param</b> проверяется не только на тип (float), но и то что бы быть положительным, а <b>--algorithm</b> на соответствие значению из списка ("auto", "ball_tree", "kd_tree", "brute").

Результаты прохождения тестов представлены на следующем скрине:

![AfterBlack](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/tests.PNG?raw=true)

Самостоятельно вызвать тесты можно выполнив команду: <b>poetry run pytest -v</b> (или, что тоже самое, запустив скрипт-файл: tests)
#### Важно!
Для выполнения тестов, необходмо, что бы установка пакета происходила:
* либо без ключа: --no-dev
* либо дополнительно были доустановлены компоненты для разработчика командой: poetry install --dev

<h3 id="g12">Форматирование кода в проекте</h2>
При форматировании кода использовалась утилита black, которая внесла изменение в некоторые файлы. Это видно на следующем скрине:

![AfterBlack](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/AfterBlack.png?raw=true)

После обработки кода утилитой black, потребовалось дополнительно подкорректировать полученный результат, чтобы избавиться от предупреждейний выданных flake8. Большая часть замечаний связана с длинной строки:

![flake8](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/flake8.PNG?raw=true)

Замечания от flake8 были устранены введением исключений E203, E501 и W503 (для чего создан конфигурационный файл .flake8 и внесены правки в файле настроек VSCode):

![Afterflake8](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/Afterflake8.PNG?raw=true)

#### Важно!
Для применения данных команд необходмо, чтобы установка пакета происходила:
* либо без ключа: --no-dev
* либо дополнительно были доустановлены компоненты для разработчика командой: poetry install --dev

<h3 id="g13">Аннотация типов в коде</h2>
Все реализованные в коде методы снабжены аннотациями типов и правильно используются во всем коде. Проверка осуществена с помощью mypy и пройдена успешно. Это видно на следующем скрине:

![mypy](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/mypy.PNG?raw=true)

#### Важно!
Для применения данной команды необходмо, чтобы установка пакета происходила:
* либо без ключа: --no-dev
* либо дополнительно были доустановлены компоненты для разработчика командой: poetry install --dev

<h3 id="g14">Проверка всего подготовленного пакета одной командой</h2>
Все предыдущие проверки можно объединить и выполнить одной командой (nox).

В конфигурационнойм файле noxfile.py прописаны все необходимые сессии для осуществления проверки. Процесс проверки изобращён на следующем скрине:

![nox](https://github.com/DIVIGL1/RSS-Evaluation-selection/blob/main/picts/nox.PNG?raw=true)

#### Важно!
Для применения данной команды необходмо, чтобы установка пакета происходила:
* либо без ключа: --no-dev
* либо дополнительно были доустановлены компоненты для разработчика командой: poetry install --dev
