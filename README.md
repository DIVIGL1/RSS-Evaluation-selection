# RSS-Evaluation-selection

Задача прогнозирования типа лесного покрова, используемая в качестве соревнования с другими участниками.
Задача является домашним заданием 9-го модуля курса RSSchool.

Данные, на которых осуществляется обучение модели взяты из соревнования "Forest Cover Type Prediction", расположенные по адресу: https://www.kaggle.com/competitions/forest-cover-type-prediction/data

## Описание данных
The study area includes four wilderness areas located in the Roosevelt National Forest of northern Colorado. Each observation is a 30m x 30m patch. You are asked to predict an integer classification for the forest cover type. The seven types are:

1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

The training set (15120 observations) contains both features and the Cover_Type. The test set contains only the features. You must predict the Cover_Type for every row in the test set (565892 observations).

### Данные:
* Elevation - Elevation in meters
* Aspect - Aspect in degrees azimuth
* Slope - Slope in degrees
* Horizontal_Distance_To_Hydrology - Horz Dist to nearest surface water features
* Vertical_Distance_To_Hydrology - Vert Dist to nearest surface water features
* Horizontal_Distance_To_Roadways - Horz Dist to nearest roadway
* Hillshade_9am (0 to 255 index) - Hillshade index at 9am, summer solstice
* Hillshade_Noon (0 to 255 index) - Hillshade index at noon, summer solstice
* Hillshade_3pm (0 to 255 index) - Hillshade index at 3pm, summer solstice
* Horizontal_Distance_To_Fire_Points - Horz Dist to nearest wildfire ignition points
* Wilderness_Area (4 binary columns, 0 = absence or 1 = presence) - Wilderness area designation
* Soil_Type (40 binary columns, 0 = absence or 1 = presence) - Soil Type designation
* Cover_Type (7 types, integers 1 to 7) - Forest Cover Type designation

Для организации command line interface (CLI) использована библиотека click, которая обладает рядом преимуществ:
* Автоматическое создание справки по параметрам командной строки.
* Поддерживает отложенную загрузку подкоманд во время выполнения.
* Меньшее количество кода по сравнению с argparse.
* argparse имеет встроенное поведение, которое пытается угадать, является ли что-то параметром или опцией. Такое поведение становится непредсказуемым при работе со сценариями, в которых не используется часть опций и/или параметров.
* argparse не поддерживает отключение перемежающихся аргументов. Без этой функции невозможно безопасно реализовать вложенный синтаксический анализ, например как в click.

