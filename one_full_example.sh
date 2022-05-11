#!/bin/bash
poetry run train --dataset-path data/train.csv --test-data-path data/test.csv --do-prediction True --predicted-data-path data/submission.csv --save-model-path data/model.joblib --fe-type 3 --model-type "rfc" --n-estimators 10 --criterion "entropy" --max-features "sqrt"
