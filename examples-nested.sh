#!/bin/bash
poetry run train --fe-type 3 --model-type "rfc" --nested-cv True
poetry run train --fe-type 3 --model-type "knn" --nested-cv True
poetry run train --fe-type 3 --model-type "svc" --nested-cv True
