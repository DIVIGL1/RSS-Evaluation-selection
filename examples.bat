call poetry run train --fe-type 0 --model-type "rfc"
call poetry run train --fe-type 1 --model-type "rfc"
call poetry run train --fe-type 2 --model-type "rfc"
call poetry run train --fe-type 3 --model-type "rfc"

call poetry run train --fe-type 0 --model-type "knn"
call poetry run train --fe-type 1 --model-type "knn"
call poetry run train --fe-type 2 --model-type "knn"
call poetry run train --fe-type 3 --model-type "knn"

call poetry run train --fe-type 0 --model-type "svc"
call poetry run train --fe-type 1 --model-type "svc"
call poetry run train --fe-type 2 --model-type "svc"
call poetry run train --fe-type 3 --model-type "svc"


call poetry run train --fe-type 3 --model-type "rfc" --n-estimators 10 --criterion "entropy" --max-features "sqrt"
call poetry run train --fe-type 3 --model-type "rfc" --n-estimators 100 --max-depth 10 --max-features "log2"
call poetry run train --fe-type 3 --model-type "rfc" --n-estimators 200 --criterion "gini" --random-state 10 --max-features "auto"

call poetry run train --fe-type 3 --model-type "knn" --n-neighbors 3 --weights "distance" --algorithm "ball_tree"
call poetry run train --fe-type 3 --model-type "knn" --n-neighbors 6 --weights "uniform" --algorithm "auto"
call poetry run train --fe-type 3 --model-type "knn" --n-neighbors 8

call poetry run train --fe-type 3 --model-type "svc" --c-param 0.1
call poetry run train --fe-type 3 --model-type "svc" --c-param 10 --kernel "rbf"  --shrinking True --tol 0.0001
call poetry run train --fe-type 3 --model-type "svc" --c-param 20 --kernel "poly" --shrinking False --tol 0.01
