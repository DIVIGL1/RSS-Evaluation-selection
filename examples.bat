call poetry run train --fe-type 0 --model "rfc"
call poetry run train --fe-type 1 --model "rfc"
call poetry run train --fe-type 2 --model "rfc"
call poetry run train --fe-type 3 --model "rfc"

call poetry run train --fe-type 0 --model "knn"
call poetry run train --fe-type 1 --model "knn"
call poetry run train --fe-type 2 --model "knn"
call poetry run train --fe-type 3 --model "knn"

call poetry run train --fe-type 0 --model "svc"
call poetry run train --fe-type 1 --model "svc"
call poetry run train --fe-type 2 --model "svc"
call poetry run train --fe-type 3 --model "svc"


call poetry run train --fe-type 3 --model "rfc" --n-estimators 100 --max-depth 10 --max-features "log2"
call poetry run train --fe-type 3 --model "rfc" --n-estimators 200 --criterion "entropy" --max-features "sqrt"
call poetry run train --fe-type 3 --model "rfc" --n-estimators 500 --random-state 10 --max-features "auto"

call poetry run train --fe-type 3 --model "knn" --n-neighbors 3 --weights "distance" --algorithm "ball_tree"
call poetry run train --fe-type 3 --model "knn" --n-neighbors 5 --weights "uniform" --algorithm "auto"
call poetry run train --fe-type 3 --model "knn" --n-neighbors 10

call poetry run train --fe-type 3 --model "svc" --c-param 0.1
call poetry run train --fe-type 3 --model "svc" --c-param 10 --kernel "rbf"  --tol 0.0001
call poetry run train --fe-type 3 --model "svc" --c-param 20 --kernel "poly" --shrinking False --tol 0.01
