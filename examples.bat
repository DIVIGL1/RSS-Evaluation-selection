call poetry run train --fe-type 0 --model "rfc"
call poetry run train --fe-type 1 --model "rfc" --n-estimators 200 --criterion "entropy" --max-depth 10 --max-features "log2"
call poetry run train --fe-type 2 --model "rfc" --n-estimators 50 --max-depth 10 --max-features "sqrt"

call poetry run train --fe-type 0 --model "knn" 
call poetry run train --fe-type 1 --model "knn" --n-neighbors 3 --weights "distance" --algorithm "ball_tree"
call poetry run train --fe-type 2 --model "knn" --n-neighbors 7 --weights "uniform"

call poetry run train --fe-type 0 --model "svc" 
call poetry run train --fe-type 1 --model "svc" --c-param 10 --kernel "rbf"  --tol 0.0001
call poetry run train --fe-type 2 --model "svc" --c-param 20 --kernel "poly" --shrinking False --tol 0.01