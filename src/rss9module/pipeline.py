from typing import Dict, Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    model_type: str = "rfc",
    use_scaler: bool = True,
    random_state: int = 42,
    **params: Dict[Any, Any],
) -> (Pipeline):

    print("Scalling used:", use_scaler)
    pipeline_steps = []
    if use_scaler:
        step = ("scaler", StandardScaler())
        pipeline_steps.append(step)

    if model_type == "rfc":
        estimater = RandomForestClassifier(
            n_jobs=-1, random_state=random_state, **params
        )
        stepname = "rfc"
    elif model_type == "knn":
        estimater = KNeighborsClassifier(n_jobs=-1, **params)
        stepname = "knn"
    elif model_type == "svc":
        estimater = SVC(random_state=random_state, **params)
        stepname = "svc"

    step = (stepname, estimater)

    pipeline_steps.append(step)

    return Pipeline(steps=pipeline_steps)
