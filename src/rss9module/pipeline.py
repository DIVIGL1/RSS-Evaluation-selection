from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(
    model_type: str = "rfc",
    use_scaler: bool = True,
    random_state: int = 42,
    **params,
) -> Pipeline:

    print("Scalling used:", use_scaler)
    pipeline_steps = []
    if use_scaler:
        step = ("scaler", StandardScaler())
        pipeline_steps.append(step)

    if model_type == "rfc":
        step = (
            "rfc",
            RandomForestClassifier(
                n_jobs=-1,
                random_state=random_state,
                **params
            )
        )
    elif model_type == "knn":
        step = ("knn", KNeighborsClassifier(n_jobs=-1, **params))
    elif model_type == "svc":
        step = ("svc", SVC(random_state=random_state, **params))

    pipeline_steps.append(step)

    return Pipeline(steps=pipeline_steps)
