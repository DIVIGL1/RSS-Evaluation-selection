from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def create_pipeline(
    random_state: int = 42,
    use_scaler: bool = True,
    n_estimators: int = 100,
    criterion: str = "gini",
    max_depth: int = None,
    max_features: str = "auto",
) -> Pipeline:

    pipeline_steps = []
    if use_scaler:
        step = ("scaler", StandardScaler())
        pipeline_steps.append(step)

    params = {
        "n_estimators": n_estimators,
        "criterion": criterion,
        "max_depth": max_depth,
        "max_features": max_features,
        "random_state": random_state,
    }
    step = ("rfc", RandomForestClassifier(**params))
    pipeline_steps.append(step)

    return Pipeline(steps=pipeline_steps)
