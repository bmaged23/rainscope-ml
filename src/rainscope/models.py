from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from .data import FEATURES, CATEGORICAL


def preprocessing(kind="onehot"):
    numeric = [c for c in FEATURES if c not in CATEGORICAL]
    num = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
        ]
    )
    cat = (
        OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        if kind == "onehot"
        else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    )
    return ColumnTransformer(
        [("numeric", num, numeric), ("category", cat, CATEGORICAL)], sparse_threshold=0
    )


def make_model(name, cfg):
    seed = cfg["seed"]
    if name == "climatology":
        model = DummyClassifier(strategy="prior")
    elif name == "logistic":
        model = LogisticRegression(max_iter=1000, C=1.0, random_state=seed)
    elif name == "random_forest":
        model = RandomForestClassifier(
            n_estimators=cfg["random_forest_trees"],
            max_depth=18,
            min_samples_leaf=8,
            n_jobs=cfg["workers"],
            random_state=seed,
        )
    elif name == "hist_gradient_boosting":
        # Explicit one-hot input avoids imposing artificial category order.
        model = HistGradientBoostingClassifier(
            max_iter=cfg["boosting_iterations"],
            max_leaf_nodes=31,
            learning_rate=0.08,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=seed,
        )
    else:
        raise ValueError(name)
    return Pipeline([("preprocess", preprocessing()), ("model", model)])
