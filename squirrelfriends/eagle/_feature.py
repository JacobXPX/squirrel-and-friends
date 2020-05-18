import numpy as np
import pandas as pd


def compute_cv_feature_importances(*feature_importances):
    """Compute the average feature importance.

    Args:
        *feature_importances: list or tuple of input objects.
            Objects are feature importances of each cv.

    Returns:
        importances (DataFrame): feature importance table
            with average importance score
    """

    importances = feature_importances[0].rename(
        columns={"importance": "importance_0"})

    for i, df in enumerate(feature_importances[1:]):
        tmp = df.rename(columns={"importance": "importance_%d" % (i + 1)})
        importances = importances.merge(tmp, on="feature", how="outer")

    importance_cols = [c for c in importances.columns if c != "feature"]
    importances["ave_importance"] = importances[importance_cols].mean(axis=1)

    return importances
