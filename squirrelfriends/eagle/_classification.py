import numpy as np
import pandas as pd
import sklearn.metrics as skmetrics
from ..utils import check_consistent_length


def compute_binary_metrics(y_true, y_pred, report=True, confusion=True):
    """Display and return all metrics of binary classfication problems.

    Args:
        y_true (array): true value.
        y_pred (array): predicted value.
        report (boolean): if show statistic report.
        confusion (boolean): if show confusion matrix.

    Returns:
        metrics_dict (dict): all metrics of binary classfication problems.
    """

    check_consistent_length(y_true, y_pred)

    tn, fp, fn, tp = skmetrics.confusion_matrix(y_true, y_pred).ravel()
    metrics_dict = {
        "accuracy": skmetrics.accuracy_score(y_true, y_pred),
        "recall": skmetrics.recall_score(y_true, y_pred),
        "precision": skmetrics.precision_score(y_true, y_pred),
        "f1": skmetrics.f1_score(y_true, y_pred),
        "roc_auc": skmetrics.roc_auc_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }

    if report:
        print(skmetrics.classification_report(y_true, y_pred))

    if confusion:
        print("""\t\tConfusion Matrix\n\n\t\tnegative\tpositive\ttotal
                 negative\t{tn}\t\t{fp}\t\t{n}
                 positive\t{fn}\t\t{tp}\t\t{p}
                 total\t{pred_n}\t\t{pred_p}\t\t{total}\n"""
              .format(tn=tn, fp=fp, fn=fn, tp=tp,
                      p=tp + fn, n=tn + fp,
                      pred_n=tn + fn, pred_p=tp + fp,
                      total=tn + fn + tp + fp))

    return metrics_dict


def compute_cumulative_gains(y_true, y_pred, y_prop, data=None, group_n=500):
    """Compute cumulative gains table.

    Args:
        y_true (str or array): data or names of variables in `data`
            represents true values.
        y_pred (str or array): data or names of variables in `data`
            represents predicited values.
        y_prop (str or array): data or names of variables in `data`
            represents propensity of prediction.
        data (DataFrame): DataFrame with columns
            `y_true`, `y_pred` and `y_prop`
        group_n (int): group samples to n groups

    Returns:
        lift (DataFrame): DataFrame with lift chart
    """

    if data is None:
        check_consistent_length(y_true, y_pred, y_prop)

        data = pd.DataFrame({
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prop": y_prop,
        })
        y_true, y_pred, y_prop = "y_true", "y_pred", "y_prop"

    data = data[[y_true, y_pred, y_prop]].copy()

    data.sort_values(by=y_prop, ascending=False, inplace=True)

    # Calculate basic gaimns
    rows = []
    for group in np.array_split(data, group_n):
        rows.append({"n_samples": len(group), "n_trues": sum(group[y_true])})
    lift = pd.DataFrame(rows)

    # Calculate cumulative gains
    total_trues = sum(lift["n_trues"])
    lift["cum_trues"] = lift["n_trues"].cumsum()
    lift["cum_trues_pct"] = lift["cum_trues"] / total_trues * 100

    lift["cum_best"] = np.clip(lift["n_samples"].cumsum(), 0, total_trues)
    lift["cum_best_pct"] = lift["cum_best"] / total_trues * 100

    return lift
