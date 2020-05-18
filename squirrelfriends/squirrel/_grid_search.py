import itertools
import logging

import numpy as np
from sklearn.model_selection import KFold

from ._lightgbm import lgbSquirrel

logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%m/%d/%Y %I:%M:%S %p",
                    level=logging.INFO)


def _parse_param_grid(grid):
    """Parse param grid to a list of all possible params combination.

    Args:
        grid (dict): (param name: list of value).

    Returns:
        res (list): dictionary of params.
    """

    items = sorted(grid.items())
    keys, values = zip(*items)
    res = []
    for v in itertools.product(*values):
        params = dict(zip(keys, v))
        res.append(params)
    return res


def search_cv(model, grid, X, y, eval_func=None, n_splits=4, seed=2020):
    """Grid search.

    Args:
        model (:obj:): model to search hyper-parameter.
        grid (dict): (param name: list of value).
        X (DataFrame): training data.
        y (DataFrame): test data.
        eval_func (func): callable function in evaluation,
            eval_func(y_true, y_pred).
        n_splits (int): number of splits in Kflod().
        seed (int): random seed in Kfold().

    Returns:
        search_results (dict): result og grid search,
            - parameter (dict): param dictionary,
            - cv_results (dict): dict of cv results,
            - eval_score (float) mean of evalation score.
    """

    params = _parse_param_grid(grid)
    search_results = []

    for param in params:
        logging.info("param: " + str(param))

        # Search cv on lightgbm
        if hasattr(model, "__model__") and model.__model__ == "lgbSquirrel":
            model.eval_func = eval_func
            if "num_iterations" in param.keys():
                model.build_lgb(param)
            else:
                model.build_sklgb(param)
            cv_results = model.kfold_cv_lgb_train(
                X=X, y=y, n_splits=n_splits, seed=seed)

        # Search cv on other sklearn model
        else:
            cv_model = model(**param)

            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

            cv_results = []

            for fold, (ind_train, ind_test) in enumerate(kfold.split(X, y)):
                logging.info("cv: %d out of %d" % (fold + 1, n_splits))

                X_train, X_test = X.loc[ind_train, :], X.loc[ind_test, :]
                y_train, y_test = y[ind_train], y[ind_test]

                cv_model.fit(X_train, y_train)

                val_pre = cv_model.predict(X_test)
                eval_score = eval_func(y_test, val_pre)

                if hasattr(cv_model, "feature_importances_"):
                    importances = cv_model.feature_importances_
                else:
                    importances = None

                model_result = {
                    "model": model,
                    "importances": importances,
                    "eval_score": eval_score
                }
                cv_results.append(model_result)

        eval_scores = [result["eval_score"] for result in cv_results]

        if None not in eval_scores:
            logging.info("average score is %f, std is %f" %
                         (np.mean(eval_scores), np.std(eval_scores)))

        one_search = {
            "parameter": param,
            "cv_results": cv_results,
            "eval_score": np.mean(eval_scores)
        }
        search_results.append(one_search)

    return search_results
