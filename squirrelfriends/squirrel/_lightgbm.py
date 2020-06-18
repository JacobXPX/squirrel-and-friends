import logging

import random
import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.model_selection import KFold, GroupKFold

from ..eagle import compute_cv_feature_importances


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)


def _log_evals_result(evals_result, metric, verbose_eval, estimator=None):
    """Log the evaluation result.

    Args:
        evals_result (dict): evaluation results.
        metric (str): evaluation metric name.
        verbose_eval (int): log evaluation result per `verbose_eval` round.
        estimator (`object`): model object.
    """

    for i in range(1, len(evals_result["training"][metric])//verbose_eval):
        eps = i * verbose_eval
        logging.info("{}: {} training: {} \t valid: {}".format(
            eps, metric,
            evals_result["training"][metric][eps-1],
            evals_result["valid_1"][metric][eps-1]))

    # log the last iteration
    logging.info("{}: {} training: {} \t valid: {}".format(
        len(evals_result["training"][metric]), metric,
        evals_result["training"][metric][-1],
        evals_result["valid_1"][metric][-1]))

    # show the best iteration
    if estimator:
        best_score = estimator.best_score
        logging.info("best_score at {}: {} training: {} \t valid: {}".format(
            estimator.best_iteration, metric,
            best_score["training"][metric],
            best_score["valid_1"][metric]))


class lgbSquirrel(object):
    """Class to implement lightgbm by using either lightgbm or sklearn API.
    It is composed with two parts:
        1. build the model.
        2. train the model.

    Attributes:
        features(list of str): column names of feature.
        cat_features(list of str): categorical features,
            specified in lgb.Dataset.
        task (str): {"regression",  "classification"},
            type of machine learning task.
        eval_func (func): callable function in evaluation,
            eval_func(y_true, y_pred).
        early_stopping_rounds (int): early stopping rounds.
        verbose_eval (int): log evaluation result per `verbose_eval` round.
        sk_api (boolean): if using sklearn API.
        feature_importances_ (DataFrame): feature importance table.
    """

    def __init__(self, features="auto", cat_features="auto",
                 task="classification", eval_func=None,
                 early_stopping_rounds=50, verbose_eval=10):
        self.__model__ = "lgbSquirrel"
        self.features = features
        self.cat_features = cat_features
        self.task = task
        self.eval_func = eval_func
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose_eval = verbose_eval
        self.sk_api = None
        self.feature_importances_ = None

    def build_sklgb(self, trn_params):
        """Parse parameters in to build the sklearn lgb.
        http://lightgbm.readthedocs.io/en/latest/Parameters.html

        Args:
            trn_params (dict): params to build model obj.
            boosting_type : string, optional (default="gbdt")
                "gbdt", traditional Gradient Boosting Decision Tree.
                "dart", Dropouts meet Multiple Additive Regression Trees.
                "goss", Gradient-based One-Side Sampling.
                "rf", Random Forest.
            num_leaves : int, optional (default=31)
                Maximum tree leaves for base learners.
            max_depth : int, optional (default=-1)
                Maximum tree depth for base learners, <=0 means no limit.
            learning_rate : float, optional (default=0.1)
                Boosting learning rate.
                You can use `callbacks` parameter of `fit` method
                    to shrink/adapt learning rate
                    in training using `reset_parameter` callback.
                Note, that this will ignore the `learning_rate`
                    argument in training.
            n_estimators : int, optional (default=100)
                Number of boosted trees to fit.
            subsample_for_bin : int, optional (default=200000)
                Number of samples for constructing bins.
            objective : string, callable or None, optional (default=None)
                Specify the learning task and the corresponding
                    learning objective or a custom objective function
                    to be used (see note below).
                Default: "regression" for LGBMRegressor, "binary" or
                    "multiclass" for LGBMClassifier,
                    "lambdarank" for LGBMRanker.
            class_weight : dict, "balanced" or None, optional (default=None)
                Weights associated with classes
                    in the form `{class_label: weight}`.
                Use this parameter only for multi-class classification task;
                    for binary classification task you may use `is_unbalance`
                    or `scale_pos_weight` parameters.
                Note, that the usage of all these parameters will result in
                    poor estimates of the individual class probabilities.
                You may want to consider performing probability calibration
                (https://scikit-learn.org/stable/modules/calibration.html)
                    of your model.
                The "balanced" mode uses the values of y
                    to automatically adjust weights
                    inversely proportional to class frequencies
                    in the input data
                    as `n_samples / (n_classes * np.bincount(y))`.
                If None, all classes are supposed to have weight one.
                Note, that these weights will be multiplied with
                    `sample_weight` (passed through the `fit` method)
                    if `sample_weight` is specified.
            min_split_gain : float, optional (default=0.)
                Minimum loss reduction required to make a further partition on
                    a leaf node of the tree.
            min_child_weight : float, optional (default=1e-3)
                Minimum sum of instance weight (hessian)
                needed in a child (leaf).
            min_child_samples : int, optional (default=20)
                Minimum number of data needed in a child (leaf).
            subsample : float, optional (default=1.)
                Subsample ratio of the training instance.
            subsample_freq : int, optional (default=0)
                Frequence of subsample, <=0 means no enable.
            colsample_bytree : float, optional (default=1.)
                Subsample ratio of columns when constructing each tree.
            reg_alpha : float, optional (default=0.)
                L1 regularization term on weights.
            reg_lambda : float, optional (default=0.)
                L2 regularization term on weights.
            random_state : int, RandomState object or None,
                optional (default=None) Random number seed.
                If int, this number is used to seed the C++ code.
                If RandomState object (numpy), a random integer
                    is picked based on its state to seed the C++ code.
                If None, default seeds in C++ code are used.
            n_jobs : int, optional (default=-1)
                Number of parallel threads.
            silent : bool, optional (default=True)
                Whether to print messages while running boosting.
            importance_type : string, optional (default="split")
                The type of feature importance to be filled into
                    `feature_importances_`.
                If "split", result contains numbers of times the feature
                    is used in a model.
                If "gain", result contains total gains of splits
                    which use the feature.
        """

        self.sk_api = True
        self.trn_params = trn_params

    def build_lgb(self, trn_params,
                  dtrain_params=None, dtest_params=None,
                  weight=None):
        """Parse parameters in to build lgb.
        https://github.com/microsoft/LightGBM/blob/master/docs/Parameters.rst

        Args:
            trn_params (dict): params parsed into lgb.train().
            dtrain_params (dict): params of lgb.Dataset() for training data.
            dtest_params (dict): params of lgb.Dataset() for testing data,
                if None, directly use `dtrain_params`.
            weight (array-like): weight for each instance.
        """

        self.sk_api = False
        self.trn_params = trn_params
        self.dtrain_params = dtrain_params
        self.dtest_params = dtest_params if dtest_params else dtrain_params
        self.weight = weight

    def _train_sklgb(self, data):
        """Train the model using sklearn lightgbm API.

        Args:
            data ((tuple of DataFrame)): training and testing data.

        Returns:
            model_result (dict): result of trained model.
                - model (:obj:): trained model,
                - importances (DataFrame): feature importance table,
                - eval_score (float): score of model using `eval_func`.
        """

        logging.info("using lightgbm sklearn api...")

        X_train, X_test, y_train, y_test = data
        X_train, X_test = X_train[self.features], X_test[self.features]

        if self.task == "regression":
            model = LGBMRegressor(**self.trn_params, n_jobs=-1)
        elif self.task == "classification":
            model = LGBMClassifier(**self.trn_params, n_jobs=-1)

        # Train model
        model.fit(X_train, y_train,
                  feature_name=self.features,
                  categorical_feature=self.cat_features,
                  eval_set=[(X_train, y_train), (X_test, y_test)],
                  early_stopping_rounds=self.early_stopping_rounds,
                  verbose=self.verbose_eval)

        # Calculate the feature importances
        importances = pd.DataFrame()
        importances["feature"] = X_train.columns.values.tolist()
        importances["importance"] = model.feature_importances_

        # Calculate eval_func function eval_score
        if self.eval_func:
            if self.task == "classification":
                # Return the predicted probability
                # for each class for each sample.
                y_test_pred = model.predict_proba(X_test)
            else:
                y_test_pred = model.predict(X_test)

            eval_score = self.eval_func(y_test, y_test_pred)
            logging.info("score on validation is %f" % eval_score)
        else:
            eval_score = None

        model_result = {
            "model": model,
            "importances": importances,
            "eval_score": eval_score
        }
        return model_result

    def _train_lgb(self, data):
        """Train the model using lightgbm.

        Args:
            data ((tuple of DataFrame)): training and testing data.

        Returns:
            model_result (dict): result of trained model.
                - model (:obj:): trained model,
                - importances (DataFrame): feature importance table,
                - eval_score (float): score of model using `eval_func`.
        """

        logging.info("using lightgbm api...")

        X_train, X_test, y_train, y_test = data
        X_train, X_test = X_train[self.features], X_test[self.features]

        # Prepare lgb.Dataset
        d_train = lgb.Dataset(X_train, y_train,
                              feature_name=self.features,
                              categorical_feature=self.cat_features,
                              params=self.dtrain_params,
                              weight=self.weight)
        d_test = lgb.Dataset(X_test, y_test,
                             feature_name=self.features,
                             categorical_feature=self.cat_features,
                             params=self.dtest_params,
                             weight=self.weight)

        # Train model
        evals_result = {}
        model = lgb.train(self.trn_params, d_train,
                          feature_name=self.features,
                          categorical_feature=self.cat_features,
                          valid_sets=[d_train, d_test],
                          early_stopping_rounds=self.early_stopping_rounds,
                          evals_result=evals_result,
                          verbose_eval=self.verbose_eval)

        # Calculate the feature importances
        importances = pd.DataFrame()
        importances["feature"] = X_train.columns.values.tolist()
        importances["importance"] = model.feature_importance()

        _log_evals_result(evals_result, self.verbose_eval, model)

        # Calculate eval_func function eval_score
        if self.eval_func:
            y_test_pred = model.predict(X_test)
            eval_score = self.eval_func(y_test, y_test_pred)
            logging.info("score on validation is %f" % eval_score)
        else:
            eval_score = None

        model_result = {
            "model": model,
            "importances": importances,
            "eval_score": eval_score
        }
        return model_result

    def train(self, data):
        """Train the model.

        Args:
            data (tuple of DataFrame): training and testing data.

        Returns:
            model_result (dict): result of trained model.
                - model (:obj:): trained model,
                - importances (DataFrame): feature importance table,
                - eval_score (float): score of model using `eval_func`.
        """

        if self.sk_api is None:
            raise Exception("Please call build function first.")

        if self.sk_api:
            model_result = self._train_sklgb(data)
        else:
            model_result = self._train_lgb(data)

        self.feature_importances_ = model_result["importances"]

        return model_result

    def kfold_cv_lgb_train(self, X, y, n_splits, seed, fold_type="KFold", groups=None):
        """Train lightGBM by Kfold cross validation.

        Args:
            X (DataFrame): training data.
            y (DataFrame): test data.
            n_splits (int): number of splits in Kflod().
            seed (int): random seed in Kfold().
            fold_type (str): type of kfold algorithm, KFold or GroupKFold.
            groups (array): group labels for the samples used
                while splitting the dataset into train/test set.

        Returns:
            models (list of :obj:): list of models for each cv.
            importances_tables (list of DataFrame):
                list of feature importance tables for each cv.
            eval_scores (list of float): list of score for each cv
        """

        if self.sk_api is None:
            raise Exception("Please call build function first.")

        if fold_type == "KFold":
            kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        elif fold_type == "GroupKFold":
            kfold = GroupKFold(n_splits=n_splits)

        cv_results = []

        for fold, (ind_train, ind_test) in enumerate(kfold.split(X, y, groups)):
            seed_everything(seed)
            logging.info("cv: %d out of %d" % (fold + 1, n_splits))
            logging.info("training: {}, testing: {}".format(
                len(ind_train), len(ind_test)))

            X_train, X_test = X.loc[ind_train, :], X.loc[ind_test, :]
            y_train, y_test = y[ind_train], y[ind_test]
            data = (X_train, X_test, y_train, y_test)

            cv_results.append(self.train(data=data))

        eval_scores = [result["eval_score"] for result in cv_results]

        if None not in eval_scores:
            logging.info("average score is %f, std is %f" %
                         (np.mean(eval_scores), np.std(eval_scores)))

        importances_lst = [result["importances"] for result in cv_results]
        self.feature_importances_ = compute_cv_feature_importances(
            *importances_lst)

        return cv_results
