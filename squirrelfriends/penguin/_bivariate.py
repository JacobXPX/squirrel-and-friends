import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import check_consistent_length


def joint_plot(x, y, data=None, lower=0.0, upper=1.0, kde=True):
    """Draw a plot of two variables with bivariate and univariate graphs.
    Special capcity is cutting lower and upper bounds on samples.

    Args:
        x (str or array): data or names of variables in `data`.
        y (str or array): data or names of variables in `data`.
        data (DataFrame): optional
            DataFrame when `x` and `y` are variable names.
        lower (float): lower bound quantile of variable to visualize.
        upper (float): upper bound quantile of variable to visualize.
        kde (boolean): if use kde density.

    Returns:
        grid (:obj:`JointGrid`): The Grid class for drawing this plot.
    """

    warnings.warn("""This plot will take some time to generate,
                  especially when the data is large and kde is True """,
                  RuntimeWarning)

    if data is not None:
        x = data.get(x, x)
        y = data.get(y, y)

    check_consistent_length(x, y)

    xlabel = x.name if hasattr(x, "name") else "x"
    ylabel = y.name if hasattr(y, "name") else "y"

    if upper < 1.0 or lower > 0.0:
        uppers = {x: x.quantile(upper), y: y.quantile(upper)}
        lowers = {x: x.quantile(lower), y: y.quantile(lower)}

    plot_data = pd.DataFrame({xlabel: x, ylabel: y})

    def _filter(r): return all(r[c] <= uppers[c] for c in [x, y]) & \
        all(r[c] >= lowers[c] for c in [x, y])

    plot_data = plot_data[plot_data.apply(lambda r:_filter(r), axis=1)]

    if kde:
        grid = sns.jointplot(xlabel, ylabel, data=plot_data,
                             kind="kde", space=0, color="b")
    else:
        grid = sns.jointplot(xlabel, ylabel, data=plot_data,
                             color="b")
    return grid


def corrmatrix(data, cols, absolute=True, plot=True,
               method="pearson", min_periods=200):
    """Compute or plot the correlation matrix, excluding missing values.

    Args:
        data (DataFrame): data with columns `cols`.
        cols (list of str): columns with which to compute the correlation.
        absolute (boolean): if use absolute correlation value.
        plot (boolean): if plot corr matrix or return corr matrix.
        method (str): {"pearson", "kendall", "spearman"} or callable,
            Method used to compute correlation:
            - pearson : Standard correlation coefficient
            - kendall : Kendall Tau correlation coefficient
            - spearman : Spearman rank correlation
            - callable: Callable with input two 1d ndarrays
                and returning a float.
        min_periods (int): optional, minimum number of observations
            needed to have a valid result.
    Returns:
        grid (matplotlib Axes): axes object with the heatmap.
        corrmat (DataFrame): correlation matrix.
    """

    corrmat = data[cols].corr(method=method, min_periods=min_periods)

    if absolute:
        corrmat = np.abs(corrmat)

    if not plot:
        return corrmat

    plot_corrmat = corrmat * 100

    grid, ax = plt.subplots(figsize=(11, 11))
    sns.heatmap(plot_corrmat, cbar=False, annot=True, square=True, fmt=".1f",
                annot_kws={"size": 150 // plot_corrmat.shape[0]})
    ax.tick_params(axis="x", labelsize=150 // plot_corrmat.shape[0])
    ax.tick_params(axis="y", labelsize=120 // plot_corrmat.shape[0])

    return grid


def compute_kl_by_prob(p, q, eps=1e-5):
    """Compute Kullback-Leibler divergence D(P || Q).

    Args:
        p (array): density probability (pdf).
        q (array): density probability (pdf).
        eps (float): epison, avoid log(0).

    Returns:
        kl (float): kl divergence.
    """

    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)
    kl = np.sum(np.where(q != 0, p * np.log((p+eps) / q), 0))
    return kl


def compute_kl(p_array, q_array, lower=0.0, upper=1.0, bins=50, eps=1e-5):
    """Compute Kullback-Leibler divergence D(P || Q) of continuous variables.

    Args:
        p_array (ndarray): sample data set p.
        q_array (ndarray): sample data set q.
        lower (float): lower bound quantile of variable to calculate.
        upper (float): upper bound quantile of variable to calculate.
        bins (int): the number of bins for continuous varible.
        eps (float): epison, avoid log(0).

    Returns:
        kl (float): kl divergence.
    """

    upper_p, lower_p = p_array.quantile(upper), p_array.quantile(lower)
    upper_q, lower_q = q_array.quantile(upper), q_array.quantile(lower)

    p_cut = p_array[(p_array <= upper_p) & (p_array >= lower_p)]
    q_cut = q_array[(q_array <= upper_q) & (q_array >= lower_q)]

    mmax, mmin = max(p_cut.max(), q_cut.max()), min(p_cut.min(), q_cut.min())

    bin_range = np.linspace(start=mmin, stop=mmax, num=bins)

    p_n_samples, _ = np.histogram(p_cut, bins=bin_range, density=False)
    q_n_samples, _ = np.histogram(q_cut, bins=bin_range, density=False)
    p_pdf = p_n_samples / p_n_samples.sum()
    q_pdf = q_n_samples / q_n_samples.sum()

    kl = compute_kl_by_prob(q_pdf, p_pdf, eps)
    return kl
