import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logging.basicConfig(format="%(asctime)s %(message)s",
                    datefmt="%m/%d/%Y %I:%M:%S %p",
                    level=logging.INFO)
sns.set(color_codes=True)


def dist_plot(data, cols, group_by=None, lower=0.01, upper=0.99):
    """Plot distribution of numeric variables group by a catgorical variable.

    Args:
        data (DataFrame): data with columns `cols` and `group_by`.
        cols (list of str or str): column name of numeric variables.
        group_by (str or None): column name to group samples.
        lower (float): lower bound quantile of variable to visualize.
        upper (float): upper bound quantile of variable to visualize.

    Retunrns:
        grid (matplotlib Axes): axes object with the distplot.
    """

    logging.info("number of samples: {}\n".format(data.shape[0]))

    # Standardize groups and data_dict in different scenarios of group_by
    if group_by is None:
        groups = ["entire dataset"]
        data_dict = {g: data for g in groups}
    else:
        groups = sorted(data[group_by].unique(), reverse=True)
        data_dict = {g: data[data[group_by] == g] for g in groups}

    for g in groups:
        logging.info("number of samples in {}: {}\n".format(
            g, data_dict[g].shape[0]))

    n_groups = len(groups)

    if type(cols) is str:
        cols = [cols]

    grid, ax = plt.subplots(len(cols), n_groups, sharex="row",
                            figsize=(8 * n_groups, 5 * len(cols)))

    for i, g in enumerate(groups):
        g_data = data_dict[g]
        for j, _col in enumerate(cols):
            # Prepare data to plot
            plot_data = g_data[_col]
            if upper < 1.0 or lower > 0.0:
                upper_value = plot_data.quantile(upper)
                lower_value = plot_data.quantile(lower)
                plot_data = plot_data[(plot_data <= upper_value) &
                                      (plot_data >= lower_value)]
            # Get correct ax per subplot
            if type(ax) is not np.ndarray:
                plot_ax = ax
            elif len(ax.shape) == 2:
                plot_ax = ax[j][i]
            elif n_groups == 1:
                plot_ax = ax[j]
            else:
                plot_ax = ax[i]

            sns.distplot(plot_data, ax=plot_ax)

    return grid


def count_plot(data, cols, group_by=None, top_n=10):
    """Plot distribution of catgorical / order variables in groups,
    especally for the imbalance data.

    Args:
        data (DataFrame): data with columns `cols` and `group_by`.
        cols (list of str or str): column name of numeric variables.
        group_by (str or None): column name to group samples.
        top_n (int): top count for each variable,
            ONLY work for catgorical variables.

    Retunrns:
        grid (matplotlib Axes): axes object with the barplot or countplot.
    """

    logging.info("number of samples: {}\n".format(data.shape[0]))

    # Standardize groups and data_dict in different scenarios of group_by
    if group_by is None:
        groups = ["entire dataset"]
        data_dict = {g: data for g in groups}
    else:
        groups = sorted(data[group_by].unique(), reverse=True)
        data_dict = {g: data[data[group_by] == g] for g in groups}

    for g in groups:
        logging.info("number of samples in {}: {}\n".format(
            g, data_dict[g].shape[0]))

    n_groups = len(groups)

    if type(cols) is str:
        cols = [cols]

    grid, ax = plt.subplots(
        len(cols), n_groups, sharex="row",
        figsize=(8 * n_groups, 5 * len(cols) * top_n // 10)
    )

    for i, g in enumerate(groups):
        g_data = data_dict[g]
        for j, _col in enumerate(cols):
            # Prepare data to plot
            plot_data = g_data[_col]
            # Get correct ax per subplot
            if type(ax) is not np.ndarray:
                plot_ax = ax
            elif len(ax.shape) == 2:
                plot_ax = ax[j][i]
            elif n_groups == 1:
                plot_ax = ax[j]
            else:
                plot_ax = ax[i]

            if plot_data.dtype != "object":
                sns.countplot(y=_col, data=g_data, ax=plot_ax)
            else:
                value_count = plot_data.value_counts()[:top_n]
                sns.barplot(y=value_count.index,
                            x=value_count.values, ax=plot_ax)
            plot_ax.set_ylabel(_col)

    return grid
