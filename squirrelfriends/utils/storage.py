import numpy as np
import pandas as pd
import logging


def num2size(num, suffix="B"):
    """Convert byte num to formated size string.
    """

    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


def size_of_df(df):
    """Get the size string of a pandas.DataFrame.
    """

    return num2size(df.memory_usage(index=True).sum())


def reduce_memory_usage(df, verbose=True):
    """Reduce the df memory usage by convert numeric colume to best fitted type.
    """

    pd_numeric_types = ['int16', 'int32', 'int64',
                        'float16', 'float32', 'float64']
    original_memory = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        original_type = df[col].dtypes
        if original_type in pd_numeric_types:
            mmin = df[col].min()
            mmax = df[col].max()
            if str(original_type)[:3] == 'int':
                if mmin > np.iinfo(np.int8).min and mmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif mmin > np.iinfo(np.int16).min and mmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif mmin > np.iinfo(np.int32).min and mmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if mmin > np.finfo(np.float16).min and mmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif mmin > np.finfo(np.float32).min and mmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    result_memory = df.memory_usage().sum() / 1024**2
    if verbose:
        logging.info('Memory usage decreased from {:5.2f} Mb to {:5.2f} Mb ({:.1f}% reduction)'.format(
            original_memory, result_memory, 100 * (original_memory - result_memory) / original_memory))
    return df
