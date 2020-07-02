import pandas as pd
import math


def get_column_entries(dataframe, column):

    # if not a grouped dataframe
    if dataframe.index.names[0] is None:
        return dataframe.drop_duplicates(subset=column, inplace=False)[column].tolist()

    for idx, name in enumerate(dataframe.index.names):
        if name == column:
            level = idx

    return dataframe.index.get_level_values(level).drop_duplicates().tolist()


def match_size(df1, df2, over_column):

    singular_entries = get_column_entries(df1, over_column)
    df2 = pd.concat([df2] * len(singular_entries)).sort_index()

    return df2


def get_dataframes(dataframe, column, ref, keep_ref=False):

    ref_df = dataframe.loc[(dataframe[column] == ref)]

    if keep_ref is True:
        experimental_df = dataframe
    else:
        experimental_df = dataframe.loc[(dataframe[column] != ref)]

    return [ref_df, experimental_df]


def get_mean(dataframe, group, timers):

    mu = dataframe.groupby(group)[timers].mean()

    return mu

def get_std_error(dataframe, group, timers):

    std = dataframe.groupby(group)[timers].std()
    # experiment repetitons = no. of rows before std / no. of rows after std
    n = dataframe.shape[0] / std.shape[0]

    std_error = std / math.sqrt(n)

    return std_error


def get_speedup(dataframe, group, timers, column, ref):

    ref_df, exp_df = get_dataframes(dataframe, 'Version', 'cusp', True)

    ref_mu = get_mean(ref_df, group, timers)
    exp_mu = get_mean(exp_df, group, timers)

    ref_std_error = get_std_error(ref_df, group, timers)
    exp_std_error = get_std_error(exp_df, group, timers)

    # Make sure size of reference dataframe matches the experimental
    ref_mu = match_size(exp_mu, ref_mu, group[len(group)-1])
    ref_std_error = match_size(exp_std_error, ref_std_error, group[len(group)-1])

    # Get the values of reference dataframes so that
    # the resulting dataframes will have the same indexing
    # with the experimental
    mu1 = ref_mu.values
    std1 = ref_std_error.values
    muP = exp_mu
    stdP = exp_std_error

    # Ratio = T1 / Tp
    speedup_df = mu1 / muP
    # Error = sqrt((std_error_ref / mean_experimental)^2 + (mean_ref * std_error_experimental / mean_experimental)^2)
    error_df = pow(pow(std1 / muP, 2) + pow(mu1 * stdP / pow(muP, 2), 2), 0.5)
    # Change timer columns before concatenating
    speedup_df.columns = [str(col) + '_mean' for col in speedup_df.columns]
    error_df.columns = [str(col) + '_stderror' for col in error_df.columns]

    return pd.concat([speedup_df, error_df], axis=1)


def get_normalized_runtime(dataframe, group, timers, column, ref):

    ref, exp = get_dataframes(dataframe, 'Version', 'cusp', False)

    ref_mu = get_mean(ref, group, timers)
    exp_mu = get_mean(exp, group, timers)

    ref_std_error = get_std_error(ref, group, timers)
    exp_std_error = get_std_error(exp, group, timers)

    # Make sure size of reference dataframe matches the experimental
    ref_mu = match_size(exp_mu, ref_mu, group[len(group)-1])
    ref_std_error = match_size(exp_std_error, ref_std_error, group[len(group)-1])

    # Get the values of reference dataframes so that
    # the resulting dataframes will have the same indexing
    # with the experimental
    mu_r = ref_mu.values
    std_r = ref_std_error.values
    mu_e = exp_mu
    std_e = exp_std_error

    # Ratio = Texp / Tref
    ratio_df = mu_e / mu_r
    # Error = sqrt((std_error_ref / mean_experimental)^2 + (mean_ref * std_error_experimental / mean_experimental)^2)
    error_df = pow(pow(std_e / mu_r, 2) + pow(mu_e * std_r / pow(mu_r, 2), 2), 0.5)
    # Change timer columns before concatenating
    ratio_df.columns = [str(col) + '_mean' for col in ratio_df.columns]
    error_df.columns = [str(col) + '_stderror' for col in error_df.columns]

    return pd.concat([ratio_df, error_df], axis=1)


def get_runtime(dataframe, group, timers):

    mu = get_mean(dataframe, group, timers)
    std_error = get_std_error(dataframe, group, timers)

    # Change timer columns before concatenating
    mu.columns = [str(col) + '_mean' for col in mu.columns]
    std_error.columns = [str(col) + '_stderror' for col in std_error.columns]

    return pd.concat([mu, std_error], axis=1)