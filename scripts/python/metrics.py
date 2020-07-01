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


def get_dataframes(dataframe, column, ref, keep_ref = False):
    ref_df = dataframe.loc[(dataframe[column] == ref)]

    if keep_ref is True:
        experimental_df = dataframe
    else:
        experimental_df = dataframe.loc[(dataframe[column] != ref)]

    return [ref_df, experimental_df]


def get_mean(dataframe, group, timers):

    return dataframe.groupby(group)[timers].mean()


def get_std(dataframe, group, timers):

    return dataframe.groupby(group)[timers].std()


def get_speedup(reference_df, experimental_df, group, timers):

    ref_mu = get_mean(reference_df, group, timers)
    exp_mu = get_mean(experimental_df, group, timers)
    ref_std = get_std(reference_df, group, timers)
    exp_std = get_std(experimental_df, group, timers)

    # Make sure size of reference dataframe matches the experimental
    ref_mu = match_size(exp_mu, ref_mu, group[len(group)-1])
    ref_std = match_size(exp_std, ref_std, group[len(group)-1])

    # Get the values of reference dataframes so that
    # the resulting dataframes will have the same indexing
    # with the experimental
    mu1 = ref_mu.values
    std1 = ref_std.values
    muP = exp_mu
    stdP = exp_std

    # experiment repetitons = no. of rows before average / no. of rows after average
    reps = experimental_df.shape[0] / ref_mu.shape[0]

    # Ratio = T1 / Tp
    speedup_df = mu1 / muP
    # Error = sqrt((std_ref / mean_experimental)^2 + (mean_ref * std_experimental / mean_experimental)^2)
    error_df = pow(pow((std1 / math.sqrt(reps)) / muP, 2)
                   + pow(mu1 * (stdP / math.sqrt(reps)) / pow(muP, 2), 2)
                   , 0.5)

    return speedup_df, error_df


def get_runtime(dataframe, group, timers):

    mu = get_mean(dataframe, group, timers)
    std = get_std(dataframe, group, timers)

    return mu, std