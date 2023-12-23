import numpy as np
import pandas as pd
import random
import os
import matplotlib.pyplot as plt
from arch.unitroot import ADF
from statsmodels.sandbox.stats.runs import runstest_1samp
import statsmodels.tsa.stattools as ts
from itertools import islice, product


def read_file(data_dir, file_name):
    df = pd.read_csv(os.path.join(data_dir, file_name), sep='\t', names=['RR Interval', 'Index'])
    all_indexes = np.arange(df['Index'][0], df['Index'][len(df['Index']) - 1] + 1)
    missing_indexes = np.setdiff1d(all_indexes, df['Index'])
    missing_indexes_df = pd.DataFrame({'Index': missing_indexes})
    df = pd.concat([df, missing_indexes_df], ignore_index=True)
    df = df.sort_values(by='Index')
    # df['RR Interval'].interpolate(method="linear", inplace=True)

    return df


def flatten(x):
    if isinstance(x, list):
        return [a for i in x for a in flatten(i)]
    else:
        return [x]


def list_to_one_value(x):
    if isinstance(x[0], float) or isinstance(x[0], int):
        return round(np.mean(x), 2)
    elif x[0] == 'nielosowy' or x[0] == 'losowy':
        return round(x.count('losowy') / len(x), 2)
    elif x[0] == 'niestacjonarny' or x[0] == 'stacjonarny':
        return round(x.count('niestacjonarny') / len(x), 2)


def merge_files(file_paths, columns):
    merged_df = pd.DataFrame()

    for file_path in file_paths:
        df = pd.read_csv(file_path, sep='\t', usecols=columns)
        merged_df = pd.concat([merged_df, df]).groupby(level=0).agg(lambda x: x.tolist())

    merged_df = merged_df.applymap(lambda x: flatten(x))
    merged_df = merged_df.applymap(lambda x: list_to_one_value(x))

    return merged_df


def interpret_WW(pvalue):
    if pvalue < 0.05:
        return "nielosowy"
    else:
        return "losowy"


def interpret_ADF(pvalue):
    if pvalue < 0.05:
        return "stacjonarny"
    else:
        return "niestacjonarny"


def chunk_list(lst, n):
    it = iter(lst)
    return iter(lambda: tuple(islice(it, n)), ())


def statistics(data):
    data = data.dropna()
    WW_pvalue = runstest_1samp(data, cutoff="median")[1]
    ADF_pvalue = ts.adfuller(data)[1]
    stats = [
        round(np.mean(data), 2),
        round(np.std(data), 2),
        round(np.min(data), 2),
        round(np.max(data), 2),
        round(WW_pvalue, 2),
        interpret_WW(WW_pvalue),
        round(ADF_pvalue, 2),
        interpret_ADF(ADF_pvalue)
    ]
    return stats


def generate_random_color():
    color = "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    return color


def rr_intervals_events(RR_intervals):
    delta_RR = RR_intervals.diff().dropna()
    series_symbolization = list(np.zeros(len(delta_RR), dtype=object))

    for i, diff in enumerate(delta_RR):
        if 0 < diff < 40:
            series_symbolization[i] = "d"
        elif -40 < diff < 0:
            series_symbolization[i] = "a"
        elif 40 <= diff:
            series_symbolization[i] = "D"
        elif diff <= -40:
            series_symbolization[i] = "A"
        else:
            series_symbolization[i] = "z"

    pairs_in_series_symbolization = [series_symbolization[i] + series_symbolization[i + 1] for i in
                                     range(len(series_symbolization) - 1)]
    tripples_in_series_symbolization = [
        series_symbolization[i] + series_symbolization[i + 1] + series_symbolization[i + 2] for i in
        range(len(series_symbolization) - 2)]

    single_events = ["z", "a", "A", "d", "D"]
    double_events = ["".join(pair) for pair in product(single_events, repeat=2)]
    tripple_events = ["".join(pair) for pair in product(single_events, repeat=3)]

    single_events_occurrences = [series_symbolization.count(event) for event in single_events]
    double_events_occurrences = [pairs_in_series_symbolization.count(event) for event in double_events]
    tripple_events_occurrences = [tripples_in_series_symbolization.count(event) for event in tripple_events]

    return single_events, single_events_occurrences, double_events, double_events_occurrences, tripple_events, tripple_events_occurrences


def windows_statistics(RR_intervals, window_size):
    chunks = [batch for batch in list(chunk_list(RR_intervals.dropna(), window_size)) if len(batch) == window_size]

    WW_results = []
    ADF_results = []

    for chunk in chunks:
        chunk_ts = pd.Series(chunk)
        max_lags = int(np.sqrt(chunk_ts.shape[0]))
        ADF_pvalue = ADF(chunk_ts, trend="c", max_lags=max_lags).pvalue
        ADF_results.append(interpret_ADF(ADF_pvalue))
        WW_pvalue = runstest_1samp(chunk_ts.dropna(), cutoff="median")[1]
        WW_results.append(interpret_WW(WW_pvalue))

    means_of_windows = np.array([np.mean(chunk) for chunk in chunks])

    mean_RR = np.nanmean(means_of_windows)
    min_RR = np.nanmin(means_of_windows)
    max_RR = np.nanmax(means_of_windows)
    var_RR = np.nanstd(means_of_windows)
    random_sequences = WW_results.count("losowy") / len(WW_results)
    nonstationary_serieses = ADF_results.count("niestacjonarny") / len(ADF_results)
    stats = [
        window_size,
        round(mean_RR, 2),
        round(var_RR, 2),
        round(min_RR, 2),
        round(max_RR, 2),
        round(random_sequences, 2),
        round(nonstationary_serieses, 2)
    ]

    return stats


def plot(x, y, file_name, dir_name, title, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(30, 7)
    ax.plot(x, y)
    ax.set_title('{} - {}'.format(title, file_name))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    path = os.path.join(dir_name, file_name, "plots", '{} - {}.png'.format(title, file_name))
    plt.savefig(path)


def hist(x, file_name, dir_name, title, xlabel=None, ylabel=None):
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 7)
    ax.hist(x)
    ax.set_title('{} - {}'.format(title, file_name))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    path = os.path.join(dir_name, file_name, "plots", '{} - {}.png'.format(title, file_name))
    plt.savefig(path)


def bar_plot(x, y, file_name, dir_name, events, length=8, hight=6, n=5, d=False):
    sorted_data = sorted(zip(x, y), key=lambda x: x[1], reverse=True)
    top_x, top_y = zip(*sorted_data[:n])
    fig, ax = plt.subplots(figsize=(length, hight))
    ax.bar(top_x, top_y)
    ax.set_xlabel('Events')
    ax.set_ylabel('Numbers of Events')
    ax.set_title('Bar Plot of Top {} {} Events - {}'.format(n, events, file_name))
    if d is False:
        path = os.path.join(dir_name, file_name, "plots", '{} - {}.png'.format(events, file_name))
        plt.savefig(path)
    else:
        path = os.path.join(dir_name, 'Group Statistics', 'D_{}_{}.png'.format(file_name, events))
        plt.savefig(path)


def stacked_plot(data, file_name, dir_name):
    fig, axes = plt.subplots(nrows=20, ncols=1, figsize=(8, 20), sharex=True)
    for i, ax in enumerate(axes, start=1):
        ax.plot(data.diff(i).dropna())
    fig.suptitle('k-Differenced Times Series - {}'.format(file_name), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(dir_name, file_name, "plots", 'Diffed Time Series - {}.png'.format(file_name)))


def stacked_hist(data, file_name, dir_name):
    fig, axes = plt.subplots(nrows=20, ncols=1, figsize=(5, 20), sharex=True)
    for i, ax in enumerate(axes, start=1):
        ax.hist(data.diff(i).dropna())
    fig.suptitle('Hist of k-Differenced TS - {}'.format(file_name), fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(os.path.join(dir_name, file_name, "plots", 'Hist of Diffed RR - {}.png'.format(file_name)))
