import pandas as pd
import numpy as np
from data_processing import metrics
import matplotlib.pyplot  as plt
from pathlib import Path


def runtime(dataframe, case, timers, columns, outdir):

    Path(outdir).mkdir(parents=True, exist_ok=True)

    for timer in timers:
        runtime_table = pd.pivot_table(dataframe, index=case, columns=columns, values=timer+'_mean')
        error_table = pd.pivot_table(dataframe, index=case, columns=columns, values=timer+'_stderror')

        fig, ax = plt.subplots(tight_layout=True)
        runtime_table.plot(kind='bar', yerr=error_table, ax=ax)
        ax.set_ylabel('Runtime (s)')
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        fig.savefig(outdir + '/runtime_' + timer + '.eps', format='eps', dpi=1000)


def speedup(dataframe, case, timers, columns, outdir):

    Path(outdir).mkdir(parents=True, exist_ok=True)
    xlabels = metrics.get_column_entries(dataframe, case)

    for timer in timers:
        speedup_table = pd.pivot_table(dataframe, index=case, columns=columns, values=timer+'_mean')
        error_table = pd.pivot_table(dataframe, index=case, columns=columns, values=timer+'_stderror')

        fig, ax = plt.subplots(tight_layout=True)
        speedup_table.plot(kind='line', yerr = error_table, marker='*', ax=ax)
        if type(xlabels[0]) is not int:
            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_xticklabels(xlabels, rotation = 90)
        ax.set_ylabel('Speed Up ' + r'$\frac{T_{1}}{T_{p}}$' + ' (Times)')
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        fig.savefig(outdir + '/speedup_' + timer + '.eps', format='eps', dpi=1000)


def normalized_runtime(dataframe, case, timers, columns, outdir):

    Path(outdir).mkdir(parents=True, exist_ok=True)
    xlabels = metrics.get_column_entries(dataframe, case)

    for timer in timers:
        ratio_table = pd.pivot_table(dataframe, index=case, columns=columns, values=timer+'_mean')
        error_table = pd.pivot_table(dataframe, index=case, columns=columns, values=timer+'_stderror')

        fig, ax = plt.subplots(tight_layout=True)
        ratio_table.plot(kind='line', yerr = error_table, marker='*', ax=ax)
        ax.set_xticks(np.arange(len(xlabels)))
        ax.set_xticklabels(xlabels, rotation = 90)
        ax.set_yticks(np.arange(0.80, 1.25, 0.05))
        ax.set_ylabel('Normalized Runtime (s)')
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0)
        fig.savefig(outdir + '/norm_runtime_' + timer + '.eps', format='eps', dpi=1000)