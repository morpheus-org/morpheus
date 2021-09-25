"""
 plot.py
 
 EPCC, The University of Edinburgh
 
 (c) 2021 The University of Edinburgh
 
 Contributing Authors:
 Christodoulos Stylianou (c.stylianou@ed.ac.uk)
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
 	http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import pandas as pd
import numpy as np
from scipy.stats import sem
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def get_args():
    """
    Returns a namedtuple with arguments extracted from the command line.
    :return: A namedtuple with arguments
    """

    parser = argparse.ArgumentParser(
        description="Data processing, analysis and plotting for morpheus benchmarks"
    )

    parser.add_argument(
        "--resdir", type=str, required=True, help="Absolute path to results directory."
    )

    parser.add_argument(
        "--filename",
        type=str,
        default="large_set_cirrus_spmv_large.csv",
        help="CSV filename to read the data from.",
    )

    parser.add_argument(
        "--outdir",
        "-o",
        type=str,
        required=True,
        help="Absolute path of the output directory to write the plot files in.",
    )

    args = parser.parse_args()

    return args


def copy_comparison(
    deep_mu,
    deep_sem,
    elem_mu,
    elem_sem,
    matrices,
    legend=[
        "COO",
        "CSR",
        "DIA",
    ],
    outdir=None,
):

    dmu = deep_mu
    dsem = deep_sem
    emu = elem_mu
    esem = elem_sem

    ratio = emu / dmu
    error = pow(
        pow(esem / dmu, 2) + pow(emu * dsem / pow(dmu, 2), 2),
        0.5,
    )

    fig, ax = plt.subplots(tight_layout=True)
    for i in range(ratio.shape[1]):
        plt.errorbar(
            matrices,
            ratio[:, i],
            yerr=error[:, i],
            marker="*",
            linestyle="None",
        )

    ax.set_xticks(np.arange(matrices.shape[0]))
    ax.set_xticklabels(matrices, rotation=90)
    ax.set_ylabel("SpeedUp (Times)")
    ax.set_xlabel("Matrix Name")
    ax.grid(True)
    ax.legend(legend)  # ignore COO format until we fix the OMP version

    if outdir:
        fig.savefig(outdir + "copy_comparison.eps", format="eps", dpi=1000)
    else:
        plt.show()


def split_to_numpy(dataframe):
    df = dataframe.reset_index()
    deep = df[["COO_Deep", "CSR_Deep", "DIA_Deep"]].to_numpy()

    elem = df[["COO_Elem", "CSR_Elem", "DIA_Elem"]].to_numpy()

    return deep, elem


args = get_args()

outdir = args.outdir
Path(outdir).mkdir(parents=True, exist_ok=True)

# Dataframes
copy_df = pd.read_csv(args.resdir + "/" + args.filename)

copy_mu_df = (
    copy_df.drop(["Machine", "Target", "Threads"], axis=1)
    .groupby(["Matrix"])
    .agg(np.mean)
)
copy_sem_df = (
    copy_df.drop(["Machine", "Target", "Threads"], axis=1).groupby(["Matrix"]).agg(sem)
)
matrices = copy_mu_df.reset_index()["Matrix"].to_numpy()

deep_mu, elem_mu = split_to_numpy(copy_mu_df)
deep_sem, elem_sem = split_to_numpy(copy_sem_df)

print("Copy Comparison plots: T_elem / T_copy")

# Copy Speed Up: Elementwise Copy / Deep Copy
legend = [
    "COO_Custom",
    "CSR_Custom",
    "DIA_Custom",
]
copy_comparison(
    deep_mu,
    deep_sem,
    elem_mu,
    elem_sem,
    matrices,
    legend=legend,
    outdir=outdir + "/serial_",
)
