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


def convert_comparison(
    concrete_mu,
    concrete_sem,
    dynamic_mu,
    dynamic_sem,
    inplace_mu,
    inplace_sem,
    matrices,
    legend=[
        "Dynamic_COO",
        "Dynamic_CSR",
        "Dynamic_DIA",
        "Inplace_COO",
        "Inplace_CSR",
        "Inplace_DIA",
    ],
    outdir=None,
):

    cmu = concrete_mu
    csem = concrete_sem
    dmu = dynamic_mu
    dsem = dynamic_sem
    inmu = inplace_mu
    insem = inplace_sem

    dratio = cmu / dmu
    derror = pow(
        pow(csem / dmu, 2) + pow(cmu * dsem / pow(dmu, 2), 2),
        0.5,
    )

    inratio = cmu / inmu
    inerror = pow(
        pow(dsem / inmu, 2) + pow(dmu * insem / pow(inmu, 2), 2),
        0.5,
    )

    labels = ["coo", "csr", "dia"]

    for i, label in enumerate(labels):

        fig, ax = plt.subplots(tight_layout=True)

        for j in range(len(labels)):

            idx = i * len(labels) + j

            plt.errorbar(
                matrices,
                dratio[:, idx],
                yerr=derror[:, idx],
                marker="*",
                linestyle="None",
            )
        for j in range(len(labels)):

            idx = i * len(labels) + j
            plt.errorbar(
                matrices,
                inratio[:, idx],
                yerr=inerror[:, idx],
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
            fig.savefig(
                outdir + label + "_convert_comparison.eps", format="eps", dpi=1000
            )
        else:
            plt.show()


def split_to_numpy(dataframe):
    df = dataframe.reset_index()

    concrete_labels = [
        "COO_COO",
        "COO_CSR",
        "COO_DIA",
        "CSR_COO",
        "CSR_CSR",
        "CSR_DIA",
        "DIA_COO",
        "DIA_CSR",
        "DIA_DIA",
    ]
    concrete = df[concrete_labels].replace({0.0: np.nan, 0: np.nan}).to_numpy()

    dynamic_labels = [
        "DYN_COO_COO",
        "DYN_COO_CSR",
        "DYN_COO_DIA",
        "DYN_CSR_COO",
        "DYN_CSR_CSR",
        "DYN_CSR_DIA",
        "DYN_DIA_COO",
        "DYN_DIA_CSR",
        "DYN_DIA_DIA",
    ]
    dynamic = df[dynamic_labels].replace({0.0: np.nan, 0: np.nan}).to_numpy()

    inplace_labels = [
        "IN_COO_COO",
        "IN_COO_CSR",
        "IN_COO_DIA",
        "IN_CSR_COO",
        "IN_CSR_CSR",
        "IN_CSR_DIA",
        "IN_DIA_COO",
        "IN_DIA_CSR",
        "IN_DIA_DIA",
    ]
    inplace = df[inplace_labels].replace({0.0: np.nan, 0: np.nan}).to_numpy()

    return concrete, dynamic, inplace


args = get_args()

outdir = args.outdir
Path(outdir).mkdir(parents=True, exist_ok=True)

# Dataframes
convert_df = pd.read_csv(args.resdir + "/" + args.filename)

convert_mu_df = (
    convert_df.drop(["Machine", "Target", "Threads"], axis=1)
    .groupby(["Matrix"])
    .agg(np.mean)
)

convert_sem_df = (
    convert_df.drop(["Machine", "Target", "Threads"], axis=1)
    .groupby(["Matrix"])
    .agg(sem)
)
matrices = convert_mu_df.reset_index()["Matrix"].to_numpy()

conmu, dynmu, inmu = split_to_numpy(convert_mu_df)
consem, dynsem, insem = split_to_numpy(convert_sem_df)

print("Convert Comparison plots:")

# Convert Speed Up: Concrete Convert / Dynamic Convert and Concrete Convert / Inplace Convert
legend = [
    "Dynamic_COO",
    "Dynamic_CSR",
    "Dynamic_DIA",
    "Inplace_COO",
    "Inplace_CSR",
    "Inplace_DIA",
]

convert_comparison(
    conmu,
    consem,
    dynmu,
    dynsem,
    inmu,
    insem,
    matrices,
    legend=legend,
    outdir=outdir + "/serial_",
)
