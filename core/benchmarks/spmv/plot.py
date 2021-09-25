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
import matplotlib.pyplot as plt


def gpu_comparison(
    openmp_mu,
    openmp_sem,
    cuda_mu,
    cuda_sem,
    matrices,
    legend=[
        "COO_Custom",
        "CSR_Custom",
        "DIA_Custom",
    ],
    kernel="cuda",
):

    omp_mu = openmp_mu
    omp_sem = openmp_sem
    cu_mu = cuda_mu
    cu_sem = cuda_sem

    if kernel.lower() == "kokkos":
        cu_mu[matrices == "dc1", 1] = float("NaN")

    if openmp_mu.shape[1] == 4:
        omp_mu = np.delete(openmp_mu, 2, axis=1)
        omp_sem = np.delete(openmp_sem, 2, axis=1)

    if cuda_mu.shape[1] == 4:
        cu_mu = np.delete(cuda_mu, 2, axis=1)
        cu_sem = np.delete(cuda_sem, 2, axis=1)

    omp_mu = np.delete(omp_mu, 0, axis=1)
    omp_sem = np.delete(omp_sem, 0, axis=1)
    cu_mu = np.delete(cu_mu, 0, axis=1)
    cu_sem = np.delete(cu_sem, 0, axis=1)

    ratio = omp_mu / cu_mu

    # Error = sqrt((concrete_sem / dynamic_mu)^2 + (concrete_mu * dynamic_sem / dynamic_mu)^2)
    error = pow(
        pow(cu_sem / omp_mu, 2) + pow(cu_mu * omp_sem / pow(omp_mu, 2), 2),
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
    if kernel == "cuda":
        error_alg1 = pow(
            pow(cuda_sem[:, 2] / openmp_mu[:, 1], 2)
            + pow(cuda_mu[:, 2] * openmp_sem[:, 1] / pow(openmp_mu[:, 1], 2), 2),
            0.5,
        )
        plt.errorbar(
            matrices,
            openmp_mu[:, 1] / cuda_mu[:, 2],
            yerr=error_alg1,
            marker="*",
            linestyle="None",
        )
        legend.append("CSR_Custom_Alg1")
    ax.set_xticks(np.arange(matrices.shape[0]))
    ax.set_xticklabels(matrices, rotation=90)
    ax.set_ylabel("SpeedUp (Times)")
    ax.set_xlabel("Matrix Name")
    ax.grid(True)
    ax.legend(legend[1:])  # ignore COO format until we fix the OMP version
    plt.show()


def kokkos_comparison(
    custom_mu, custom_sem, kokkos_mu, kokkos_sem, matrices, arch="serial"
):

    cu_mu = custom_mu
    cu_sem = custom_sem
    ko_mu = kokkos_mu
    ko_sem = kokkos_sem

    if custom_mu.shape[1] == 4:
        cu_mu = np.delete(custom_mu, 2, axis=1)
        cu_sem = np.delete(custom_sem, 2, axis=1)

    if kokkos_mu.shape[1] == 4:
        ko_mu = np.delete(kokkos_mu, 2, axis=1)
        ko_sem = np.delete(kokkos_sem, 2, axis=1)

    legend = [
        "COO_Kokkos",
        "CSR_Kokkos",
        "DIA_Kokkos",
    ]

    ratio = cu_mu / ko_mu

    # Error = sqrt((concrete_sem / dynamic_mu)^2 + (concrete_mu * dynamic_sem / dynamic_mu)^2)
    error = pow(
        pow(ko_sem / cu_mu, 2) + pow(ko_mu * cu_sem / pow(cu_mu, 2), 2),
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
    ax.set_xticks(np.arange(len(matrices)))
    ax.set_xticklabels(matrices, rotation=90)
    ax.set_ylabel("Ratio (Times)")
    ax.set_xlabel("Matrix Name")
    ax.grid(True)
    ax.legend(legend)
    plt.show()


def format_performance(concrete_mu, concrete_sem, matrices, arch="serial"):

    concr_mu = concrete_mu
    concr_sem = concrete_sem
    if arch.lower() == "serial":
        concr_mu[matrices == "whitaker3_dual", 3] = float("NaN")
        concr_mu[matrices == "raefsky2", 3] = float("NaN")
        concr_mu[matrices == "whitaker3_dual", 3] = float("NaN")
        concr_mu[matrices == "olafu", 3] = float("NaN")
        concr_mu[matrices == "bcsstk17", 3] = float("NaN")
        concr_mu[matrices == "FEM_3D_thermal1", 3] = float("NaN")
        concr_mu = np.delete(concr_mu, 2, axis=1)
        concr_sem = np.delete(concr_sem, 2, axis=1)
        legend = ["COO", "CSR_Alg0", "DIA"]
    elif arch.lower() == "openmp":
        concr_mu = np.delete(concr_mu, 2, axis=1)
        concr_sem = np.delete(concr_sem, 2, axis=1)
        legend = ["COO", "CSR_Alg0", "DIA"]
    elif arch.lower() == "cuda":
        concr_mu[matrices == "dc1", 2] = float("NaN")
        concr_mu[matrices == "whitaker3_dual", 3] = float("NaN")
        legend = ["COO", "CSR_Alg0", "CSR_Alg1", "DIA"]

    ref_mu = concr_mu[:, 0]
    ratio = ref_mu[:, None] / concr_mu
    error = concr_sem

    fig, ax = plt.subplots(tight_layout=True)
    for i in range(ratio.shape[1]):
        plt.errorbar(
            matrices,
            ratio[:, i],
            yerr=error[:, i],
            marker="*",
            linestyle="None",
        )
    ax.set_xticks(np.arange(len(matrices)))
    ax.set_xticklabels(matrices, rotation=90)
    ax.set_ylabel("Ratio (Times)")
    ax.set_xlabel("Matrix Name")
    ax.grid(True)
    ax.legend(legend)
    plt.show()


def dynamic_overheads(
    concrete_mu, concrete_sem, dynamic_mu, dynamic_sem, matrices, arch="serial"
):

    if concrete_mu.shape != dynamic_mu.shape:
        concr_mu = np.delete(concrete_mu, 2, axis=1)
        concr_sem = np.delete(concrete_sem, 2, axis=1)
    else:
        concr_mu = concrete_mu
        concr_sem = concrete_sem

    legend = [
        "COO_Custom",
        "CSR_Custom",
        "DIA_Custom",
    ]

    ratio = concr_mu / dynamic_mu

    # Error = sqrt((concrete_sem / dynamic_mu)^2 + (concrete_mu * dynamic_sem / dynamic_mu)^2)
    error = pow(
        pow(dynamic_sem / concr_mu, 2)
        + pow(dynamic_mu * concr_sem / pow(concr_mu, 2), 2),
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
    ax.set_xticks(np.arange(len(matrices)))
    ax.set_xticklabels(matrices, rotation=90)
    ax.set_ylabel("Ratio (Times)")
    ax.set_xlabel("Matrix Name")
    ax.grid(True)
    ax.legend(legend)
    plt.show()


def split_to_numpy(dataframe):
    df = dataframe.reset_index()
    custom = df[
        [
            "SpMv_COO_Custom",
            "SpMv_CSR_Custom_Alg0",
            "SpMv_CSR_Custom_Alg1",
            "SpMv_DIA_Custom",
        ]
    ].to_numpy()

    dyncustom = df[
        [
            "SpMv_DYN_COO_Custom",
            "SpMv_DYN_CSR_Custom",
            "SpMv_DYN_DIA_Custom",
        ]
    ].to_numpy()

    kokkos = df[
        [
            "SpMv_COO_Kokkos",
            "SpMv_CSR_Kokkos",
            "SpMv_CSR_Kokkos",
            "SpMv_DIA_Kokkos",
        ]
    ].to_numpy()

    dynkokkos = df[
        [
            "SpMv_DYN_COO_Kokkos",
            "SpMv_DYN_CSR_Kokkos",
            "SpMv_DYN_DIA_Kokkos",
        ]
    ].to_numpy()

    deep = df[["COO_Deep", "CSR_Deep", "DIA_Deep"]]

    return custom, dyncustom, kokkos, dynkokkos, deep


def reshape_to_threads(mu, sem, ref_len):

    mu = mu.reshape(int(mu.shape[0] / ref_len), ref_len, mu.shape[1])
    sem = sem.reshape(int(sem.shape[0] / ref_len), ref_len, sem.shape[1])

    return mu, sem


results_path = "/Volumes/PhD/Code/Projects/morpheus/core/benchmarks/results/"

Serial_df = pd.read_csv(results_path + "spmv-Serial/large_set_cirrus_spmv_large.csv")
OpenMP_df = pd.read_csv(results_path + "spmv-OpenMP/large_set_cirrus_spmv_large.csv")
Cuda_df = pd.read_csv(results_path + "spmv-Cuda/large_set_cirrus_spmv_large.csv")

ser_mu_df = (
    Serial_df.drop(["Machine", "Target", "Threads", "Reps"], axis=1)
    .groupby(["Matrix"])
    .agg(np.mean)
)
ser_sem_df = (
    Serial_df.drop(["Machine", "Target", "Threads", "Reps"], axis=1)
    .groupby(["Matrix"])
    .agg(sem)
)

omp_mu_df = (
    OpenMP_df.drop(["Machine", "Target", "Reps"], axis=1)
    .groupby(["Threads", "Matrix"])
    .agg(np.mean)
)
omp_sem_df = (
    OpenMP_df.drop(["Machine", "Target", "Reps"], axis=1)
    .groupby(["Threads", "Matrix"])
    .agg(sem)
)
cu_mu_df = (
    Cuda_df.drop(["Machine", "Target", "Threads", "Reps"], axis=1)
    .groupby(["Matrix"])
    .agg(np.mean)
)
cu_sem_df = (
    Cuda_df.drop(["Machine", "Target", "Threads", "Reps"], axis=1)
    .groupby(["Matrix"])
    .agg(sem)
)

matrices = ser_mu_df.reset_index()["Matrix"].to_numpy()

ser_cmu, ser_dcmu, ser_kmu, ser_dkmu, ser_deep_mu = split_to_numpy(ser_mu_df)
ser_csem, ser_dcsem, ser_ksem, ser_dksem, ser_deep_sem = split_to_numpy(ser_sem_df)
omp_cmu, omp_dcmu, omp_kmu, omp_dkmu, omp_deep_mu = split_to_numpy(omp_mu_df)
omp_csem, omp_dcsem, omp_ksem, omp_dksem, omp_deep_sem = split_to_numpy(omp_sem_df)
cu_cmu, cu_dcmu, cu_kmu, cu_dkmu, cu_deep_mu = split_to_numpy(cu_mu_df)
cu_csem, cu_dcsem, cu_ksem, cu_dksem, cu_deep_sem = split_to_numpy(cu_sem_df)

omp_cmu, omp_csem = reshape_to_threads(omp_cmu, omp_csem, len(matrices))
omp_dcmu, omp_dcsem = reshape_to_threads(omp_dcmu, omp_dcsem, len(matrices))
omp_kmu, omp_ksem = reshape_to_threads(omp_kmu, omp_ksem, len(matrices))
omp_dkmu, omp_dksem = reshape_to_threads(omp_dkmu, omp_dksem, len(matrices))

sz = omp_cmu.shape[0] - 1

print("Format Performance plots: T_ref / T_format")

# Format Selection: Plot normalized time for each format wrt COO (Serial)
format_performance(ser_cmu, ser_csem, matrices, arch="serial")
format_performance(omp_cmu[sz], omp_csem[sz], matrices, arch="openmp")
format_performance(cu_cmu, cu_csem, matrices, arch="cuda")

print("Dynamic Overheads plots: T_concr / T_dynamic")

# Dynamic Overheads: Plot ratio of Dynamic/Concrete (Serial)
dynamic_overheads(ser_cmu, ser_csem, ser_dcmu, ser_dcsem, matrices, arch="serial")
dynamic_overheads(
    omp_cmu[sz], omp_csem[sz], omp_dcmu[sz], omp_dcsem[sz], matrices, arch="openmp"
)
dynamic_overheads(cu_cmu, cu_csem, cu_dcmu, cu_dcsem, matrices, arch="cuda")

print("Kokkos Comparison plots: T_custom / T_kokkos")

# Kokkos Overheads: Plot ratio of Kokkos/Custom SpMV (Serial, OpenMP, Cuda)
kokkos_comparison(ser_cmu, ser_csem, ser_kmu, ser_ksem, matrices, arch="serial")
kokkos_comparison(
    omp_cmu[sz], omp_csem[sz], omp_kmu[sz], omp_ksem[sz], matrices, arch="openmp"
)
kokkos_comparison(cu_cmu, cu_csem, cu_kmu, cu_ksem, matrices, arch="cuda")

print("GPU Comparison plots: T_omp / T_gpu")

# GPU Speed Up: GPU/OpenMP(Ncores) for Concrete
legend = [
    "COO_Custom",
    "CSR_Custom",
    "DIA_Custom",
]
gpu_comparison(omp_cmu[sz], omp_csem[sz], cu_cmu, cu_csem, matrices, legend=legend)

legend = [
    "COO_Kokkos",
    "CSR_Kokkos",
    "DIA_Kokkos",
]
gpu_comparison(
    omp_kmu[sz], omp_ksem[sz], cu_kmu, cu_ksem, matrices, legend=legend, kernel="kokkos"
)
