import pandas as pd
import numpy as np
filename = "/Users/cstyl/Desktop/Projects/morpheus/benchmark/variant_bench/results/processed_data_local.csv"
df = pd.read_csv(filename)

Matrices = df.drop_duplicates(subset="Matrix", keep='first', inplace=False)["Matrix"]
Versions = df.drop_duplicates(subset="Version", keep='first', inplace=False)["Version"]

reduced_data=[]

for m in Matrices:
    for v in Versions:
        reduced_df  = df.loc[(df['Matrix'] == m) & (df['Version'] == v)][["Total", "Reader", "Writer", "SpMv"]]
        reduced_data.append([m,v] + reduced_df.mean().tolist() + reduced_df.std().tolist())

timing_df  = pd.DataFrame(reduced_data,  columns=['Matrix', 'Version', 'TotalMean', 'ReaderMean',
                                                  'WriterMean', 'SpMvMean', 'TotalStd',
                                                  'ReaderStd', 'WriterStd', 'SpMvStd'])

print(timing_df)