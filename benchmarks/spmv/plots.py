"""
 plots.py
 
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

header="Machine,Matrix,Reader,Set_Vecs,SpMv_COO,SpMv_CSR,SpMv_DIA,SpMv_DYN_COO,SpMv_DYN_CSR,SpMv_DYN_DIA"

other_timers = ['Reader', 'Set_Vecs']
concrete_timers = ['SpMv_COO', 'SpMv_CSR', 'SpMv_DIA']
dynamic_timers = ['SpMv_DYN_COO', 'SpMv_DYN_CSR', 'SpMv_DYN_DIA']
filename = '/work/e609/e609/cstyl/morpheus/benchmarks/spmv/results/clSpMV_archer_spmv_gcc_10.1.0_test_1.csv'

df = pd.read_csv(filename).groupby('Matrix').mean()

other_df = df[other_timers]
concrete_df = df[concrete_timers]
concrete_df.columns = ['COO', 'CSR', 'DIA']
dynamic_df = df[dynamic_timers]
dynamic_df.columns = ['COO', 'CSR', 'DIA']

# for timer in timers:
#     print(df[timer])

print(concrete_df)
print(dynamic_df)
print(dynamic_df/concrete_df)
