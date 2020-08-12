import pandas as pd
from data_processing import metrics
from data_processing import cmd_line_parser as parser
from data_processing import plot

counters = ['Runtime', 'Execution Stalls', 'L1 stalls', 
            'L2 Stalls', 'Memory Stalls', 'Execution Stall Rate', 
            'L1 stall Rate', 'L2 Stall Rate', 'Memory Stall Rate']

main_counters = ['Execution Stall Rate', 'L1 stall Rate', 'L2 Stall Rate', 'Memory Stall Rate']

group = ['Matrix', 'Format']

args = parser.get_args()

df = pd.read_csv(args.filename)

counters_df = metrics.get_runtime(df, group, main_counters)

print(counters_df)

plot.unit(counters_df, 'Matrix', ['L1 stall Rate'], 'Format', 'L1 stall Rate (%)', args.outdir, 'l1.eps')
plot.unit(counters_df, 'Matrix', ['L2 Stall Rate'], 'Format', 'L2 stall Rate (%)', args.outdir, 'l2.eps')
plot.unit(counters_df, 'Matrix', ['Memory Stall Rate'], 'Format', 'Memory stall Rate (%)', args.outdir, 'mem_stall.eps')