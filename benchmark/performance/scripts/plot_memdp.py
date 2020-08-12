import pandas as pd
from data_processing import metrics
from data_processing import cmd_line_parser as parser
from data_processing import plot

counters = ['Runtime', 'Power', 'Power Ram', 
            'CPI', 'Flops', 'Flops AVX', 
            'Read Bandwidth', 'Write Bandwidth', 
            'Read Volume', 'Write Volume', 
            'Bandwidth', 'Memory', 
            'Arithmetic Intensity']

main_counters = ['Flops', 'Bandwidth', 'Arithmetic Intensity', 'CPI']

group = ['Matrix', 'Format']

args = parser.get_args()

df = pd.read_csv(args.filename)

counters_df = metrics.get_runtime(df, group, main_counters)

plot.unit(counters_df, 'Matrix', ['Flops'], 'Format', 'Flops/s', args.outdir, 'flops.eps')
plot.unit(counters_df, 'Matrix', ['Bandwidth'], 'Format', 'Bandwidth (MBytes/s)', args.outdir, 'bandwidth.eps')
plot.unit(counters_df, 'Matrix', ['Arithmetic Intensity'], 'Format', 'Arithmetic Intensity', args.outdir, 'arithmetic_int.eps')
plot.unit(counters_df, 'Matrix', ['CPI'], 'Format', 'CPI', args.outdir, 'cpi.eps')