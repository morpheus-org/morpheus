import pandas as pd
from data_processing import metrics
from data_processing import cmd_line_parser as parser
from data_processing import plot

counters = ['Runtime', 'Power', 'Power Ram', 
            'Flops', 'Flops AVX', 
            'Read Bandwidth', 'Write Bandwidth', 
            'Read Volume', 'Write Volume', 
            'Bandwidth', 'Memory', 
            'Arithmetic Intensity']

main_counters = ['Flops', 'Bandwidth', 'Arithmetic Intensity']

group = ['Matrix', 'Format']

args = parser.get_args()

df = pd.read_csv(args.filename)

runtime_df = metrics.get_runtime(df, group, main_counters)
plot.runtime(runtime_df, 'Matrix', main_counters, 'Format', args.outdir)