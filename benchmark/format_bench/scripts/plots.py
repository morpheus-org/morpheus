import pandas as pd
from data_processing import metrics
from data_processing import cmd_line_parser as parser
from data_processing import plot

timers = ['Total', 'Reader', 'SpMv']
group = ['Matrix', 'Format']

args = parser.get_args()

df = pd.read_csv(args.filename)

runtime_df = metrics.get_runtime(df, group, timers)
ratio_df = metrics.get_normalized_runtime(df, group, timers, 'Format', 'csr')

plot.runtime(runtime_df, 'Matrix', timers, 'Format', args.outdir)
plot.normalized_runtime(ratio_df, 'Matrix', timers, 'Format', args.outdir)