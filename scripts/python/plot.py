import pandas as pd
import metrics
import cmd_line_parser as parser
import matplotlib.pyplot as plt

timers = ['Total', 'Reader', 'Writer', 'SpMv']
group = ['Matrix', 'Version']

args = parser.get_args()

df = pd.read_csv(args.filename)

runtime_df, runtime_error_df = metrics.get_runtime(df, group, timers)

table_runtime = pd.pivot_table(runtime_df, index='Matrix', columns='Version', values='Total')
table_runtime.plot(kind='bar')

ref_df, experimental_df = metrics.get_dataframes(df, 'Version', 'cusp', False)
speedup_df, speed_up_error_df = metrics.get_speedup(ref_df, experimental_df, group, timers)
table_speedup = pd.pivot_table(speedup_df, index='Matrix', columns='Version', values='Total')
# table_speedup.plot(kind='line', linestyle='None', marker='*')
table_speedup.plot(kind='line', marker='*')
plt.show()

