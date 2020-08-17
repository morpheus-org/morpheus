import pandas as pd
from data_processing import metrics
from data_processing import cmd_line_parser as parser
from data_processing import plot

timers = ['Total', 'Reader', 'SpMv']
group = ['Matrix', 'Version']

args = parser.get_args()

df = pd.read_csv(args.filename)

variant_df = df.loc[df['Version'].isin(['static','dynamic_01','dynamic_06','dynamic_12','dynamic_20'])]
boost_df = df.loc[df['Version'].isin(['static','dynamic_01_boost','dynamic_06_boost','dynamic_12_boost','dynamic_20_boost'])]
variant_O2_df = df.loc[df['Version'].isin(['static_O2','dynamic_01_O2','dynamic_06_O2','dynamic_12_O2','dynamic_20_O2'])]

boost_df['Version'].replace({'dynamic_01_boost' : 'dynamic_01', 'dynamic_06_boost' : 'dynamic_06',
                                                   'dynamic_12_boost' : 'dynamic_12', 'dynamic_20_boost' : 'dynamic_20'}, inplace=True)

variant_O2_df['Version'].replace({'static_O2' : 'static', 'dynamic_01_O2' : 'dynamic_01', 
                                                             'dynamic_06_O2' : 'dynamic_06', 'dynamic_12_O2' : 'dynamic_12', 
                                                             'dynamic_20_O2' : 'dynamic_20'}, inplace=True)

variant_runtime_df = metrics.get_runtime(variant_df, group, timers)
variant_ratio_df = metrics.get_normalized_runtime(variant_df, group, timers, 'Version', 'static')
boost_runtime_df = metrics.get_runtime(boost_df, group, timers)
boost_ratio_df = metrics.get_normalized_runtime(boost_df, group, timers, 'Version', 'static')
variant_O2_runtime_df = metrics.get_runtime(variant_O2_df, group, timers)
variant_O2_ratio_df = metrics.get_normalized_runtime(variant_O2_df, group, timers, 'Version', 'static')

plot.runtime(variant_runtime_df, 'Matrix', timers, 'Version', args.outdir + "/std")
plot.normalized_runtime(variant_ratio_df, 'Matrix', timers, 'Version', args.outdir + "/std")
plot.runtime(boost_runtime_df, 'Matrix', timers, 'Version', args.outdir + "/boost")
plot.normalized_runtime(boost_ratio_df, 'Matrix', timers, 'Version', args.outdir + "/boost")
plot.runtime(variant_O2_runtime_df, 'Matrix', timers, 'Version', args.outdir + "/std_O2")
plot.normalized_runtime(variant_O2_ratio_df, 'Matrix', timers, 'Version', args.outdir + "/std_O2")