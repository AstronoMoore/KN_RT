#combine dataframes from 1.5 - 4.5 days
#plot combined data

'''set up'''

import os
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from astropy import units, constants
from pathlib import Path
import glob
import sys


lam_min = 2000
lam_max = 18000
nu_min = constants.c.value / lam_max
nu_max = constants.c.value / lam_min
nbins = 100
npacks = 10000

'''read in csv files and combine, save final file to csv'''

path = '.'

dfs = list()


'''set random seed for introducing jitter
   timestep units = hours
   used to make packet arrival times more continuous
   arrival time moved by +/- 0.5 timestep'''

np.random.seed(42)
timestep = 2.4

'''simulation defines t = 0 as the beginning of each run
   therefore need to add on the time differences between each run to give accurate arrival times
   Day flag added to each csv as simulation runs
   convert this to s and add to Arrival Time'''

directory_path = sys.argv[1]
csv_files = glob.glob(f"{directory_path}" + '/*.csv')


for filename in csv_files:
    data = pd.read_csv(filename)
    data['Fixed arrival time'] = data.apply(lambda row: row['Arrival time'] + (row['Day'] * 24 * 60 * 60) + (np.random.random()-0.5)*timestep*3600, axis=1)
    dfs.append(data)

'''NB: this is hideously inefficient and incredibly slow
   but was the only method I could get working in the timeframe given for the project'''

'''
for i in range(len(df)):

    rnd = np.random.random()
#    print(rnd)
    if rnd >= 0.5:
        data['Fixed arrival time'] = data['Fixed arrival time'] + (0.5 * timestep * 60 * 60)
    else:
        data['Fixed arrival time'] = data['Fixed arrival time'] - (0.5 * timestep * 60 * 60)
'''
df = []
df = pd.DataFrame({'Arrival time': [], 'Fixed arrival time':[], 'Wavelength':[],'BB Flux':[],'Day':[]} )

try: 
    df = pd.concat(dfs, ignore_index = True)
    df.to_csv(sys.argv[1]+'/run.csv')
except ValueError as e:
    print(e)
#sys.exit()


'''add column to sort packets as a fraction of the maximum arrival time recorded'''

max_arrival_time = df['Arrival time'].max()
df['Fraction time'] = (df['Arrival time'] / max_arrival_time)

'''if uncommented: section used to omit packets based on scattering flag if plotting only (un)scattered packets'''

#no_scatter_df = df[(df['Scattered'] != True)]
#no_scatter_df.to_csv('combined_df_scatter_only.csv', index = False)
#df = no_scatter_df

'''function to filter the packets based on percentage of max arrival time'''

def frac_df(df, low_bound, high_bound):

    fraction_dataframe = df[(df['Fraction time'] > low_bound) & (df['Fraction time'] <= high_bound)]
    return(fraction_dataframe)

'''function to filter the packets based on arrival time in days
   input time in days and convert to seconds for filtering df'''

def day_df(df, low_time, high_time):
    
    low_time = low_time * 24 * 60 * 60
    high_time = high_time * 24 * 60 * 60
    time_dataframe = df[(df['Fixed arrival time'] > low_time) & (df['Fixed arrival time'] <= high_time)]
    return(time_dataframe)

'''convert the contents of each filtered data frame to numpy arrays to plot histograms'''

def generate_arrays(hour_dataframe):
    
    hour_wavelength = (hour_dataframe[['Wavelength']].dropna().to_numpy())
    hour_bb = (hour_dataframe[['BB Flux']].dropna().to_numpy())
    release_day = (hour_dataframe[['Day']].dropna().to_numpy())

    bb_weight_hour = np.ones(len(hour_wavelength))
    
    for i in range(0, len(hour_wavelength)):
        bb_weight_hour[i] = (bb_weight_hour[i] *float(nbins) / npacks) * hour_bb[i] * ((0.1 * release_day[i]) ** 2)
    return(hour_wavelength, bb_weight_hour)

'''df needs converted to arrays to plot: sections commented out based on timeframe needing covered by histograms'''

'''this section plots 1.5 - 5 days'''

'''this section plots 1 - 9 days'''
df_01 = day_df(df, 1, 2)
df_02 = day_df(df, 2, 3)
df_03 = day_df(df, 3, 4)
df_04 = day_df(df, 4, 5)
df_05 = day_df(df, 5, 6)
df_06 = day_df(df, 6, 7)
df_07 = day_df(df, 7, 8)
df_08 = day_df(df, 8, 9)

'''leave this section uncommented all the time'''

lam_01, bb_01 = generate_arrays(df_01)
lam_02, bb_02 = generate_arrays(df_02)
lam_03, bb_03 = generate_arrays(df_03)
lam_04, bb_04 = generate_arrays(df_04)
lam_05, bb_05 = generate_arrays(df_05)
lam_06, bb_06 = generate_arrays(df_06)
lam_07, bb_07 = generate_arrays(df_07)
lam_08, bb_08 = generate_arrays(df_08)

fig, axs = plt.subplots(2, 4, sharex = True, sharey = False)

axs[0,0].hist((lam_01 / 1e-10),
        bins=np.linspace(lam_min, lam_max, nbins),
        histtype="step",
        weights = bb_01,
        label= '1 < t < 2 days',
        color = 'forestgreen')

axs[0,1].hist((lam_02 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_02,
            label= '2 < t < 3 days',
            color = 'forestgreen')

axs[0,2].hist((lam_03 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_03,
            label= '3 < t < 4 days',
            color = 'forestgreen')

axs[0,3].hist((lam_04 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_04,
            label= '4 < t < 5 days',
            color = 'forestgreen')

axs[1,0].hist((lam_05 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_05,
            label= '5 < t < 6 days',
            color = 'forestgreen')       

axs[1,1].hist((lam_06 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_06,
            label= '6 < t < 7 days',
            color = 'forestgreen') 

axs[1,2].hist((lam_07 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_07,
            label= '7 < t < 8 days',
            color = 'forestgreen')

axs[1,3].hist((lam_08 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_08,
            label= '8 < t < 9 days',
            color = 'forestgreen')

for ax in axs.flat:
    ax.set(xlabel = r"$\lambda$ $[\mathrm{\AA}]$",
           ylabel = r"$F_{\lambda}$")
    #ax.label_outer()
    ax.legend()

ax.set_xlim(lam_min, lam_max)
ax.legend()

'''uncomment if plotting 1.5 - 5 days'''

'''axs[2,1].axis('off')
axs[2,1].legend_ = None
axs[2,2].axis('off')
axs[2,2].legend_ = None
axs[2,3].axis('off')
axs[2,3].legend_ = None'''



plt.text(0.85, 0.02, 'combined data, \nscattered packets included, \ntimestep = 0.5 hour', fontsize=10, transform=plt.gcf().transFigure)

#plt.show()
print(sys.argv[1]+'/0.5h_ScatteringEffectIncl.csv')
df.to_csv(sys.argv[1]+'/0.5h_ScatteringEffectIncl.csv')
