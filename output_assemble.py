'''set up'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os 
from astropy import units, constants
plt.rcParams.update({'font.size': 10})

lam_min = 2000
lam_max = 18000
nu_min = constants.c.value / lam_max
nu_max = constants.c.value / lam_min
nbins = 100
npacks = 10000

scale_f = 2.4*0.5

#observation csv files are imported and plotted from original data - no changes made
#df contains the combined results from every run of the simulation 

df = pd.read_csv(sys.argv[1]+'/run.csv')
obs1_df = pd.read_csv('spectra/AT2017gfo_XSHOOTER_MJD-57983.969_Phase-1.43.dat', sep = '\t', names = ['Wavelength', 'Flux'])
obs2_df = pd.read_csv('spectra/AT2017gfo_XSHOOTER_MJD-57984.969_Phase-2.42.dat', sep = ' ', names = ['Wavelength', 'Flux'])
obs3_df = pd.read_csv('spectra/AT2017gfo_XSHOOTER_MJD-57985.974_Phase-3.41.dat', sep = ' ', names = ['Wavelength', 'Flux'])
obs4_df = pd.read_csv('spectra/AT2017gfo_XSHOOTER_MJD-57986.974_Phase-4.40.dat', sep = ' ', names = ['Wavelength', 'Flux'])
obs5_df = pd.read_csv('spectra/AT2017gfo_XSHOOTER_MJD-57987.980_Phase-5.40.dat', sep = ' ', names = ['Wavelength', 'Flux'])
obs6_df = pd.read_csv('spectra/AT2017gfo_XSHOOTER_MJD-57988.990_Phase-6.40.dat', sep = ' ', names = ['Wavelength', 'Flux'])
obs7_df = pd.read_csv('spectra/AT2017gfo_XSHOOTER_MJD-57990.000_Phase-7.40.dat', sep = ' ', names = ['Wavelength', 'Flux'])
obs8_df = pd.read_csv('spectra/AT2017gfo_XSHOOTER_MJD-57991.000_Phase-8.40.dat', sep = ' ', names = ['Wavelength', 'Flux'])

#add column to sort packets as a fraction of the maximum arrival time recorded

max_arrival_time = df['Arrival time'].max()
df['Fraction time'] = (df['Arrival time'] / max_arrival_time)

'''eliminate the scattered packets: those with flag = true
altered to plot just scattered packets or just unscattered packets
commented out to plot all packets'''

#no_scatter_df = df[(df['Scattered'] != True)]
#no_scatter_df.to_csv('combined_df_scatter_only.csv', index = False)
#df = no_scatter_df

'''function to filter the packets based on percentage of max arrival time'''
#not used for plots in final project
#if needed: replace 'day_df' with 'frac_df' in lines 75-82

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
        bb_weight_hour[i] = (bb_weight_hour[i] *float(nbins) / npacks) * hour_bb[i] #* ((0.1 * constants.c.value * release_day[i]) ** 2)
    return(hour_wavelength, bb_weight_hour)

df_01 = day_df(df, 1, 2)
df_02 = day_df(df, 2, 3)
df_03 = day_df(df, 3, 4)
df_04 = day_df(df, 4, 5)
df_05 = day_df(df, 5, 6)
df_06 = day_df(df, 6, 7)
df_07 = day_df(df, 7, 8)
df_08 = day_df(df, 8, 9)

lam_01, bb_01 = generate_arrays(df_01)
lam_02, bb_02 = generate_arrays(df_02)
lam_03, bb_03 = generate_arrays(df_03)
lam_04, bb_04 = generate_arrays(df_04)
lam_05, bb_05 = generate_arrays(df_05)
lam_06, bb_06 = generate_arrays(df_06)
lam_07, bb_07 = generate_arrays(df_07)
lam_08, bb_08 = generate_arrays(df_08)

'''plot histogram'''
#divison by 7.5e-47 is a constant scale factor chosen to align observed data and simulation results

fig, axs = plt.subplots(2, 4, sharex = True, sharey = False,dpi = 500,figsize = (15,5))

axs[0,0].hist((lam_01 / 1e-10),
        bins=np.linspace(lam_min, lam_max, nbins),
        histtype="step",
        weights = bb_01,
        #label= '1 < t < 2 days',
        label = 'Simulated',
        color = 'deeppink',
        linewidth = 1,
        zorder = 2)

scale_f = np.sum(obs1_df['Flux'])/np.sum(bb_01)/7.5e-47/len(obs1_df['Flux'])*nbins

#print(np.sum(obs1_df['Flux']))
#print(np.sum(bb_01))

axs[0,0].plot(obs1_df['Wavelength'], (obs1_df['Flux'] / 7.5e-47 / scale_f), zorder = 1, color = 'b', label = 'Observed')

axs[0,0].set_title('1 < t < 2 days', fontsize = 10)
axs[0,0].set_ylabel(r"$F_{\lambda}$"
                    "\n"
                    r"$[x10^{30} ergs/s/cm^{2}/\mathrm{\AA}]$")

axs[0,1].hist((lam_02 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_02,
            #label= '2 < t < 3 days',
            color = 'deeppink',
            linewidth = 1,
            zorder = 2)

axs[0,1].plot(obs2_df['Wavelength'], (obs2_df['Flux'] / 7.5e-47 / scale_f), zorder = 1, color = 'b')

axs[0,1].set_title('2 < t < 3 days', fontsize = 10)
axs[0,1].set_ylabel(r"$F_{\lambda}$"
                    "\n"
                    r"$[x10^{30} ergs/s/cm^{2}/\mathrm{\AA}]$")

axs[0,2].hist((lam_03 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_03,
            #label= '3 < t < 4 days',
            color = 'deeppink',
            linewidth = 1,
            zorder = 2)

axs[0,2].plot(obs3_df['Wavelength'], (obs3_df['Flux'] / 7.5e-47 / scale_f), zorder = 1, color = 'b')

axs[0,2].set_title('3 < t < 4 days', fontsize = 10)
axs[0,2].set_ylabel(r"$F_{\lambda}$"
                    "\n"
                    r"$[x10^{30} ergs/s/cm^{2}/\mathrm{\AA}]$")

axs[0,3].hist((lam_04 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_04,
            #label= '4 < t < 5 days',
            color = 'deeppink',
            linewidth = 1,
            zorder = 2)

axs[0,3].plot(obs4_df['Wavelength'], (obs4_df['Flux'] / 7.5e-47 / scale_f), zorder = 1, color = 'b')

axs[0,3].set_title('4 < t < 5 days', fontsize = 10)
axs[0,3].set_ylabel(r"$F_{\lambda}$"
                    "\n"
                    r"$[x10^{30} ergs/s/cm^{2}/\mathrm{\AA}]$")

axs[1,0].hist((lam_05 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_05,
            #label= '5 < t < 6 days',
            color = 'deeppink',
            linewidth = 1,
            zorder = 2)   

axs[1,0].plot(obs5_df['Wavelength'], (obs5_df['Flux'] / 7.5e-47 / scale_f), zorder = 1, color = 'b')    

axs[1,0].set_title('5 < t < 6 days', fontsize = 10)
axs[1,0].set_ylabel(r"$F_{\lambda}$"
                    "\n"
                    r"$[x10^{29} ergs/s/cm^{2}/\mathrm{\AA}]$")

axs[1,1].hist((lam_06 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_06,
            #label= '6 < t < 7 days',
            color = 'deeppink',
            linewidth = 1,
            zorder = 2) 

axs[1,1].plot(obs6_df['Wavelength'], (obs6_df['Flux'] / 7.5e-47 / scale_f), zorder = 1, color = 'b')

axs[1,1].set_title('6 < t < 7 days', fontsize = 10)
axs[1,1].set_ylabel(r"$F_{\lambda}$"
                    "\n"
                    r"$[x10^{29} ergs/s/cm^{2}/\mathrm{\AA}]$")

axs[1,2].hist((lam_07 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_07,
            #label= '7 < t < 8 days',
            color = 'deeppink',
            linewidth = 1,
            zorder = 2)

axs[1,2].plot(obs7_df['Wavelength'], (obs7_df['Flux'] / 7.5e-47 / scale_f), zorder = 1, color = 'b')

axs[1,2].set_title('7 < t < 8 days', fontsize = 10)
axs[1,2].set_ylabel(r"$F_{\lambda}$"
                    "\n"
                    r"$[x10^{29} ergs/s/cm^{2}/\mathrm{\AA}]$")

axs[1,3].hist((lam_08 / 1e-10),
            bins=np.linspace(lam_min, lam_max, nbins),
            histtype="step",
            weights = bb_08,
            #label= '8 < t < 9 days',
            color = 'deeppink',
            linewidth = 1,
            zorder = 2)

axs[1,3].plot(obs8_df['Wavelength'], (obs8_df['Flux'] / 7.5e-47 / scale_f), zorder = 1, color = 'b')

axs[1,3].set_title('8 < t < 9 days', fontsize = 10)
axs[1,3].set_ylabel(r"$F_{\lambda}$"
                    "\n"
                    r"$[x10^{29} ergs/s/cm^{2}/\mathrm{\AA}]$")

for ax in axs.flat:
    ax.set(xlabel = r"$\lambda$ $[\mathrm{\AA}]$")
           #ylabel = r"$F_{\lambda}$")
    #ax.label_outer()
    #ax.legend()
    ax.set_xlim(lam_min, lam_max)
    ax.set_ylim(0)
    #ax.legend()
    ax.yaxis.offsetText.set_visible(False)

plt.figlegend(loc = 9, ncol = 2)
'''axs[2,1].axis('off')
axs[2,1].legend_ = None
axs[2,2].axis('off')
axs[2,2].legend_ = None
axs[2,3].axis('off')
axs[2,3].legend_ = None'''

#plt.text(0.85, 0.02, 'combined data, \nscattered packets included, \ntimestep = 0.5 hour', fontsize=10, transform=plt.gcf().transFigure)
path = sys.argv[1]
head_tail = os.path.split(path)
print(head_tail[1])

plt.savefig(sys.argv[1]+'/'+head_tail[1]+'_'+'comparison.png')
