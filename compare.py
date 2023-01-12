# ## Compare nwf to wf parameterization

# ### Imports

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
from windrose import plot_windrose
import matplotlib.cm as cm
import argparse
from mycolorpy import colorlist as mcp
import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(
                description='Compare LLJs for nwf and wf',
                prog='compare.py')

parser.add_argument('--nwf_file',
                    help='Name of the file with nwf data',
                    required=True)
parser.add_argument('--wf_file',
                    help='The name of the file with wf data',
                    required=True)
parser.add_argument('--wf_name',
                    help='The name of the wf parametrization',
                    required=True)
parser.add_argument('--location',
                    help='The location of the two files',
                    required=True)
parser.add_argument('--plot_path',
                    help='Path to save plots to',
                    required=True)

args = parser.parse_args()

# ### Read in data

wf_name = args.wf_name
location = args.location

if wf_name == "CA100":
    nwf_df = pd.read_csv(f'LLJ_data/{args.nwf_file}')
    nwf = pd.concat([nwf_df.set_index('Time').loc['2019-09-01':'2019-11-01'], 
                     nwf_df.set_index('Time').loc['2020-07-01':]]).reset_index()
    time_period = "July-October"
    m1 = 7
    m2 = 11
    # These values for windrose subplots
    srows = 1
    scols = 5
    lineup = 6
else:
    nwf = pd.read_csv(f'LLJ_data/{args.nwf_file}')
    time_period = "whole_year"
    m1 = 1
    m2 = 13
    # These values for windrose subplots
    srows = 2
    scols = 6
    lineup = 0
wf =  pd.read_csv(f'LLJ_data/{args.wf_file}')

# ### NEW PLOT: LLJ Classification histogram

plt.figure()

ax = plt.gca()
    
def find_hgts(ds):
    hgts = []
    for i in range(4):
        if i in ds['LLJ-classification'].unique():
            hgts.append(ds['LLJ-classification'].value_counts()[i])
        else:
            hgts.append(0)
    return hgts

nwf_class = find_hgts(nwf)
wf_class = find_hgts(wf)

ax.bar(x=np.array([0, 1, 2, 3]) -0.2,
       height=nwf_class,
       width=0.4,
       label='NWF',
       color='cadetblue')

ax.bar(x=np.array([0, 1, 2, 3]) +0.2,
       height=wf_class,
       width=0.4,
       label=wf_name,
       color='powderblue')

plt.title(f'LLJ classifications at {location}')

ax.set_xticks(np.arange(0, 4))

plt.legend()

plt.savefig(f'{args.plot_path}/classifications.png')

plt.close()


# ### PRINT TO TERMINAL: Number of events

print(f'Number of LLJs at {location}:')
print(f'{wf_name}: ', len(wf) - wf['LLJ-classification'].isna().sum())
print('NWF: ', len(nwf) - nwf['LLJ-classification'].isna().sum())

# ### NEW PLOT: Time of day

wf.Time = pd.to_datetime(wf.Time)
nwf.Time = pd.to_datetime(nwf.Time)

wf = wf.dropna(axis=0)
nwf = nwf.dropna(axis=0)

plt.figure(figsize=(11, 5))

ax = plt.gca()
ax2 = ax.twiny()

ax.bar(x=nwf.Time.groupby(nwf.Time.dt.hour).count().index -0.2, 
       height=nwf.Time.groupby(nwf.Time.dt.hour).count().values,
       width=0.4,
       label='NWF',
       color='cadetblue')
ax.bar(x=wf.Time.groupby(wf.Time.dt.hour).count().index+0.2, 
       height=wf.Time.groupby(wf.Time.dt.hour).count().values,
       width=0.4, 
       label=wf_name,
       color='powderblue')

ax2.bar(x=nwf.Time.groupby(nwf.Time.dt.hour).count().index -0.2, 
       height=nwf.Time.groupby(nwf.Time.dt.hour).count().values,
       width=0.4,
       label='NWF',
       color='cadetblue')
ax2.bar(x=wf.Time.groupby(wf.Time.dt.hour).count().index+0.2, 
       height=wf.Time.groupby(wf.Time.dt.hour).count().values,
       width=0.4, 
       label=wf_name,
       color='powderblue')

ax.set_xticks(np.arange(0, 24))
ax2.set_xticks(ax.get_xticks())
ax2.set_xticklabels((np.arange(0, 24) - 5) % 24)

ax2.set_xlabel('Hour of the Day (EST)')
ax.set_xlabel('Hour of the day (UTC)')
ax.set_ylabel('LLJs')

plt.title(f'Time of day LLJs occurred: nwf and {wf_name} at {location}')
ax.legend()

plt.savefig(f'{args.plot_path}/timeofday.png')

plt.close();


# ### NEW PLOT: Seasonality

if wf_name != 'CA100':

    nwf.Time = pd.to_datetime(nwf.Time)
    nwf = nwf.dropna(axis=0)

    plt.figure(figsize=(11, 5))

    ax = plt.gca()

    ax.bar(x=nwf.Time.groupby(nwf.Time.dt.month).count().index -0.2, 
           height=nwf.Time.groupby(nwf.Time.dt.month).count().values,
           width=0.4,
           label='nwf',
           color='cadetblue')

    ax.bar(x=wf.Time.groupby(wf.Time.dt.month).count().index +0.2, 
           height=wf.Time.groupby(wf.Time.dt.month).count().values,
           width=0.4,
           label=wf_name,
           color='powderblue')

    ax.set_xticks(np.arange(1, 13))
    ax.set_xlabel('Month of the year')
    ax.set_ylabel('LLJs')
    plt.title(f'Months LLJs occurred at {location}')
    plt.legend();

    plt.savefig(f'{args.plot_path}/months.png')

    plt.close()

# ### NEW PLOT: Nose height by LLJ classification 

# make df variable for each level of LLJ
l0_nwf = nwf[nwf['LLJ-classification']==0]
l1_nwf = nwf[nwf['LLJ-classification']==1]
l2_nwf = nwf[nwf['LLJ-classification']==2]
l3_nwf = nwf[nwf['LLJ-classification']==3]

l0_wf = wf[wf['LLJ-classification']==0]
l1_wf = wf[wf['LLJ-classification']==1]
l2_wf = wf[wf['LLJ-classification']==2]
l3_wf = wf[wf['LLJ-classification']==3]

custom_lines = [Line2D([0], [0], color='cadetblue', lw=5),
                Line2D([0], [0], color='powderblue', lw=5)]

# make a box plot
plt.figure()
ax = plt.gca()

bplots = plt.boxplot([l0_nwf['Nose height (meters)'], 
                      l0_wf['Nose height (meters)'],
                      l1_nwf['Nose height (meters)'], 
                      l1_wf['Nose height (meters)'],
                      l2_nwf['Nose height (meters)'], 
                      l2_wf['Nose height (meters)'],
                      l3_nwf['Nose height (meters)'], 
                      l3_wf['Nose height (meters)']])

# fill with colors
colors = ['cadetblue', 'powderblue']
# loop through each plot
for i in range(8):
    box = bplots['boxes'][i]
    box_x = []
    box_y = []
    for j in range(4):
        box_x.append(box.get_xdata()[j])
        box_y.append(box.get_ydata()[j])
    box_coords = np.column_stack([box_x, box_y])
    ax.add_patch(Polygon(box_coords, facecolor=colors[i%2]))

ax.set_xticks(np.arange(0, 8, 2)+1.5)
ax.set_xticklabels(np.arange(0, 4))

plt.xlabel('LLJ classification')
plt.ylabel('Nose height (meters)')
plt.title(f'Nose height distribution by LLJ-classification at {location}')
ax.legend(custom_lines, ['NWF', wf_name]);

plt.savefig(f'{args.plot_path}/noseheights.png')

plt.close()

# ### Wind rose plots

# NEW PLOT: Whole year - nwf and wf subplot

fig = plt.figure(figsize=(14, 10))

for i in range(2):
    if i%2 == 0:
        df = nwf
        title = f"NWF - {time_period}"
    else:
        df = wf
        title = f'{wf_name} - {time_period}'

    direction = df['Wind direction at nose (degrees)']
    speed = df['Nose windspeed (m/s)']
    
    ax = fig.add_subplot(1, 2, i + 1, projection="windrose")
    ax.bar(direction, speed, bins=np.arange(10, 30, 5))
    ax.set_legend(prop={'size': 6})
    ax.set_title(title)
    
plt.savefig(f'{args.plot_path}/windrose_{time_period}.png')

plt.close()

# NEW PLOT: Monthly - NWF

fig = plt.figure(figsize=(25, 11))

plt.axis('off')
plt.title("Monthly wind roses - NWF", fontsize=20)

for i in range(m1-lineup, m2-lineup):
    df = nwf[nwf.Time.dt.month == i + lineup]
    
    direction = df['Wind direction at nose (degrees)']
    speed = df['Nose windspeed (m/s)']
    
    ax = fig.add_subplot(srows, scols, i, projection="windrose")
    ax.bar(direction, speed, bins=np.arange(10, 30, 5))
    ax.set_legend(prop={'size': 6})
    ax.set_title(f'Month={i + lineup}', fontsize=13)
    # Avoid matplotlib bug with ylimits for empty data
    ylim = ax.get_ylim()
    ax.set_ylim(ylim)
plt.box(False)
    
plt.savefig(f'{args.plot_path}/windrose_monthly_nwf.png')

plt.close()

# NEW PLOT: Monthly - WF

fig = plt.figure(figsize=(25, 11))

plt.axis('off')
plt.title(f"Monthly wind roses - {wf_name}", fontsize=20)

for i in range(m1, m2):
    df = wf[wf.Time.dt.month == i]
    
    direction = df['Wind direction at nose (degrees)']
    speed = df['Nose windspeed (m/s)']
    
    ax = fig.add_subplot(2, 6, i, projection="windrose")
    ax.bar(direction, speed, bins=np.arange(10, 30, 5))
    ax.set_legend(prop={'size': 6})
    ax.set_title(f'Month={i}', fontsize=13)
plt.box(False)
    
plt.savefig(f'{args.plot_path}/windrose_monthly_{wf_name}.png')

plt.close()


# NEW PLOT: LLJ classification by month

class_ds = pd.concat([nwf, wf]).dropna()
class_ds.Time = pd.to_datetime(class_ds.Time)

llj0 = class_ds[class_ds['LLJ-classification']==0]
llj1 = class_ds[class_ds['LLJ-classification']==1]
llj2 = class_ds[class_ds['LLJ-classification']==2]
llj3 = class_ds[class_ds['LLJ-classification']==3]

lljs = [llj0, llj1, llj2, llj3]
colors = mcp.gen_color(cmap="viridis", n=4)

plt.figure(figsize=(11, 5))

ax = plt.gca()

for i, el in enumerate(lljs):
    
    hgts = []
    for j in range(m1, m2):
        if j in el.Time.groupby(el.Time.dt.month).count():
            hgts.append(el.Time.groupby(el.Time.dt.month).count()[j])
        else:
            hgts.append(0)
    ax.bar(x=np.arange(m1, m2), height=hgts, label=f'LLJ{i}', color=colors[i])

ax.set_xticks(np.arange(m1, m2))
ax.set_xlabel('Month of the year')
ax.set_ylabel('Number of LLJs')
plt.title(f'LLJ classification by month at {location}')
plt.legend()

plt.savefig(f'{args.plot_path}/class_by_month.png')

plt.close()

# NEW PLOT: plot for time of day

plt.figure(figsize=(11, 5))

ax = plt.gca()

for i, el in enumerate(lljs):
    
    hgts = []
    for j in range(24):
        if j in el.Time.groupby(el.Time.dt.hour).count():
            hgts.append(el.Time.groupby(el.Time.dt.hour).count()[j])
        else:
            hgts.append(0)
    ax.bar(x=np.arange(0, 24), height=hgts, label=f'LLJ{i}', color=colors[i])

ax.set_xticks(np.arange(0, 24))
ax.set_xlabel('Hour of the Day')
ax.set_ylabel('Number of LLJs')
plt.title(f'LLJ classification by time of day at {location}')
plt.legend()

plt.savefig(f'{args.plot_path}/class_by_time.png')

plt.close()
