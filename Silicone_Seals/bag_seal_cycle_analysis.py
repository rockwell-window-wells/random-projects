# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 14:56:00 2023

@author: Ryan.Larson
"""


import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.stats.api as sms
import resin_ttest_experiments as rtt
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.cbook import boxplot_stats


keys = ["green", "orange", "pink", "red", "purple", "brown"]

def step_1():
    file_list = ["green.csv", "orange.csv", "pink.csv", "red.csv", "purple.csv", "brown.csv"]
    
    mold_dfs = {key: None for key in keys}
    
    
    dropcols = ["Leak Time", "Leak Count", "Parts Count", "Weekly Count",
                "Monthly Count", "Trash Count", "Lead", "Assistant 1", "Assistant 2",
                "Assistant 3"]
    
    
    for file in file_list:
        color = file.replace(".csv","")
        df = pd.read_csv(file)
        
        # Deal with time column
        df["time"] = pd.to_datetime(df["time"])
        df["Date"] = pd.to_datetime(df["time"]).dt.date
        df.drop("time",axis=1,inplace=True)
        first_column = df.pop("Date")
        df.insert(0,"Date", first_column)
        
        for col in dropcols:
            df = df.drop(col,axis=1)
        df = df.dropna(how="all")
        for column in df.columns:
            df = df[df[column] != 0]
        
        mold_dfs[color] = df
        
        df.to_csv("{}2.csv".format(color),index=False)
        # df = df.dropna(axis=0)
        # df.to_csv(bag, index=False)
        
    return mold_dfs
        

def step_2():
    """
    

    Returns
    -------
    None.

    """
    file_list = ["green2.csv", "orange2.csv", "pink2.csv", "red2.csv", "purple2.csv", "brown2.csv"]
    
    for i,file in enumerate(file_list):
        if i == 0:
            df_all = pd.read_csv(file)
        else:
            df = pd.read_csv(file)
            df_all = pd.concat([df_all, df], axis=0, ignore_index=True)
            
    df_all.dropna(axis=0,inplace=True)
    
    return df_all


def filter_saturated(df):
    df = df[df["Layup Time"] != 276]
    df = df[df["Layup Time"] != 275]
    df = df[df["Close Time"] != 90]
    df = df[df["Resin Time"] != 180]
    df.drop_duplicates(inplace=True)
    return df
    
    # mold_dfs = {key: None for key in keys}
    
    # for file in file_list:
    #     color = file.replace("2.csv","")
    #     df = pd.read_csv(file)
    #     df = df[df["Layup Time"] != 276]
    #     df = df[df["Layup Time"] != 275]
    #     df = df[df["Close Time"] != 90]
    #     df = df[df["Resin Time"] != 180]
        
    #     # Remove duplicates
    #     df.drop_duplicates(inplace=True)
        
    #     mold_dfs[color] = df
        
    #     df.to_csv("{}3.csv".format(color),index=False)
    
    
# alldata = pd.concat(frames)

# feature_vals = list(alldata["Bag"].unique())
# n_feature_vals = len(feature_vals)

# df_features = [alldata.where(alldata["Bag"] == feature_val) for feature_val in feature_vals]
# df_features = [df_feature.dropna(axis=0) for df_feature in df_features]


# # rtt.oneway_anova(df_features, "Cycle Time")

# rtt.find_stat_difference_2group(df_features[0], df_features[1], "Resin Time")
# # rtt.find_stat_difference_2group(df_features[0], df_features[1], "Cycle Time")


if __name__ == "__main__":
    # mold_dfs = step_1()
    df_all = step_2()
    
    df10 = df_all[df_all["Bag"]==10.0]
    df11 = df_all[df_all["Bag"]==11.0]
    df12 = df_all[df_all["Bag"]==12.0]
    df13 = df_all[df_all["Bag"]==13.0]
    df14 = df_all[df_all["Bag"]==14.0]
    df15 = df_all[df_all["Bag"]==15.0]
    df16 = df_all[df_all["Bag"]==16.0]
    
    df10_filtered = filter_saturated(df10)
    df11_filtered = filter_saturated(df11)
    df12_filtered = filter_saturated(df12)
    df13_filtered = filter_saturated(df13)
    df14_filtered = filter_saturated(df14)
    df15_filtered = filter_saturated(df15)
    df16_filtered = filter_saturated(df16)
    df_all_filtered = filter_saturated(df_all)
    
    df10_no_outliers = df10_filtered.copy()
    df11_no_outliers = df11_filtered.copy()
    df12_no_outliers = df12_filtered.copy()
    df13_no_outliers = df13_filtered.copy()
    df14_no_outliers = df14_filtered.copy()
    df15_no_outliers = df15_filtered.copy()
    df16_no_outliers = df16_filtered.copy()
    
    ## Filter out outliers for each bag ##
    df_no_outliers_list = [df10_no_outliers, df11_no_outliers, df12_no_outliers, df13_no_outliers, df14_no_outliers, df15_no_outliers, df16_no_outliers]
    
    for i,df in enumerate(df_no_outliers_list):
        layup_stats = boxplot_stats(list(df["Layup Time"]))[0]
        resin_stats = boxplot_stats(list(df["Resin Time"]))[0]
        close_stats = boxplot_stats(list(df["Close Time"]))[0]
        cycle_stats = boxplot_stats(list(df["Cycle Time"]))[0]
        
        layup_conditions = [(df["Layup Time"] > layup_stats["whishi"]) | (df["Layup Time"] < layup_stats["whislo"])]
        close_conditions = [(df["Close Time"] > close_stats["whishi"]) | (df["Close Time"] < close_stats["whislo"])]
        resin_conditions = [(df["Resin Time"] > resin_stats["whishi"]) | (df["Resin Time"] < resin_stats["whislo"])]
        cycle_conditions = [(df["Cycle Time"] > cycle_stats["whishi"]) | (df["Cycle Time"] < cycle_stats["whislo"])]
        
        df["Layup Outlier"] = np.transpose(np.where(layup_conditions, True, False))
        df["Close Outlier"] = np.transpose(np.where(close_conditions, True, False))
        df["Resin Outlier"] = np.transpose(np.where(resin_conditions, True, False))
        df["Cycle Outlier"] = np.transpose(np.where(cycle_conditions, True, False))
        
        
    df_all_no_outliers = pd.concat([df10_no_outliers, df11_no_outliers])
    df_all_no_outliers = pd.concat([df_all_no_outliers, df12_no_outliers])
    df_all_no_outliers = pd.concat([df_all_no_outliers, df13_no_outliers])
    df_all_no_outliers = pd.concat([df_all_no_outliers, df14_no_outliers])
    df_all_no_outliers = pd.concat([df_all_no_outliers, df15_no_outliers])
    df_all_no_outliers = pd.concat([df_all_no_outliers, df16_no_outliers])
    
    df_all_layup_filtered = df_all_no_outliers[df_all_no_outliers["Layup Outlier"] == False]
    df_all_close_filtered = df_all_no_outliers[df_all_no_outliers["Close Outlier"] == False]
    df_all_resin_filtered = df_all_no_outliers[df_all_no_outliers["Resin Outlier"] == False]
    df_all_cycle_filtered = df_all_no_outliers[df_all_no_outliers["Cycle Outlier"] == False]
    
    sns.set(rc={"figure.dpi":300, "figure.figsize":(15.0, 8.27)})
    sns.set_style("whitegrid")
    palette_str = "Paired"
    
    ### Bag Days as x
    # Layup Time
    plt.figure()
    sns.relplot(data=df_all_layup_filtered, x="Bag Days", y="Layup Time", hue="Bag", palette=palette_str)
    plt.title("Age Effects on Layup Time")
    
    # Close Time
    plt.figure()
    sns.relplot(data=df_all_close_filtered, x="Bag Days", y="Close Time", hue="Bag", palette=palette_str)
    plt.title("Age Effects on Close Time")
    
    # Resin Time
    plt.figure()
    sns.relplot(data=df_all_resin_filtered, x="Bag Days", y="Resin Time", hue="Bag", palette=palette_str)
    plt.title("Age Effects on Resin Time")
    
    # Cycle Time
    plt.figure()
    sns.relplot(data=df_all_cycle_filtered, x="Bag Days", y="Cycle Time", hue="Bag", palette=palette_str)
    plt.title("Age Effects on Cycle Time")
    
    ### Bag Cycles as x
    # Layup Time
    plt.figure()
    sns.relplot(data=df_all_layup_filtered, x="Bag Cycles", y="Layup Time", hue="Bag", palette=palette_str)
    plt.title("Age Effects on Layup Time")
    
    # Close Time
    plt.figure()
    sns.relplot(data=df_all_close_filtered, x="Bag Cycles", y="Close Time", hue="Bag", palette=palette_str)
    plt.title("Age Effects on Close Time")
    
    # Resin Time
    plt.figure()
    sns.relplot(data=df_all_resin_filtered, x="Bag Cycles", y="Resin Time", hue="Bag", palette=palette_str)
    plt.title("Age Effects on Resin Time")
    
    # Cycle Time
    plt.figure()
    sns.relplot(data=df_all_cycle_filtered, x="Bag Cycles", y="Cycle Time", hue="Bag", palette=palette_str)
    plt.title("Age Effects on Cycle Time")
    
    ### Boxplot by bag for overall differences
    # Layup Time
    plt.figure()
    ax = sns.boxplot(data=df_all_layup_filtered, x="Bag", y="Layup Time", palette=palette_str)
    medians = df_all_layup_filtered.groupby(['Bag'])['Layup Time'].median().values
    nobs = df_all_layup_filtered['Bag'].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]
    pos = range(len(nobs))
    for tick,label in zip(pos,ax.get_xticklabels()):
        ax.text(pos[tick],
                medians[tick] + 0.04,
                nobs[tick],
                horizontalalignment='center',
                size='small',
                color='w',
                weight='semibold')
    
    # Close Time
    plt.figure()
    ax = sns.boxplot(data=df_all_close_filtered, x="Bag", y="Close Time", palette=palette_str)
    medians = df_all_close_filtered.groupby(['Bag'])['Close Time'].median().values
    nobs = df_all_close_filtered['Bag'].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]
    pos = range(len(nobs))
    for tick,label in zip(pos,ax.get_xticklabels()):
        ax.text(pos[tick],
                medians[tick] + 0.04,
                nobs[tick],
                horizontalalignment='center',
                size='small',
                color='w',
                weight='semibold')
    
    # Resin Time
    plt.figure()
    ax = sns.boxplot(data=df_all_resin_filtered, x="Bag", y="Resin Time", palette=palette_str)
    medians = df_all_resin_filtered.groupby(['Bag'])['Resin Time'].median().values
    nobs = df_all_resin_filtered['Bag'].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]
    pos = range(len(nobs))
    for tick,label in zip(pos,ax.get_xticklabels()):
        ax.text(pos[tick],
                medians[tick] + 0.04,
                nobs[tick],
                horizontalalignment='center',
                size='small',
                color='w',
                weight='semibold')
    
    # Cycle Time
    plt.figure()
    ax = sns.boxplot(data=df_all_cycle_filtered, x="Bag", y="Cycle Time", palette=palette_str)
    medians = df_all_cycle_filtered.groupby(['Bag'])['Cycle Time'].median().values
    nobs = df_all_cycle_filtered['Bag'].value_counts().values
    nobs = [str(x) for x in nobs.tolist()]
    nobs = ["n: " + i for i in nobs]
    pos = range(len(nobs))
    for tick,label in zip(pos,ax.get_xticklabels()):
        ax.text(pos[tick],
                medians[tick] + 0.04,
                nobs[tick],
                horizontalalignment='center',
                size='small',
                color='w',
                weight='semibold')
    
    