# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 09:33:24 2022

@author: Ryan.Larson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def get_proportionality(df):
    proportionality = []
    for row in range(len(df)):
        if row == 0 or row == len(df)-1:
            continue
        else:
            prop = (df.loc[row+1, "Stress (MPa)"] - df.loc[row-1, "Stress (MPa)"]) / (df.loc[row+1, "Strain (%)"] - df.loc[row-1, "Strain (%)"])
            proportionality.append(prop)
    return proportionality
    

F15 = "F15.xlsx"
F15i300 = "F15 i300.xlsx"
F16_1m100 = "F16_1 m100.xlsx"

filenames = [F15, F15i300, F16_1m100]

# Plot all the stress strain curves together for reference
for filename in filenames:
    fig = plt.figure(dpi=300)
    
    for testnum in range(1,7):
        # Load the data for the current test as a DataFrame
        df = pd.read_excel(filename, sheet_name=str(testnum))
        proportionality = get_proportionality(df)
        
        pct_change = []
        for line in range(1,len(proportionality)):
            pct = proportionality[line]/proportionality[line-1]
            pct = pct - 1
            pct_change.append(pct)
            
        
        
        sns.lineplot(data=df, x="Strain (%)", y="Stress (MPa)").set(title=filename)