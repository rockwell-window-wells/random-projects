# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 12:51:08 2022

@author: Ryan.Larson
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

filename = "Bag Seal Force Results.xlsx"

side_df = pd.read_excel(filename, sheet_name="Side")
close_df = pd.read_excel(filename, sheet_name="Close_Corner")
opposite_df = pd.read_excel(filename, sheet_name="Opposite_Corner")

side_df["Pressure Drop"] = side_df["Max Pressure"] - side_df["15 sec pressure"]
close_df["Pressure Drop"] = close_df["Max Pressure"] - close_df["15 sec pressure"]
opposite_df["Pressure Drop"] = opposite_df["Max Pressure"] - opposite_df["15 sec pressure"]

side_df["Max Force in Z"] = side_df["Max Force"]
close_df["Max Force in Z"] = close_df["Max Force"]*np.sin(1.0304)
opposite_df["Max Force in Z"] = opposite_df["Max Force"]*np.sin(1.0304)

side_df["Load Location"] = ["Side" for x in range(len(side_df))]
close_df["Load Location"] = ["Close Corner" for x in range(len(close_df))]
opposite_df["Load Location"] = ["Opposite Corner" for x in range(len(opposite_df))]

alldata = pd.concat([side_df, close_df, opposite_df], axis=0)

fig1 = plt.figure(dpi=300)
sns.scatterplot(data=alldata, x="Max Pressure", y="Max Force in Z", hue="Bag", style="Load Location", palette="coolwarm")

fig2 = plt.figure(dpi=300)
sns.scatterplot(data=alldata, x="Pressure Drop", y="Max Force in Z", hue="Bag", style="Load Location", palette="coolwarm")

fig3 = plt.figure(dpi=300)
sns.scatterplot(data=alldata, x="Max Pressure", y="Max Force", hue="Bag", style="Load Location", palette="coolwarm")

fig4 = plt.figure(dpi=300)
sns.scatterplot(data=alldata, x="Pressure Drop", y="Max Force", hue="Bag", style="Load Location", palette="coolwarm")

# fig1 = plt.figure(dpi=300)
# sns.scatterplot(data=side_df, x="Max Pressure", y="Max Force in Z", hue="Bag", palette="coolwarm").set(title="Side Lift Point")

# fig2 = plt.figure(dpi=300)
# sns.scatterplot(data=close_df, x="Max Pressure", y="Max Force in Z", hue="Bag", palette="coolwarm").set(title="Close Corner Lift Point")

# fig3 = plt.figure(dpi=300)
# sns.scatterplot(data=opposite_df, x="Max Pressure", y="Max Force in Z", hue="Bag", palette="coolwarm").set(title="Opposite Corner Lift Point")

# fig4 = plt.figure(dpi=300)
# sns.scatterplot(data=side_df, x="Pressure Drop", y="Max Force in Z", hue="Bag", palette="coolwarm").set(title="Side Lift Point")

# fig5 = plt.figure(dpi=300)
# sns.scatterplot(data=close_df, x="Pressure Drop", y="Max Force in Z", hue="Bag", palette="coolwarm").set(title="Close Corner Lift Point")

# fig6 = plt.figure(dpi=300)
# sns.scatterplot(data=opposite_df, x="Pressure Drop", y="Max Force in Z", hue="Bag", palette="coolwarm").set(title="Opposite Corner Lift Point")
