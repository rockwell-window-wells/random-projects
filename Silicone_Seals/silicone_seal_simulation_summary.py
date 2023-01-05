# -*- coding: utf-8 -*-
"""
Created on Fri Oct 28 14:17:17 2022

@author: Ryan.Larson
"""

import pandas as pd
import plotly.express as px

filename = "Flat Bag Backer Chamber Results.xlsx"
directory = "C:/Users/Ryan.Larson.ROCKWELLINC/OneDrive - Rockwell Inc/Desktop/"

file = directory + filename

df = pd.read_excel(file, sheet_name="Rect_chamber_rounded_bag")
# df = pd.read_excel(file, sheet_name="Rect_chamber_thick_bag")
df["Pressure/Displacement"] = df["Avg Contact Pressure (psi)"] / df["Max Displacement (in)"]

pressures = list(df["Avg Contact Pressure (psi)"])
sufficient_pressure = ["Sufficient" if pressure>12.28 else "Insufficient" for pressure in pressures]
df["Sufficient Pressure"] = sufficient_pressure

fig1 = px.scatter_3d(df, x="wc", y="hc", z="tc", color="Max Displacement (in)", symbol="Sufficient Pressure", title="Max Displacement")
fig2 = px.scatter_3d(df, x="wc", y="hc", z="tc", color="Avg Contact Pressure (psi)", symbol="Sufficient Pressure", title="Contact Pressure")
fig3 = px.scatter_3d(df, x="wc", y="hc", z="tc", color="Pressure/Displacement", symbol="Sufficient Pressure", title="Pressure/Displacement")


fig1.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))
fig2.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))
fig3.update_layout(coloraxis_colorbar=dict(yanchor="top", y=1, x=0, ticks="outside"))

fig1.show()
fig2.show()
fig3.show()