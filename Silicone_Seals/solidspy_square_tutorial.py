# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 09:20:44 2022

@author: Ryan.Larson
"""

import matplotlib.pyplot as plt  # load matplotlib
from solidspy import solids_GUI  # import our package
import os


directory = os.getcwd() + "\\"

UC, E_nodes, S_nodes = solids_GUI(plot_contours=True, compute_strains=True, folder=directory)

plt.show()