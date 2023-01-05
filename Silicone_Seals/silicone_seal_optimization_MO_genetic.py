# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:06:12 2022

@author: Ryan.Larson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solidspy import solids_GUI
# from shapely.geometry import Polygon, Point
import netgen.geom2d as geom2d
import os
# from scipy.optimize import minimize
# from datetime import datetime
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator
import random
import plotly.express as px
# import math

###############################################################################
############################ SIMULATION FUNCTIONS #############################
###############################################################################

def rectangular_single_legs(w, wc, hc, tc, maxmesh):
    """
    

    Parameters
    ----------
    w : float
        Overall width of seal profile.
    wc : float
        Width of rectangular chamber, centered in the profile
    hc : float
        Height of the rectangular chamber.
    tc : float
        Thickness of the bag above the chamber.

    Returns
    -------
    None.

    """
    # # Validate inputs (wl < w/2, hl < h)
    # if wl > (w/2):
    #     raise ValueError("ERROR:\twl is too large")
    # if h < hl:
    #     raise ValueError("ERROR:\thl is too large")
        
    # Construct the points that define the seal profile
    wl = (w - wc)/2
    h = hc + tc
    
    coords = [[0,0],[wl,0],[wl,hc],[w-wl,hc],[w-wl,0],[w,0],[w,h],[0,h]]
    mesh = mesh_from_coords(coords,maxmesh)
    
    return mesh

def mesh_from_coords(coords,maxmesh):
    geo = geom2d.SplineGeometry()
    pts = [geo.AppendPoint(coord[0],coord[1]) for coord in coords]
    
    for i,pt in enumerate(pts):
        if i==len(pts)-1:
            geo.Append(["line",pt,pts[0]])
        else:
            geo.Append(["line",pt,pts[i+1]])
    
    mesh = geo.GenerateMesh(maxh=maxmesh)
    
    return mesh
    
def generate_solidspy_files_rectangular_single_legs(mesh, parameters, pressure, youngs_modulus, poisson_ratio):
    elements = list(mesh.Elements2D())
    pts = list(mesh.Points())
    
    nodes_df = pd.DataFrame()
    eles_df = pd.DataFrame()
    mater_df = pd.DataFrame()
    
    ## Prepare nodes.txt data ##
    nodex = []
    nodey = []
    for pt in pts:
        nodex.append(pt.p[0])
        nodey.append(pt.p[1])
    
    xbound = []
    ybound = []
    xboundnode = []
    yboundnode = []
    for i,pt in enumerate(pts):
        if i == 0:
            xbound.append(-1)
            ybound.append(-1)
            xboundnode.append(pt.p[0])
            yboundnode.append(pt.p[1])
        elif i !=0 and pt[1] == 0:
            xbound.append(-1)    # Fully fixed feet
            # xbound.append(0)    #  Roller boundary condition
            ybound.append(-1)
            xboundnode.append(pt.p[0])
            yboundnode.append(pt.p[1])
        else:
            xbound.append(0)
            ybound.append(0)
            
    nodes_df["X-coordinate"] = nodex
    nodes_df["Y-coordinate"] = nodey
    nodes_df["Boundary condition x"] = xbound
    nodes_df["Boundary condition y"] = ybound
    
    nodes_df.to_string('nodes.txt', header=False)
    
    ## Prepare eles.txt data ##
    eltype = []
    for el in elements:
        eltype.append(3)
        
    elmaterial = []
    for el in elements:
        elmaterial.append(0)
        
    elconnect1 = []
    elconnect2 = []
    elconnect3 = []
    for el in elements:
        elconnect1.append(int(str(el.vertices[0]))-1)
        elconnect2.append(int(str(el.vertices[1]))-1)
        elconnect3.append(int(str(el.vertices[2]))-1)
        
    eles_df["Element type"] = eltype
    eles_df["Material profile"] = elmaterial
    eles_df["Connection 1"] = elconnect1
    eles_df["Connection 2"] = elconnect2
    eles_df["Connection 3"] = elconnect3
    
    eles_df.to_string('eles.txt', header=False)
    
    ## Prepare mater.txt data ##
    youngs = []
    poissons = []
    youngs.append(youngs_modulus)
    poissons.append(poisson_ratio)
        
    mater_df["Youngs Modulus"] = youngs
    mater_df["Poisson Ratio"] = poissons
    
    mater_df.to_string('mater.txt', index=False, header=False)
        
    ## Prepare loads.txt data ##
    xload = []
    yload = []
    xloadnode = []
    yloadnode = []
    load_index = []
    
    # Find the max y node value
    w = 6.0
    wc = parameters[0]
    hc = parameters[1]
    tc = parameters[2]
    
    wl = (w - wc)/2
    h = hc + tc
    
    # Find the number of points at the max height
    hcount = nodey.count(h)
    avg_elsize = w/(hcount-1)
    
    # Apply gauge pressure to the top and sides of the vacuum chamber
    for i,pt in enumerate(pts):
        # top of vacuum chamber
        if pt.p[1] == hc and pt.p[0] >= wl and pt.p[0] <= w-wl:
            load_index.append(i)
            xload.append(0.0)
            if pt.p[0] == wl or pt.p[0] == w-wl:
                yload.append(pressure*avg_elsize/2)
                xloadnode.append(pt.p[0])
                yloadnode.append(pt.p[1])
            else:
                yload.append(pressure*avg_elsize)
                xloadnode.append(pt.p[0])
                yloadnode.append(pt.p[1])
        
        if pt.p[0] == wl and pt.p[1] >= 0.0 and pt.p[1] <= hc:
            load_index.append(i)
            yload.append(0.0)
            if pt.p[1] == 0.0 or pt.p[1] == hc:
                xload.append(pressure*avg_elsize/2)
                xloadnode.append(pt.p[0])
                yloadnode.append(pt.p[1])
            else:
                xload.append(pressure*avg_elsize)
                xloadnode.append(pt.p[0])
                yloadnode.append(pt.p[1])
        
        if pt.p[0] == w-wl and pt.p[1] >= 0.0 and pt.p[1] <= hc:
            load_index.append(i)
            yload.append(0.0)
            if pt.p[1] == 0.0 or pt.p[1] == hc:
                xload.append(pressure*avg_elsize/2)
                xloadnode.append(pt.p[0])
                yloadnode.append(pt.p[1])
            else:
                xload.append(pressure*avg_elsize)
                xloadnode.append(pt.p[0])
                yloadnode.append(pt.p[1])

    loads_df = pd.DataFrame({"X Load Magnitude":xload, "Y Load Magnitude":yload},index=load_index)    
    loads_df.to_string('loads.txt', header=False)
    
    # # Plot to verify which nodes are being used as boundary conditions (blue
    # # are regular nodes, red are fixtures, green are loads)
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal')
    # plt.scatter(nodex,nodey,c='b')
    # plt.scatter(xboundnode,yboundnode,c='r')
    # plt.scatter(xloadnode,yloadnode,c='g')
    
    
    top_pts_i = []
    base_pts_i = []
    for i,pt in enumerate(pts):
        # Catalog chamber top points
        if pt.p[1] == hc:
            if pt.p[0] >= (w-wc)/2:
                if pt.p[0] <= ((w-wc)/2 + wc):
                    top_pts_i.append(i)
            
        if pt.p[1] == 0 and pt.p[0] < (w-wc)/2 or pt.p[0] > ((w-wc)/2):
            base_pts_i.append(i)
            
    return top_pts_i, base_pts_i


def max_top_displacement(UC, top_pts_i):
    max_displacement = 0
    for i in range(np.shape(UC)[0]):
        if i in top_pts_i:
            if UC[i][1] < max_displacement:
                max_displacement = UC[i][1]
                
    return max_displacement


def max_y_stress(S_nodes, base_pts_i):
    max_y_stress = 0
    base_stresses = []
    for i in range(np.shape(S_nodes)[0]):
        if i in base_pts_i:
            base_stresses.append(S_nodes[i][1])
            if S_nodes[i][1] < max_y_stress:
                max_y_stress = S_nodes[i][1]
                
    # avg_stress = np.median(base_stresses)
    avg_stress = np.mean(base_stresses)
    
    return [max_y_stress, base_stresses, avg_stress]
    

def get_stress_displace(parameters, plot_contours):
    w = 6.0
    wc = parameters[0]
    hc = parameters[1]
    tc = parameters[2]
    
    elsize = 0.05
    
    pressure = -12.28
    youngs_modulus = 200.0
    poisson_ratio = 0.47
    
    mesh = rectangular_single_legs(w, wc, hc, tc, elsize)
    pts = list(mesh.Points())
    
    top_pts_i, base_pts_i = generate_solidspy_files_rectangular_single_legs(mesh, parameters, pressure, youngs_modulus, poisson_ratio)
    
    # Modify base_pts_i to only include mesh points that are within distance of opening
    thresh_dist = 0.25
    chamber_low = w/2 - wc/2
    chamber_high = w/2 + wc/2
    checkpos_low = chamber_low - thresh_dist
    checkpos_high = chamber_high + thresh_dist
    base_pts_i = [i for i in base_pts_i if (pts[i].p[0]>checkpos_low and pts[i].p[0]<chamber_low) or (pts[i].p[0]>chamber_high and pts[i].p[0]<checkpos_high)]
    
    directory = os.getcwd() + "\\"
    
    try:
        UC, E_nodes, S_nodes = solids_GUI(plot_contours=plot_contours, compute_strains=True, folder=directory)

        ### Metrics ###
        # Max displacement of the top surface of the chamber
        max_displace = max_top_displacement(UC, top_pts_i)
        # max_displace = max_displace / hc # Trying scaling displacement by the chamber height to make different designs more equivalent
        
        # Max sigma yy stress along the mold surface
        max_stress, base_stresses, avg_stress = max_y_stress(S_nodes, base_pts_i)
    except:
        max_displace = 0
        max_stress = 0
        avg_stress = 0
    
    return avg_stress, np.abs(max_displace)


###############################################################################
########################## OPTIMIZATION FUNCTIONS #############################
###############################################################################

def generate_design(bounds):
    parameters = []
    for i in range(len(bounds)):
        parameters.append(random.uniform(bounds[i][0], bounds[i][1]))
    return parameters


def generate_initial_population(bounds, npop):
    pop = []
    for i in range(npop):
        pop.append(generate_design(bounds))
    return pop


# Tournament selection
def selection(pop, scores, k=3):
    # First random selection
    selection_ix = np.random.randint(len(pop))
    for ix in np.random.randint(0, len(pop), k-1):
        # Check if better (e.g. perform a tournament)
        if scores[ix] < scores[selection_ix]:
            selection_ix = ix
    return pop[selection_ix]


# Crossover two parents to create two children
def crossover(p1, p2, r_cross, bounds):
    # Children are copies of parents by default
    c1, c2 = p1.copy(), p2.copy()
    
    # Check for recombination
    if np.random.rand() < r_cross:
        # select crossover point that is not on the end of the parameter list
        pt = np.random.randint(1, len(p1))
        # perform crossover
        c1 = p1[:pt] + p2[pt:]
        c2 = p2[:pt] + p1[pt:]
    return [c1, c2]


# Mutation
def mutation(individual, r_mut, bounds, max_mutate_pct):        
    for i in range(len(individual)):
        # Check for a mutation
        if np.random.rand() < r_mut:
            boundrange = np.abs(bounds[i][1] - bounds[i][0])
            max_perturb = max_mutate_pct*boundrange
            
            while True:
                perturb = np.random.uniform(-max_perturb, max_perturb)
                newval = individual[i] + perturb
                if newval >= bounds[i][0] and newval <= bounds[i][1]:
                    individual[i] = newval
                    break
            
            # individual[i] = np.random.uniform(bounds[i][0], bounds[i][1])
            

# def maximin_fitness(pop):
#     # Take in a design and calculate the maximin score against the population of designs
#     npop = len(pop)
    
#     print("\nCalculating stress and displacement for {} designs\n".format(npop))
#     fitnesses = []
#     for i in range(npop):
#         print("CALCULATING STRESS AND DISPLACEMENT FOR DESIGN {}/{}".format(i,npop-1))
#         avg_stress, max_displace = get_stress_displace(pop[i], False)
#         fitnesses.append([avg_stress, max_displace])
        
#     numobj = len(fitnesses[0])
    
#     # Scale fitness values to be the same scale (0 to 1)
#     stresses = [fitness[0] for fitness in fitnesses]
#     displacements = [fitness[1] for fitness in fitnesses]
    
    
#     norm_stresses = [(stress - np.min(stresses)) / (np.max(stresses) - np.min(stresses)) for stress in stresses]
#     norm_displace = [(displace - np.min(displacements)) / (np.max(displacements) - np.min(displacements)) for displace in displacements]
    
#     normalize = True
    
#     if normalize is True:
#         normfitnesses = []
#         for i in range(npop):
#             normfitnesses.append([norm_stresses[i], norm_displace[i]])
#     else:
#         normfitnesses = fitnesses
    
#     # Calculate the maximin score for each particle
#     print("\nCalculating maximin scores")
#     scores = []
#     for i in range(npop):
#         popfitnesses = []
#         designfit = normfitnesses[i]
#         for j,fitness in enumerate(normfitnesses):
#             if j != i:
#                 popfitnesses.append(fitness)
                
#         minvals = np.zeros(((npop-1),numobj))
#         for j in range(npop-1):
#             for k in range(numobj):
#                 minvals[j,k] = designfit[k] - popfitnesses[j][k]
                
#         mins = np.zeros(npop-1)
#         for j in range(npop-1):
#             mins[j] = np.min(minvals[j,:])
        
#         score = np.max(mins)
#         # pop[i].fitness_particle_position = score        
        
#         scores.append(score)
        
#     scores = scores[0:npop]
    
#     # Return scores, which is the list of the maximin scores by design index
#     return scores


def maximin_fitness(pop, prevfitnesses):
    # Take in a design and calculate the maximin score against the population of designs
    npop = len(pop)
    
    print("\nCalculating stress and displacement for {} designs\n".format(npop))
    fitnesses = []
    for i in range(npop):
        print("CALCULATING STRESS AND DISPLACEMENT FOR DESIGN {}/{}".format(i,npop-1))
        avg_stress, max_displace = get_stress_displace(pop[i], False)
        fitnesses.append([avg_stress, max_displace])
        
    numobj = len(fitnesses[0])
    
    allfitnesses = fitnesses + prevfitnesses
    
    # Get maximum and minimum stress and displacements from all generations for scaling
    allstresses = [fitness[0] for fitness in allfitnesses]
    alldisplace = [fitness[1] for fitness in allfitnesses]
    minstress = np.min(allstresses)
    maxstress = np.max(allstresses)
    mindisplace = np.min(alldisplace)
    maxdisplace = np.max(alldisplace)
    
    # Scale fitness values to be the same scale (0 to 1)
    stresses = [fitness[0] for fitness in fitnesses]
    displacements = [fitness[1] for fitness in fitnesses]
    norm_stresses = [(stress - minstress) / (maxstress - minstress) for stress in stresses]
    norm_displace = [(displace - mindisplace) / (maxdisplace - mindisplace) for displace in displacements]
    # norm_stresses = [(stress - np.min(stresses)) / (np.max(stresses) - np.min(stresses)) for stress in stresses]
    # norm_displace = [(displace - np.min(displacements)) / (np.max(displacements) - np.min(displacements)) for displace in displacements]
    
    normalize = True
    
    if normalize is True:
        normfitnesses = []
        for i in range(npop):
            normfitnesses.append([norm_stresses[i], norm_displace[i]])
    else:
        normfitnesses = fitnesses
    
    # Calculate the maximin score for each particle
    print("\nCalculating maximin scores")
    scores = []
    for i in range(npop):
        popfitnesses = []
        designfit = normfitnesses[i]
        for j,fitness in enumerate(normfitnesses):
            if j != i:
                popfitnesses.append(fitness)
                
        minvals = np.zeros(((npop-1),numobj))
        for j in range(npop-1):
            for k in range(numobj):
                minvals[j,k] = designfit[k] - popfitnesses[j][k]
                
        mins = np.zeros(npop-1)
        for j in range(npop-1):
            mins[j] = np.min(minvals[j,:])
        
        score = np.max(mins)
        # pop[i].fitness_particle_position = score        
        
        scores.append(score)
        
    scores = scores[0:npop]
    
    # Return scores, which is the list of the maximin scores by design index
    return scores, allfitnesses


def genetic_algorithm(n_iter, npop, bounds, r_cross, r_mut, max_mutate_pct):
    # Visualization
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot()
    # fig.show()
    
    # Initial population
    pop = generate_initial_population(bounds, npop)
    
    # Keep track of best solution
    best, best_eval = 0, np.inf
    
    best_evals = []
    prevfitnesses = []
    
    wcvals = []
    hcvals = []
    tcvals = []
    rgbvals_list = []
    generations = []
    wcpareto = []
    hcpareto = []
    tcpareto = []
    generationspareto = []
    # Enumerate generations
    for gen in range(n_iter):
        redval = gen/(n_iter)
        blueval = 1.0-redval
        rgbvals = (redval, 0, blueval)
        
        
        print("\nGeneration {}/{}".format(gen,n_iter-1))
        # Evaluate all candidates in the population
        # scores = maximin_fitness(pop)
        scores, prevfitnesses = maximin_fitness(pop, prevfitnesses)
        
        
        # Check for new best solution
        for i in range(npop):
            if scores[i] < best_eval:
                best, best_eval = pop[i], scores[i]
                # print("\nGeneration {}: new best {} = {}".format(gen, pop[i], scores[i]))
            if scores[i] < 0:
                wcpareto.append(pop[i][0])
                hcpareto.append(pop[i][1])
                tcpareto.append(pop[i][2])
                generationspareto.append(gen)
        best_evals.append(best_eval)
        
        for i in range(npop):
            # rgbvals_list.append(redval)
            # If the design is a Pareto design for the current generation,
            # if scores[i] < 0:
            #     generations.append(n_iter*2)
            # else:
            #     generations.append(gen)
            generations.append(gen)
            wcvals.append(pop[i][0])
            hcvals.append(pop[i][1])
            tcvals.append(pop[i][2])
            
        # Visualization
        # ax.plot(A, color='r')
        
        scores_array = np.asarray(scores)
        nkeep = int(len(scores)/2)
        best_half_indices = scores_array.argsort()[:nkeep]
        
        # wc = [design[0] for design in pop]
        # h = [design[1] for design in pop]
        
        # ax.scatter(wc,h,color=rgbvals)
        # fig.canvas.draw()
        # # ax.set_xlim(left=max(0, i - iterations), right=i + 3)
        # ax.set_xlabel("wc")
        # ax.set_ylabel("hc")
        
        # select parents
        selected = [selection(pop, scores) for x in range(npop)]
        
        # Create the next generation
        children = []
        for i in range(0, npop, 2):
            # Get selected parents in pairs
            p1, p2 = selected[i], selected[i+1]
            
            c1, c2 = crossover(p1, p2, r_cross, bounds)
            
            mutation(c1, r_mut, bounds, max_mutate_pct)
            mutation(c2, r_mut, bounds, max_mutate_pct)
            
            children.append(c1)
            children.append(c2)
        
        # for ind in best_half_indices:
        #     children.append(pop[ind])
        #     child = pop[ind].copy()
        #     mutation(child, r_mut, bounds, max_mutate_pct)
        #     # mut_allow = max_mutate_pct - (gen/n_iter)*max_mutate_pct
        #     # mutation(child, r_mut, bounds, mut_allow)
        #     children.append(child) 
        
        # replace population if not the last iteration
        if gen != n_iter-1:
            pop = children
       
    # Make a dataframe of the parameters and colors for plotting
    data = {"wc":wcvals, "hc":hcvals, "tc":tcvals, "Generation":generations}
    # data = {"wc":wcvals, "hc":hcvals, "tc":tcvals, "RGB":rgbvals_list}
    df_param = pd.DataFrame(data)
    
    data_pareto = {"wc":wcpareto, "hc":hcpareto, "tc":tcpareto, "Generation":generationspareto}
    df_pareto = pd.DataFrame(data_pareto)
    
    fig = px.scatter_3d(df_param, x="wc", y="hc", z="tc", color="Generation", title="All Designs")
    # fig = px.scatter_3d(df_param, x="wc", y="hc", z="tc", color="RGB")
    fig.show()
    
    fig2 = px.scatter_3d(df_pareto, x="wc", y="hc", z="tc", color="Generation", title="Pareto Designs")
    fig2.show()
    # plt.show()
        
    return [best, best_eval], best_evals, pop



if __name__ == "__main__":
    n_iter = 50
    npop = 80
    r_cross = 0.95
    r_mut = 0.3
    max_mutate_pct = 1.0
    
    wc_lb = 0.125
    hc_lb = 0.125
    tc_lb = 0.125
    wc_ub = 2.5
    hc_ub = 2.5
    tc_ub = 2.5
    bounds = [(wc_lb, wc_ub), (hc_lb, hc_ub), (tc_lb, tc_ub)]  # upper and lower bounds of variables
    
    [best, score], best_evals, pop = genetic_algorithm(n_iter, npop, bounds, r_cross, r_mut, max_mutate_pct)