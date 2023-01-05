# -*- coding: utf-8 -*-
"""
Created on Mon Sep 26 10:02:07 2022

@author: Ryan.Larson
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from solidspy import solids_GUI
# from shapely.geometry import Polygon, Point
import netgen.geom2d as geom2d
import os
from scipy.optimize import minimize

###############################################################################
############################ SIMULATION FUNCTIONS #############################
###############################################################################

def rectangular_single_legs(w, wl, h, hl, maxmesh):
    """
    

    Parameters
    ----------
    w : float
        Overall width of seal profile.
    wl : float
        Width of single seal leg.
    h : float
        Overall height of seal profile.
    hl : float
        Height of single seal leg.

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
    coords = [[0,0],[wl,0],[wl,hl],[w-wl,hl],[w-wl,0],[w,0],[w,h],[0,h]]
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
    w = parameters[0]
    wchamber = parameters[1]
    wl = (w-wchamber)/2
    h = parameters[2]
    hl = parameters[3]
    
    # Find the number of points at the max height
    hcount = nodey.count(h)
    avg_elsize = w/(hcount-1)
    
    # Apply gauge pressure to the top and sides of the vacuum chamber
    for i,pt in enumerate(pts):
        # top of vacuum chamber
        if pt.p[1] == hl and pt.p[0] >= wl and pt.p[0] <= w-wl:
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
        
        if pt.p[0] == wl and pt.p[1] >= 0.0 and pt.p[1] <= hl:
            load_index.append(i)
            yload.append(0.0)
            if pt.p[1] == 0.0 or pt.p[1] == hl:
                xload.append(pressure*avg_elsize/2)
                xloadnode.append(pt.p[0])
                yloadnode.append(pt.p[1])
            else:
                xload.append(pressure*avg_elsize)
                xloadnode.append(pt.p[0])
                yloadnode.append(pt.p[1])
        
        if pt.p[0] == w-wl and pt.p[1] >= 0.0 and pt.p[1] <= hl:
            load_index.append(i)
            yload.append(0.0)
            if pt.p[1] == 0.0 or pt.p[1] == hl:
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
        if pt.p[1] == hl and pt.p[0] >= 0.0 and pt.p[0] <= w-wl:
            top_pts_i.append(i)
            
        if pt.p[1] == 0.0:
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
    
    return [max_y_stress, base_stresses]
    

def get_stress_displace(parameters, plot_contours):
    w = parameters[0]
    wchamber = parameters[1]
    wl = (w-wchamber)/2
    h = parameters[2]
    hl = parameters[3]
    
    elsize = 0.05
    
    pressure = -12.28
    youngs_modulus = 200.0
    poisson_ratio = 0.47
    
    mesh = rectangular_single_legs(w, wl, h, hl, elsize)
    # while True:
    #     try:
    #         mesh = rectangular_single_legs(w, wl, h, hl, elsize)
    #     except:
    #         print("Impossible design generated")
    #     else:
    #         break
    
    top_pts_i, base_pts_i = generate_solidspy_files_rectangular_single_legs(mesh, parameters, pressure, youngs_modulus, poisson_ratio)
    
    directory = os.getcwd() + "\\"
    
    UC, E_nodes, S_nodes = solids_GUI(plot_contours=plot_contours, compute_strains=True, folder=directory)
    
    ### Metrics ###
    # Max displacement of the top surface of the chamber
    max_displace = max_top_displacement(UC, top_pts_i)
    
    # Max sigma yy stress along the mold surface
    max_stress, base_stresses = max_y_stress(S_nodes, base_pts_i)
    
    return max_stress, max_displace


def obj_fun(parameters):
    max_stress, _ = get_stress_displace(parameters, False)
    print("\nParameters this iteration:\t{}".format(parameters))
    print("Max stress this iteration:\t{}\n".format(max_stress))
    return max_stress

def constraint1(parameters):
    _, max_displace = get_stress_displace(parameters, False)
    return np.abs(max_displace) - np.abs(parameters[3]/2)

def Boltzmann(dE, dEavg, T):
    P = np.exp(-dE/(dEavg*T))
    return P

def SimAnneal(x0,xub,xlb):
    # Starting design
    xs = np.array(x0)
    xub = np.array(xub)
    xlb = np.array(xlb)
    if xs[0] > xub[0]:
        raise ValueError("ds0 out of bounds")
    elif xs[1] > xub[1]:
        raise ValueError("S0 out of bounds")
    elif xs[2] > xub[2]:
        raise ValueError("xp0 out of bounds")
    elif xs[3] > xub[3]:
        raise ValueError("yp0 out of bounds")
    elif xs[0] < xlb[0]:
        raise ValueError("ds0 out of bounds")
    elif xs[1] < xlb[1]:
        raise ValueError("S0 out of bounds")
    elif xs[2] < xlb[2]:
        raise ValueError("xp0 out of bounds")
    elif xs[3] < xlb[3]:
        raise ValueError("yp0 out of bounds")
    fs = obj_fun(xs)
    xsearch = [xs]

    # Select Ps, Pf, N, and calculate Ts, Tf, and F
    Ps = 0.3                # Probability of acceptance at start
    Pf = 0.00001             # Probability of acceptance at finish
    N = 50                # Number of cycles

    Ts = -1/np.log(Ps)      # Temperature at start
    Tf = -1/np.log(Pf)      # Temperature at finish
    F = (Tf/Ts)**(1/(N-1))  # Temperature reduction factor each cycle

    # Perturbation information
    delta = 2.0               # Max perturbation
    n = 2                   # Starting number of perturbations per cycle

    # Holding variables
    dE = 0.0
    dEavg = 0.0
    perturbations = list(range(N))
    objvals = [0] * N

    # Set starting values
    xc = xs
    fc = fs
    T = Ts

    # Step through the cycles
    for i, perturb in enumerate(perturbations):
        print("\n\n####################")
        print("ITERATION:\t{}".format(i))
        print("####################")
        
        # Add the current objective value to the objective vector for plotting
        objvals[i] = obj_fun(xc)

        # Step through the perturbations
        for j in range(n):
            print("\nPerturbation:\t{}".format(j))
            # Perturb xc by some random value within delta. If any perturbed
            # value falls outside the specified bounds, retry the perturbation.
            while True:
                dsp = np.random.uniform(-delta, delta)
                Sp = np.random.uniform(-delta, delta)
                xpp = np.random.uniform(-delta, delta)
                ypp = np.random.uniform(-delta, delta)
                perturb = np.array([dsp, Sp, xpp, ypp])
                xp = xc + perturb
                
                # Ensure perturbed design is within bounds
                if xp[0] > xub[0]:
                    continue
                elif xp[1] > xub[1]:
                    continue
                elif xp[2] > xub[2]:
                    continue
                elif xp[3] > xub[3]:
                    continue
                elif xp[0] < xlb[0]:
                    continue
                elif xp[1] < xlb[1]:
                    continue
                elif xp[2] < xlb[2]:
                    continue
                elif xp[3] < xlb[3]:
                    continue
                # Ensure perturbed design does not violate constraints
                elif xp[0] < xp[1]:
                    continue
                elif xp[2] < xp[3]:
                    continue
                elif xp[0]-xp[1] < 2.0:
                    continue
                elif xp[2]-xp[3] < 0.2:
                    continue

            # print(xp)

                # Get the objective value at the perturbed point
                fp, displace = get_stress_displace(xp, False)
                if np.abs(displace) > xp[3]/2:
                    "\nExcessive deflection detected"
                    continue
                # fp = obj_fun(xp)
                else:
                    break

            # Calculate values for Boltzmann function in case they're needed
            dE = np.abs(fp - fc)
            if i == 0 and j == 0:
                dEavg = dE
            else:
                dEavg = (dEavg + dE)/2

            P = Boltzmann(dE, dEavg, T)

            # Check if the new design is better than the old design
            if fp < fc:
                xc = xp     # Accept as current design if better
                fc = obj_fun(xc)
            else:
                # If the new design is worse, generate a random number and
                # compare to the Boltzmann probability. If the random number is
                # lower than the Boltzmann probability, accept the worse design
                # as the current design
                randnum = np.random.uniform(0,1)

                if randnum < P:
                    xc = xp
                    fc = obj_fun(xc)

        # Decrease the temperature by factor F
        T = F*T

        # Increase the number of perturbations every few cycles
        if (i % 3) == 1:
            n += 1

        # Save the new search position at the end of each cycle
        xsearch.append(xc)

    return perturbations, objvals, xsearch

if __name__ == "__main__":
    # Initial values
    w0 = 8.0
    wchamber0 = 1.0
    # wl0 = (w0-wchamber0)/2
    h0 = 1.5
    hl0 = 0.25
    x0 = [w0, wchamber0, h0, hl0]
    
    w_lb = 3.0
    wchamber_lb = 0.25
    h_lb = 0.25
    hl_lb = 0.1
    xlb = [w_lb, wchamber_lb, h_lb, hl_lb]
    
    w_ub = 10.0
    wchamber_ub = 8.0
    h_ub = 2.0
    hl_ub = 1.5
    xub = [w_ub, wchamber_ub, h_ub, hl_ub]
    
    
    perturbations, objvals, xsearch = SimAnneal(x0, xub, xlb)

    w_end = xsearch[-1][0]
    wchamber_end = xsearch[-1][1]
    h_end = xsearch[-1][2]
    hl_end = xsearch[-1][3]

    best_ind = objvals.index(min(objvals))
    w_best = xsearch[best_ind][0]
    wchamber_best = xsearch[best_ind][1]
    h_best = xsearch[best_ind][2]
    hl_best = xsearch[best_ind][3]

    # Plot the cooling curve
    fig1 = plt.figure(1, figsize=(12,8))
    plt.plot(perturbations, objvals)
    plt.xlabel("Cycles")
    plt.ylabel("Objective")
    plt.title("Cooling Curve - Simple Bag Seal Profile")
    end_annotation = "Final design:\nObj: {}\nw: {}\nwchamber: {}\nh: {}\nhl: {}".format(objvals[-1],w_end, wchamber_end, h_end, hl_end)
    plt.annotate(end_annotation, (perturbations[-1], objvals[-1]), (0.8*perturbations[-1], 0.8*np.max(objvals)), arrowprops=dict(facecolor='black', shrink = 0.01, width=0.5))
    best_annotation = "Best design:\nObj: {}\nw: {}\nwchamber: {}\nh: {}\nhl: {}".format(objvals[best_ind],w_best, wchamber_best, h_best, hl_best)
    mid_ind = int(len(perturbations)/3)
    plt.annotate(best_annotation, (perturbations[best_ind], objvals[best_ind]), (0, 0.25*(np.max(objvals)-np.min(objvals))+np.min(objvals)), arrowprops=dict(facecolor='black', shrink = 0.01, width=0.5))
    plt.show()
    
    
    # bounds = ((w_lb,w_ub), (wchamber_lb,wchamber_ub), (h_lb,h_ub), (hl_lb,hl_ub))
    
    
    # res = minimize(obj_fun,
    #                x0,
    #                constraints = (
    #                    {'type': 'ineq', 'fun': constraint1},
    #                    {'type': 'ineq', 'fun': lambda x: x[0] - x[1]},
    #                    {'type': 'ineq', 'fun': lambda x: x[2] - x[3]}
    #                        ),
    #                bounds = bounds,
    #                method='SLSQP')
    
    print("\nOptimized results:")
    print("Best design:\nObj: {}\nds: {}\nS: {}\nxp: {}\nyp: {}".format(objvals[best_ind], w_best, wchamber_best, h_best, hl_best))
    
    print("\nPlotting FEA results...\n")
    max_stress, max_displace = get_stress_displace([w_best, wchamber_best, h_best, hl_best], True)
    print("\nMax stress:\t{}".format(max_stress))
    print("Max displacement:\t{}".format(max_displace))