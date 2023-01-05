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
import datetime as dt

def half_ellipse_chamber(w, rc, h, maxmesh):
    """
    
    Parameters
    ----------
    w : float
        Overall width of seal profile.
    rc : float
        Radius of vacuum chamber.
    h : float
        Overall height of seal profile.

    Returns
    -------
    None.

    """
    wc = 2*rc
    hc = rc
        
    # Construct the points that define the seal profile
    coords = [[0,0],[(w-wc)/2,0],[(w-wc)/2,hc],[w/2,hc],[(w-wc)/2+wc,hc],[(w-wc)/2+wc,0],[w,0],[w,h],[0,h]]
    mesh = mesh_from_coords(coords,maxmesh)
    
    return mesh


def mesh_from_coords(coords,maxmesh):
    geo = geom2d.SplineGeometry()
    pts = [geo.AppendPoint(coord[0],coord[1]) for coord in coords]
    
    geo.Append(["line",pts[0],pts[1]])
    geo.Append(["spline3",pts[1],pts[2],pts[3]])
    geo.Append(["spline3",pts[3],pts[4],pts[5]])
    geo.Append(["line",pts[5],pts[6]])
    geo.Append(["line",pts[6],pts[7]])
    geo.Append(["line",pts[7],pts[8]])
    geo.Append(["line",pts[8],pts[0]])
    
    mesh = geo.GenerateMesh(maxh=maxmesh)
    
    return mesh
    

def get_semicircle_boundary(w,rc,pts):
    refpt = (w/2, 0.0)
    dists = [np.sqrt((pt.p[0]-refpt[0])**2 + (pt.p[1]-refpt[1])**2) for pt in pts]
    # round_dists = [np.around(dist,2) for dist in dists]
    
    chamber_pts = []
    chamber_idx = []
    for i,dist in enumerate(dists):
        if dist - rc < 0.0001:
            chamber_pts.append(pts[i])
            chamber_idx.append(i)
            
    return chamber_pts, chamber_idx, dists
    

def generate_solidspy_files_semicircle(mesh, parameters, pressure, youngs_modulus, poisson_ratio):
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
            # xbound.append(-1)    # Fully fixed feet
            xbound.append(0)    #  Roller boundary condition
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
    rc = parameters[1]
    h = parameters[2]
    wc = 2*rc
    hc = rc
    
    refpt = (w/2,0.0)
    
    chamber_pts, chamber_idx, dists = get_semicircle_boundary(w,rc,pts)
    
    # Find the number of points at the max height
    hcount = nodey.count(h)
    avg_elsize = w/(hcount-1)
    
    # Apply gauge pressure to the top and sides of the vacuum chamber
    for i,pt in enumerate(chamber_pts):
        angle = np.arccos((pt.p[0]-refpt[0])/dists[chamber_idx[i]])
        forcemag = pressure*avg_elsize
        xload.append(forcemag*np.cos(angle))
        yload.append(forcemag*np.sin(angle))
        xloadnode.append(pt.p[0])
        yloadnode.append(pt.p[1])

    loads_df = pd.DataFrame({"X Load Magnitude":xload, "Y Load Magnitude":yload},index=chamber_idx)    
    loads_df.to_string('loads.txt', header=False)
    
    # # Plot to verify which nodes are being used as boundary conditions (blue
    # # are regular nodes, red are fixtures, green are loads)
    # fig = plt.figure(dpi=300)
    # ax = fig.add_subplot(111)
    # ax.set_aspect('equal')
    # plt.scatter(nodex,nodey,c='b')
    # plt.scatter(xboundnode,yboundnode,c='r')
    # plt.scatter(xloadnode,yloadnode,c='g')
    
    
    top_pts_i = chamber_idx
    base_pts_i = []
    for i,pt in enumerate(pts):
        # Catalog chamber top points
        # if pt.p[1] == hl and pt.p[0] >= 0.0 and pt.p[0] <= w-wl:
        #     top_pts_i.append(i)
            
        if pt.p[1] == 0.0:
            base_pts_i.append(i)
            
    return top_pts_i, base_pts_i


def max_top_displacement(UC, top_pts_i):
    max_displacement = 0
    for i in range(np.shape(UC)[0]):
        if i in top_pts_i:
            if UC[i][1] < max_displacement:
                max_displacement = UC[i][1]
                
    # if np.abs(max_displacement) > h/2:
    #     max_displacement = 0
                
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
    rc = parameters[1]
    h = parameters[2]
    
    elsize = w*0.01
    
    # timeout = 120.0
    
    pressure = -12.28
    youngs_modulus = 200.0
    poisson_ratio = 0.47
    
    mesh = half_ellipse_chamber(w, rc, h, elsize)
    
    top_pts_i, base_pts_i = generate_solidspy_files_semicircle(mesh, parameters, pressure, youngs_modulus, poisson_ratio)
    
    directory = os.getcwd() + "\\"
    
    try:
        UC, E_nodes, S_nodes = solids_GUI(plot_contours=plot_contours, compute_strains=True, folder=directory)
        
        ### Metrics ###
        # Max displacement of the top surface of the chamber
        max_displace = max_top_displacement(UC, top_pts_i)
        
        # Max sigma yy stress along the mold surface
        max_stress, base_stresses = max_y_stress(S_nodes, base_pts_i)
        
    except:
        max_stress = 0
        max_displace = 0
        print("SOLVER ERROR")
    
    return max_stress, max_displace


def obj_fun(parameters):
    print("\nParameters this iteration:\t{}".format(parameters))
    max_stress, _ = get_stress_displace(parameters, False)
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
        raise ValueError("w out of bounds")
    elif xs[1] > xub[1]:
        raise ValueError("rc out of bounds")
    elif xs[2] > xub[2]:
        raise ValueError("h out of bounds")
    # elif xs[3] > xub[3]:
    #     raise ValueError("yp0 out of bounds")
    elif xs[0] < xlb[0]:
        raise ValueError("w out of bounds")
    elif xs[1] < xlb[1]:
        raise ValueError("rc out of bounds")
    elif xs[2] < xlb[2]:
        raise ValueError("h out of bounds")
    # elif xs[3] < xlb[3]:
    #     raise ValueError("yp0 out of bounds")
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
    delta = 0.5               # Max perturbation
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
                wp = np.random.uniform(-delta, delta)
                rcp = np.random.uniform(-delta, delta)
                hp = np.random.uniform(-delta, delta)
                perturb = np.array([wp, rcp, hp])
                xp = xc + perturb
                
                elsize = xp[0]*0.01
                
                # Ensure perturbed design is within bounds
                if xp[0] > xub[0]:
                    continue
                if xp[1] > xub[1]:
                    continue
                if xp[2] > xub[2]:
                    continue
                # elif xp[3] > xub[3]:
                #     continue
                if xp[0] < xlb[0]:
                    continue
                if xp[1] < xlb[1]:
                    continue
                if xp[2] < xlb[2]:
                    continue
                # elif xp[3] < xlb[3]:
                #     continue
                # Ensure perturbed design does not violate constraints
                if xp[0] < 2*xp[1]:
                    continue
                if xp[1] > xp[2]-2*elsize:
                    continue
                if xp[0]-xp[1] < 2.0:
                    continue

            # print(xp)
                print("Perturbed design successfully generated: {}".format(xp))
                # Get the objective value at the perturbed point
                fp, displace = get_stress_displace(xp, False)
                if np.abs(displace) > np.abs(xp[1]/2):
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
    w0 = 5.0
    rc0 = 1.0
    h0 = 1.5
    x0 = [w0, rc0, h0]
    
    w_lb = 3.0
    rc_lb = 0.25
    h_lb = 0.25
    xlb = [w_lb, rc_lb, h_lb]
    
    w_ub = 10.0
    rc_ub = 8.0
    h_ub = 2.0
    xub = [w_ub, rc_ub, h_ub]
    
    start = dt.datetime.now()
    
    perturbations, objvals, xsearch = SimAnneal(x0, xub, xlb)

    w_end = xsearch[-1][0]
    rc_end = xsearch[-1][1]
    h_end = xsearch[-1][2]

    best_ind = objvals.index(min(objvals))
    w_best = xsearch[best_ind][0]
    rc_best = xsearch[best_ind][1]
    h_best = xsearch[best_ind][2]

    # Plot the cooling curve
    # fig1 = plt.figure(1, figsize=(12,8))
    plt.plot(perturbations, objvals)
    plt.xlabel("Cycles")
    plt.ylabel("Objective")
    plt.title("Cooling Curve - Simple Bag Seal Profile")
    end_annotation = "Final design:\nObj: {}\nw: {}\nrc: {}\nh: {}".format(objvals[-1], w_end, rc_end, h_end)
    plt.annotate(end_annotation, (perturbations[-1], objvals[-1]), (0.8*perturbations[-1], 0.8*np.max(objvals)), arrowprops=dict(facecolor='black', shrink = 0.01, width=0.5))
    best_annotation = "Best design:\nObj: {}\nw: {}\nrc: {}\nh: {}".format(objvals[best_ind], w_best, rc_best, h_best)
    mid_ind = int(len(perturbations)/3)
    plt.annotate(best_annotation, (perturbations[best_ind], objvals[best_ind]), (0, 0.25*(np.max(objvals)-np.min(objvals))+np.min(objvals)), arrowprops=dict(facecolor='black', shrink = 0.01, width=0.5))
    plt.show()
    
    print("\nOptimized results:")
    print("Best design:\nObj: {}\nw: {}\nrc: {}\nh: {}".format(objvals[best_ind], w_best, rc_best, h_best))
    
    print("\nPlotting FEA results...\n")
    max_stress, max_displace = get_stress_displace([w_best, rc_best, h_best], True)
    print("\nMax stress:\t{}".format(max_stress))
    print("Max displacement:\t{}".format(max_displace))
    
    end = dt.datetime.now()
    
    duration = end - start
    print("\nRun time:\t{}".format(duration))