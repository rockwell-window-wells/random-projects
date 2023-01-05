# -*- coding: utf-8 -*-
"""
Multi-objective particle swarm optimization of silicone seal using flat bag
over gap design.

Created on Thu Oct 20 08:42:45 2022

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
from datetime import datetime
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import random
import math


def rectangular_flat(w, wc, h, maxmesh):
    """
    Parameters
    ----------
    w : float
        Overall width of seal profile.
    wc : float
        Width of vacuum chamber
    h : float
        Overall height of silicone bag.

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
    coords = [[0,0],[w,0],[w,h],[0,h]]
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
    
    # Find the max y node value
    w = 5.0
    wc = parameters[0]
    h = parameters[1]
    
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
        elif i !=0 and pt[1] == 0 and pt[0] < (w-wc)/2:
            # xbound.append(-1)    # Fully fixed feet
            xbound.append(0)    #  Roller boundary condition
            ybound.append(-1)
            xboundnode.append(pt.p[0])
            yboundnode.append(pt.p[1])
        elif i !=0 and pt[1] == 0 and pt[0] > ((w-wc)/2 + wc):
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
    
    
    # Find the number of points at the max height
    hcount = nodey.count(h)
    avg_elsize = w/(hcount-1)
    
    # Apply gauge pressure to the top and sides of the vacuum chamber
    for i,pt in enumerate(pts):
        # top of vacuum chamber
        if pt.p[1] == 0 and pt.p[0] >= (w-wc)/2 and pt.p[0] <= ((w-wc)/2 + wc):
            load_index.append(i)
            xload.append(0.0)
            if pt.p[0] == (w-wc)/2 or pt.p[0] == ((w-wc)/2 + wc):
                yload.append(pressure*avg_elsize/2)
                xloadnode.append(pt.p[0])
                yloadnode.append(pt.p[1])
            else:
                yload.append(pressure*avg_elsize)
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
        if pt.p[1] == 0:
            if pt.p[0] >= (w-wc)/2:
                if pt.p[0] <= ((w-wc)/2 + wc):
                    top_pts_i.append(i)
            
        if pt.p[1] == 0 and pt.p[0] < (w-wc)/2 or pt.p[0] > ((w-wc)/2):
            base_pts_i.append(i)
            
    return top_pts_i, base_pts_i


def max_top_displacement(UC, top_pts_i, h):
    max_displacement = 0
    for i in range(np.shape(UC)[0]):
        if i in top_pts_i:
            if UC[i][1] < max_displacement:
                max_displacement = UC[i][1]
                
    # if np.abs(max_displacement) > h:
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
                
    avg_stress = np.mean(base_stresses)
    
    return [max_y_stress, base_stresses, avg_stress]
    

def get_stress_displace(parameters, plot_contours):
    w = 5.0
    wc = parameters[0]
    h = parameters[1]
    # hl = parameters[3]
    
    elsize = 0.05
    
    pressure = -12.28
    youngs_modulus = 200.0
    poisson_ratio = 0.47
    
    mesh = rectangular_flat(w, wc, h, elsize)
    pts = list(mesh.Points())
    # while True:
    #     try:
    #         mesh = rectangular_single_legs(w, wl, h, hl, elsize)
    #     except:
    #         print("Impossible design generated")
    #     else:
    #         break
    
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
        max_displace = max_top_displacement(UC, top_pts_i, h)
        
        # Max sigma yy stress along the mold surface
        max_stress, base_stresses, avg_stress = max_y_stress(S_nodes, base_pts_i)
    except:
        max_displace = 0
        max_stress = 0
        avg_stress = 0
    
    return avg_stress, np.abs(max_displace)


class Particle:
    def __init__(self, bounds, initial_fitness):
        self.particle_position = []  # particle position
        self.particle_velocity = []  # particle velocity
        self.local_best_particle_position = []  # best position of the particle
        self.fitness_local_best_particle_position = initial_fitness  # initial objective function value of the best particle position
        self.fitness_particle_position = initial_fitness  # objective function value of the particle position
  
        for i in range(len(bounds)):
            self.particle_position.append(
                random.uniform(bounds[i][0], bounds[i][1]))  # generate random initial position
            self.particle_velocity.append(random.uniform(-1, 1))  # generate random initial velocity
  
    # def evaluate(self, designidx, list_of_swarm_particles):
    #     score = maximin_fitness(designidx, list_of_swarm_particles)
    #     self.fitness_particle_position = score
    #     if mm == -1:
    #         if self.fitness_particle_position < self.fitness_local_best_particle_position:
    #             self.local_best_particle_position = self.particle_position # update the local best
    #             self.fitness_local_best_particle_position = self.fitness_particle_position  # update the fitness of the local best
    #     if mm == 1:
    #         if self.fitness_particle_position > self.fitness_local_best_particle_position:
    #             self.local_best_particle_position = self.particle_position
    #             self.fitness_local_best_particle_position = self.fitness_particle_position
                
    def update_velocity(self, global_best_particle_position, w, c1, c2):
        for i in range(nv):
            r1 = random.random()
            r2 = random.random()
            
            cognitive_velocity = c1 * r1 * (self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2 * r2 * (global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w * self.particle_velocity[i] + cognitive_velocity + social_velocity
            
    def update_position(self, bounds):
        for i in range(nv):
            self.particle_position[i] = self.particle_position[i] + self.particle_velocity[i]
            
            # check and repair to satisfy the upper bounds
            if self.particle_position[i] > bounds[i][1]:
                self.particle_position[i] = bounds[i][1]
            # check and repair to satisfy the lower bounds
            if self.particle_position[i] < bounds[i][0]:
                self.particle_position[i] = bounds[i][0]
                

def optimize_particle_swarm(mm, bounds, particle_size, iterations, w, c1, c2):              
    # -----------------------------------------------------------------------------
    if mm == -1:
        initial_fitness = float("inf") # for minimization problem
    if mm == 1:
        initial_fitness = -float("inf") # for maximization problem
        
    # -----------------------------------------------------------------------------
    # Visualization
    fig = plt.figure(dpi=300)
    ax = fig.add_subplot()
    fig.show()
    # Begin solving the optimization problem here
    fitness_global_best_particle_position = initial_fitness
    global_best_particle_position = []
    
    # Create the group of swarm particles
    print("\nInitializing particle swarm...")
    swarm_particle = []
    for i in range(particle_size):
        swarm_particle.append(Particle(bounds, initial_fitness))
    A = []
    
    print("\nBegin optimization")
    for i in range(iterations):
        print("\nIteration {}\n".format(i))
        
        redval = i/(iterations)
        blueval = 1.0-redval
        rgbvals = (redval, 0, blueval)
        # Refactor maximin so it only evaluates each particle once, saves the
        # maximin fitness scores, and then evaluates the particle velocities
        # relative to the global best position. This should cut down on
        # the number of calculations required considerably.
        scores = maximin_fitness(swarm_particle)
        
        # Use the maximum score to determine the global_best_particle_position
        best_ind = scores.index(min(scores))
        global_best_particle_position = list(swarm_particle[best_ind].particle_position)
        fitness_global_best_particle_position = scores[best_ind]
        
        
        # for j in range(particle_size):
        #     swarm_particle[j].evaluate(j, swarm_particle)
            
        #     if mm == -1:
        #         if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
        #             global_best_particle_position = list(swarm_particle[j].particle_position)
        #             fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
        #     if mm == 1:
        #         if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
        #             global_best_particle_position = list(swarm_particle[j].particle_position)
        #             fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
        for j in range(particle_size):
            swarm_particle[j].update_velocity(global_best_particle_position, w, c1, c2)
            swarm_particle[j].update_position(bounds)
            
        A.append(fitness_global_best_particle_position) # record the best fitness
        
        # Visualization
        # ax.plot(A, color='r')
        
        wc = [particle.particle_position[0] for particle in swarm_particle]
        h = [particle.particle_position[1] for particle in swarm_particle]
        
        ax.scatter(wc,h,color=rgbvals)
        fig.canvas.draw()
        # ax.set_xlim(left=max(0, i - iterations), right=i + 3)
        ax.set_xlabel("wc")
        ax.set_ylabel("h")
        
    print("\nOptimal solution:", global_best_particle_position)
    print("Objective function value:", fitness_global_best_particle_position)
    plt.show()
    
    return scores, swarm_particle
    
    
def maximin_fitness(swarm_particle):
    # Take in a design and calculate the maximin score against the population of designs
    population = len(swarm_particle)
    
    print("\nCalculating stress and displacement for {} particles\n".format(population))
    fitnesses = []
    for i in range(population):
        avg_stress, max_displace = get_stress_displace(swarm_particle[i].particle_position, False)
        fitnesses.append([avg_stress, max_displace])
        
    numobj = len(fitnesses[0])
    
    # Scale fitness values to be the same scale (0 to 1)
    stresses = [fitness[0] for fitness in fitnesses]
    displacements = [fitness[1] for fitness in fitnesses]
    norm_stresses = [(stress - np.min(stresses)) / (np.max(stresses) - np.min(stresses)) for stress in stresses]
    norm_displace = [(displace - np.min(displacements)) / (np.max(displacements) - np.min(displacements)) for displace in displacements]
    # norm_displace = displacements
    
    fitnesses = []
    for i in range(population):
        fitnesses.append([norm_stresses[i], norm_displace[i]])
    
    # Calculate the maximin score for each particle
    print("\nCalculating maximin scores")
    scores = []
    for i in range(population):
        popfitnesses = []
        designfit = fitnesses[i]
        for j,fitness in enumerate(fitnesses):
            if j != i:
                popfitnesses.append(fitness)
                
        minvals = np.zeros(((population-1),numobj))
        for j in range(population-1):
            for k in range(numobj):
                minvals[j,k] = designfit[k] - popfitnesses[j][k]
                
        mins = np.zeros(population-1)
        for j in range(population-1):
            mins[j] = np.min(minvals[j,:])
        
        score = np.max(mins)
        swarm_particle[i].fitness_particle_position = score
        
        if mm == -1:
            if swarm_particle[i].fitness_particle_position < swarm_particle[i].fitness_local_best_particle_position:
                swarm_particle[i].local_best_particle_position = swarm_particle[i].particle_position # update the local best
                swarm_particle[i].fitness_local_best_particle_position = swarm_particle[i].fitness_particle_position  # update the fitness of the local best
        if mm == 1:
            if swarm_particle[i].fitness_particle_position > swarm_particle[i].fitness_local_best_particle_position:
                swarm_particle[i].local_best_particle_position = swarm_particle[i].particle_position
                swarm_particle[i].fitness_local_best_particle_position = swarm_particle[i].fitness_particle_position
        
        
        scores.append(score)
    
    # Return scores, which is the list of the maximin scores by particle index
    
    return scores
    
                
                
if __name__ == "__main__":
    wc_lb = 0.25
    h_lb = 0.125
    wc_ub = 1.5
    h_ub = 1.0
    bounds = [(wc_lb, wc_ub), (h_lb, h_ub)]  # upper and lower bounds of variables
    nv = len(bounds)  # number of variables
    mm = -1  # if minimization problem, mm = -1; if maximization problem, mm = 1
      
    # THE FOLLOWING PARAMETERS ARE OPTIONAL
    particle_size = 40  # number of particles
    iterations = 10  # max number of iterations
    w = 0.1  # inertia constant
    c1 = 4  # cognitive constant
    c2 = 0.5  # social constant
    
    scores, swarm_particle = optimize_particle_swarm(mm, bounds, particle_size, iterations, w, c1, c2)