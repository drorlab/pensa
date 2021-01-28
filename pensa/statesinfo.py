#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: neilthomson
"""

import numpy as np
from queue import PriorityQueue 
import math
import re
import glob
import itertools
#from tqdm import tqdm
from multiprocessing import Pool
from time import gmtime, strftime
import ast
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import os

"""
FUNCTIONS
"""

#CHECK IF VALUE IS BETWEEN X AND Y
def check(value,x,y):
    if x <= value <= y:
        return 1
    else:
        return 0
    

#CORRECTING FOR THE PERIODICITY OF ANGLES
def periodic_correction(angle1):
    new_dist=angle1.copy()
    ##removing any discrete states for water occupancy 
    continuous_angles = [i for i in new_dist if i != 10000.0]
    ##indexing all the continuous distribution angles
    index_cont_angles = [i for i, x in enumerate(new_dist) if x != 10000.0]      
    ##generating a histogram of the chi angles
    heights=np.histogram(continuous_angles, bins=90, density=True)
    ##if the first bar height is not the minimum bar height
    ##then find the minimum bar and shift everything before that bar by 360
    if heights[0][0] > min(heights[0]):   
        ##define the new periodic boundary for the shifted values as the first minima
        j=heights[1][np.where(heights[0] == min(heights[0]))[0][0]+1]
        for k in range(len(continuous_angles)):
            ##if the angle is before the periodic boundary, shift by 2*pi
            ## the boundaries in pyEMMA are in radians. [-pi, pi]
            if continuous_angles[k] < j:
                continuous_angles[k]+=2*np.pi
    for i in range(len(index_cont_angles)):
        new_dist[index_cont_angles[i]] = continuous_angles[i]
    return new_dist


def import_distribution(simulation_folder, file_name):
    dists = np.genfromtxt(simulation_folder + file_name, dtype=float, delimiter=",")
    return dists
    
    
# this function makes sure that the two simulations are the same length
def match_sim_lengths(sim1,sim2):
    if len(sim1)!=len(sim2):
        if len(sim1)>len(sim2):
            sim1=sim1[0:len(sim2)]
        if len(sim1)<len(sim2):
            sim2=sim2[0:len(sim1)]  
    return sim1, sim2
        

def get_filenames(folder):  
    files = [f.split(folder)[1] for f in glob.glob(folder + "*", recursive=True)]
    return files
    

#smoothing the kde data so that the extrema can be found without any of the small noise appearing as extrema
def smooth(x,window_len,window=None):
    if window is None:
        window_type='hanning'
    if x.ndim != 1:
        raise ValueError
    if x.size < window_len:
        raise ValueError
    if window_len<3:
        return x
    if not window_type in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError
    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window_type  ==  'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window_type+'(window_len)')
    y=np.convolve(w/w.sum(),s,mode='valid')
    return y


#FINDING THE NEAREST NEIGHBOUR FUNCTION
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


#GAUSSIAN FUNCTIONS
def gauss(x, x0, sigma, a):
    """ Gaussian function: """
    return abs(a*np.exp(-(x-x0)**2/(2*sigma**2)))
def bimodal(x,mu1,sigma1,A1,mu2,sigma2,A2):
    """ Two gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)
def trimodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3):
    """ Three gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)
def quadmodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4):
    """ Four gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)
def quinmodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5):
    """ Five gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)
def sexmodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5,mu6,sigma6,A6):
    """ Six gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)   
def septmodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5,mu6,sigma6,A6,mu7,sigma7,A7):
    """ Seven gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)+gauss(x,mu7,sigma7,A7)    
def octomodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5,mu6,sigma6,A6,mu7,sigma7,A7,mu8,sigma8,A8):
    """ Seven gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)+gauss(x,mu7,sigma7,A7)+gauss(x,mu8,sigma8,A8)    
def nonamodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5,mu6,sigma6,A6,mu7,sigma7,A7,mu8,sigma8,A8,mu9,sigma9,A9):
    """ Seven gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)+gauss(x,mu7,sigma7,A7)+gauss(x,mu8,sigma8,A8)+gauss(x,mu9,sigma9,A9)      
def decamodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5,mu6,sigma6,A6,mu7,sigma7,A7,mu8,sigma8,A8,mu9,sigma9,A9,mu10,sigma10,A10):
    """ Seven gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)+gauss(x,mu7,sigma7,A7)+gauss(x,mu8,sigma8,A8)+gauss(x,mu9,sigma9,A9)+gauss(x,mu10,sigma10,A10)        





#PRINT K CLOSEST VALUES TO SPECIFIED VALUE
def printKclosest(arr,n,x,k): 
    a=[]
    # Make a max heap of difference with  
    # first k elements.  
    pq = PriorityQueue() 
    for i in range(k): 
        pq.put((-abs(arr[i]-x),i)) 
    # Now process remaining elements 
    for i in range(k,n): 
        diff = abs(arr[i]-x) 
        p,pi = pq.get() 
        curr = -p 
        # If difference with current  
        # element is more than root,  
        # then put it back.  
        if diff>curr: 
            pq.put((-curr,pi)) 
            continue
        else: 
            # Else remove root and insert 
            pq.put((-diff,i))           
    # Print contents of heap. 
    while(not pq.empty()): 
        p,q = pq.get() 
        a.append(str("{} ".format(arr[q])))
    return a

## Gaussian integral for evaluating state probabilities
def integral(x, mu, sigma, A):
    integral = (A/2) * (1 + math.erf((x - mu) / (sigma * np.sqrt(2))))
    return integral

## obatining the gaussians that fit the distribution
## bin number is chosen based on 3 degree resolution (120 bins for 360 degrees)
## smoothing is chosen based on 95% significance level (i.e. 0.05*binnumber) 
def get_gaussian_fit(distribution, gauss_bin, gauss_smooth):    
    histo=np.histogram(distribution, bins=gauss_bin, density=True)
    distributionx=smooth(histo[1][0:-1],gauss_smooth)
    ##this shifts the histo values down by the minimum value to help with finding a minimum
    distributiony=smooth(histo[0]-min(histo[0]),gauss_smooth)
    ##getting an array of all the maxima indices
    maxima = [distributiony[item] for item in argrelextrema(distributiony, np.greater)][0]
    ##the maxima may be an artifact of undersampling
    ##this grabs only the maxima that correspond to a density greater than the cutoff
    ##cutoff= 0.75% at a 99.25% significance level
    corrected_extrema=[item for item in maxima if item > max(distributiony)*0.0075]
    ##finding the guess parameters for the plots
    ##empty lists for the guess params to be added to
    mean_pop=[]
    sigma_pop=[]
    ##number of closest neighbours
    ##setting for the sigma finding function
    noc=28
    ##for all the extrema, find the 'noc' yval 
    ##closest to half max yval
    ##this is to find the edges of the gaussians for calculating sigma
    sig_vals=[]
    for extrema in corrected_extrema:
        
        mean_xval=distributionx[np.where(distributiony==extrema)[0][0]]
        ##finding the "noc" closest y values to the 1/2 max value of each extrema
        closest=printKclosest(distributiony, len(distributiony), extrema*0.5, noc)
        ##finding the x closest to the mean
        xlist=[np.where(distributiony==float(closesty))[0][0] for closesty in closest]
        xsig=find_nearest(distributionx[xlist],mean_xval)
        ##obtaining the width of the distribution
        sig=np.absolute(xsig-mean_xval)
        sig_vals.append(sig)        
    ##the mean x of the gaussian is the value of x at the peak of y
    mean_vals=[distributionx[np.where(distributiony==extrema)[0][0]] for extrema in corrected_extrema]
    for i in range(len(corrected_extrema)):
        mean_pop.append(mean_vals[i])
        sigma_pop.append(sig_vals[i])
    ##x is the space of angles
    xline=np.linspace(min(distribution),max(distribution),10000)                
    ##choosing the fitting mode
    peak_number=[gauss,bimodal,trimodal,quadmodal,quinmodal,sexmodal,septmodal,octomodal,nonamodal,decamodal]
    mode=peak_number[len(sig_vals)-1]    
    expected=[]
    for i in range(len(mean_pop)):
        expected.append(mean_pop[i])
        expected.append(sigma_pop[i])
        expected.append(corrected_extrema[i])    
    # try:
    params,cov=curve_fit(mode,distributionx,distributiony,expected,maxfev=1000000)   
    # except:
        
    gaussians=[]
    gaussnumber=np.linspace(0,(len(params))-3,int(len(params)/3))    
    # for j in gaussnumber:
    #     gaussians.append(gauss(xline, params[0+int(j)], params[1+int(j)], params[2+int(j)]))
    for j in gaussnumber:
        intmax = integral(max(distribution),params[0+int(j)], params[1+int(j)], params[2+int(j)])
        intmin = integral(min(distribution),params[0+int(j)], params[1+int(j)], params[2+int(j)])
        if np.abs(intmax-intmin)>0.02:
            gaussians.append(gauss(xline, params[0+int(j)], params[1+int(j)], params[2+int(j)]))
    return gaussians, xline


# OBTAINING THE GAUSSIAN INTERSECTS
def get_intersects(gaussians,distribution,xline, write_plots=None,write_name=None):
    ##adding the minimum angle value as the first boundary
    all_intersects=[min(distribution)]
    mean_gauss_xval=[]
    for i in range(len(gaussians)):
        mean_gauss_xval.append(xline[list(gaussians[i]).index(max(gaussians[i]))])
        
    reorder_indices=[mean_gauss_xval.index(i) for i in sorted(mean_gauss_xval)]    
    ##sort gaussians in order of their mean xval and ignore gaussians with maxima below 0.0001
    reorder_gaussians=[gaussians[i] for i in reorder_indices if max(gaussians[i])>0.0001]
        
    for i in range(len(reorder_gaussians)-1):    
        ##Find indices between neighbouring gaussians
        idx = np.argwhere(np.diff(np.sign(reorder_gaussians[i] - reorder_gaussians[i+1]))).flatten()      
        if len(idx)==1:
            all_intersects.append(xline[idx][0])   
        elif len(idx)!=0:
            ##selects the intersect with the maximum probability
            ##to stop intersects occuring when the gaussians trail to zero further right on the plot
            intersect_ymax=max([reorder_gaussians[i][intersect] for intersect in idx])
            intersect_ymax_index=[item for item in idx if reorder_gaussians[i][item]==intersect_ymax]            
            all_intersects.append(xline[intersect_ymax_index])
        ##for gaussian neighbours that don't intersect, set state limit as center between maxima
        elif len(idx)==0:            
            gauss_max1=list(reorder_gaussians[i]).index(max(reorder_gaussians[i]))
            gauss_max2=list(reorder_gaussians[i+1]).index(max(reorder_gaussians[i+1]))
            intersect =  0.5* np.abs(xline[gauss_max2] +  xline[gauss_max1])
            all_intersects.append(intersect)
            
    all_intersects.append(max(distribution))  
        
    if write_plots is not None:
        if not os.path.exists('ssi_plots/'):
            os.makedirs('ssi_plots/')
        plt.figure()      
        plt.ion()
        plt.hist(distribution,bins=360, density=True, alpha=0.5)
        for j in range(len(reorder_gaussians)):
            plt.plot(xline, reorder_gaussians[j], lw=2)        
        for i in range(len(all_intersects)):
            plt.axvline(all_intersects[i],color='k',lw=1,ls='--')   
        plt.xlabel('Radians')
        plt.ylabel('Count')
        plt.title(write_name)        
        plt.ioff()
        plt.savefig('ssi_plots/'+write_name+".png")
    
    return all_intersects
    

##this function requires a distribution to cluster/discretize into states
##The function handles both residue angle distributions and water orientation/occupancy
##distributions. For waters, the assignment of an additional non-angular state is performed if
## changes in pocket occupancy occur.
def determine_state_limits(distr, gauss_bins=120, gauss_smooth=10, write_plots=None, write_name=None):    
    new_dist=distr.copy()
    distribution=[item for item in new_dist if item != 10000.0]
    ##obtaining the gaussian fit
    gaussians, xline = get_gaussian_fit(distribution,gauss_bins,gauss_smooth)            
    ##discretising each state by gaussian intersects       
    intersection_of_states = get_intersects(gaussians, distribution, xline,  write_plots, write_name)   
    if distr.count(10000.0)>=1:
        intersection_of_states.append(20000.0)  
    
    order_intersect=np.sort(intersection_of_states)  
    return list(order_intersect)

def calculate_entropy(state_limits,distribution_list):
    ## subtract 1 since number of partitions = number of states - 1
    mut_prob=np.zeros(([len(state_limits[i])-1 for i in range(len(state_limits))]))     
    ##obtaining the entropy
    entropy=0
    ##iterating over every multidimensional index in the array
    it = np.nditer(mut_prob, flags=['multi_index'])
    while not it.finished:
        ##grabbing the indices of each element in the matrix
        arrayindices=list(it.multi_index)
        ##making an array of the state occupancy of each distribution
        limit_occupancy_checks=np.zeros((len(arrayindices), len(distribution_list[0])))
        for i in range(len(arrayindices)):
            ##obtaining the limits of each state 
            ##in the identified array position            
            limits=[state_limits[i][arrayindices[i]],state_limits[i][arrayindices[i]+1]]
            ##grabbing the distribution that the limits corresponds to
            distribution=distribution_list[i]
            ##checking the occupancy of this state with the distribution
            for j in range(len(distribution)):
                limit_occupancy_checks[i][j]=check(distribution[j],limits[0],limits[1]) 
        ##calculating the probability as a function of the occupancy
        mut_prob[it.multi_index]=sum(np.prod(limit_occupancy_checks,axis=0)) / len(limit_occupancy_checks[0])
        ##calculating the entropy as the summation of all -p*log(p) 
        if mut_prob[it.multi_index] != 0:
            entropy+=-1*mut_prob[it.multi_index]*math.log(mut_prob[it.multi_index],2)
        it.iternext()
    return entropy

##this function requires a list of angles for SSI
##SSI(A,B) = H(A) + H(B) - H(A,B)
def calculate_ssi(set_distr_a, set_distr_b=None, a_states=None, b_states=None,
                  gauss_bins=120, gauss_smooth=10, write_plots=None, write_name=None):
        
    
    ##calculating the entropy for set_distr_a
    ## if set_distr_a only contains one distributions
    if any(isinstance(i, list) for i in set_distr_a)==0:
        distr_a=[periodic_correction(set_distr_a)]
    ## else set_distr_a is a nested list of multiple distributions (bivariate)
    else:
        distr_a=[periodic_correction(i) for i in set_distr_a]
    
    if a_states is None:    
        distr_a_states=[]
        for i in range(len(distr_a)):
            if write_name is not None:
                plot_name = write_name + '_dist' + str(i)
            else:
                plot_name = None
            distr_a_states.append(determine_state_limits(distr_a[i], gauss_bins, gauss_smooth, write_plots, plot_name))
    else:
        distr_a_states = a_states
        
    H_a=calculate_entropy(distr_a_states,distr_a) 
            
    ##calculating the entropy for set_distr_b
    ## if no dist (None) then apply the binary dist for two simulations
    if set_distr_b is None:       
        H_b=1
        distr_b=[[0.5]*int(len(distr_a[0])/2) + [1.5]*int(len(distr_a[0])/2)]
        distr_b_states= [[0,1,2]]  
        
    else:
        if any(isinstance(i, list) for i in set_distr_b)==0:
            distr_b=[periodic_correction(set_distr_b)]
        else:
            distr_b=[periodic_correction(i) for i in set_distr_b]
        if b_states is None:    
            distr_b_states=[]
            for i in range(len(distr_b)):
                if write_name is not None:
                    plot_name = write_name + '_dist' + str(i)
                else:
                    plot_name = None
                distr_b_states.append(determine_state_limits(distr_b[i], gauss_bins, gauss_smooth, write_plots, plot_name))
        else:
            distr_b_states = b_states
            
                
        H_b=calculate_entropy(distr_b_states,distr_b)

    ab_joint_states= distr_a_states + distr_b_states
    ab_joint_distributions= distr_a + distr_b
    
    H_ab=calculate_entropy(ab_joint_states,ab_joint_distributions)

    SSI = (H_a + H_b) - H_ab
        
    return SSI


#CoSSI = H_a + H_b + H_c - H_ab - H_bc - H_ac + H_abc
def calculate_cossi(set_distr_a, set_distr_b, set_distr_c=None, a_states=None, b_states=None,
                    c_states=None, gauss_bins=120, gauss_smooth=10, write_plots=None,write_name=None):
        
        
    ##calculating the entropy for set_distr_a
    if any(isinstance(i, list) for i in set_distr_a)==0:
        distr_a=[periodic_correction(set_distr_a)]
    else:
        distr_a=[periodic_correction(i) for i in set_distr_a]
    if a_states is None:    
        distr_a_states=[]
        for i in range(len(distr_a)):
            if write_name is not None:
                plot_name = write_name + '_dist' + str(i)
            else:
                plot_name = None
            distr_a_states.append(determine_state_limits(distr_a[i], gauss_bins, gauss_smooth, write_plots, plot_name))
    else:
        distr_a_states = a_states    
    H_a=calculate_entropy(distr_a_states,distr_a)
        
    ##----------------
    ##calculating the entropy for set_distr_b
    if any(isinstance(i, list) for i in set_distr_b)==0:
        distr_b=[periodic_correction(set_distr_b)]
    else:
        distr_b=[periodic_correction(i) for i in set_distr_b]
    if b_states is None:    
        distr_b_states=[]
        for i in range(len(distr_b)):
            if write_name is not None:
                plot_name = write_name + '_dist' + str(i)
            else:
                plot_name = None
            distr_b_states.append(determine_state_limits(distr_b[i], gauss_bins, gauss_smooth, write_plots, plot_name))
    else:
        distr_b_states = b_states
    H_b=calculate_entropy(distr_b_states,distr_b) 
    
    ##----------------
    ##calculating the entropy for set_distr_c
    ## if no dist (None) then apply the binary dist for two simulations
    if set_distr_c is None:
        H_c=1
        distr_c=[[0.5]*int(len(distr_a[0])/2) + [1.5]*int(len(distr_a[0])/2)]
        distr_c_states= [[0,1,2]]  
    else:
        if any(isinstance(i, list) for i in set_distr_c)==0:
            distr_c=[periodic_correction(set_distr_c)]
        else:
            distr_c=[periodic_correction(i) for i in set_distr_c]
        if c_states is None:    
            distr_c_states=[]
            for i in range(len(distr_c)):
                if write_name is not None:
                    plot_name = write_name + '_dist' + str(i)
                else:
                    plot_name = None
                distr_c_states.append(determine_state_limits(distr_c[i], gauss_bins, gauss_smooth, write_plots, plot_name))
        else:
            distr_c_states = c_states
        H_c=calculate_entropy(distr_c_states,distr_c)

    ##----------------
    ab_joint_states= distr_a_states + distr_b_states
    ab_joint_distributions= distr_a + distr_b
    
    H_ab=calculate_entropy(ab_joint_states,ab_joint_distributions)
    ##----------------
    ac_joint_states= distr_a_states + distr_c_states 
    ac_joint_distributions= distr_a + distr_c
    
    H_ac= calculate_entropy(ac_joint_states,ac_joint_distributions)
    ##----------------
    bc_joint_states= distr_b_states + distr_c_states 
    bc_joint_distributions= distr_b + distr_c
    
    H_bc= calculate_entropy(bc_joint_states,bc_joint_distributions)
    ##----------------
    abc_joint_states= distr_a_states + distr_b_states + distr_c_states 
    abc_joint_distributions= distr_a + distr_b + distr_c
    
    H_abc=calculate_entropy(abc_joint_states,abc_joint_distributions)    
    
    
    SSI = (H_a + H_b) - H_ab
    coSSI = (H_a + H_b + H_c) - (H_ab + H_ac + H_bc) + H_abc 
        
    return SSI, coSSI

