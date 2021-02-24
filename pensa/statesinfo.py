#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Neil J Thomson
"""

import numpy as np
from queue import PriorityQueue 
import math
import glob
#from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import argrelextrema
import os
import re


def check(value,x,y):
    """
    Check if a value is between x and y

    Parameters
    ----------
    value : float
        Value of interest.
    x : float
        Limit x.
    y : float
        Limit y.

    Returns
    -------
    int
        Numerical bool if value is between limits x and y.

    """
    if x <= value <= y:
        return 1
    else:
        return 0
    

def periodic_correction(angle1):
    """
    Correcting for the periodicity of angles in radians. This ensures state 
    clustering defines states that span over the -pi, pi boundary as one state. 
    Waters featurized using PENSA are assigned dscrete state of 10000.0 for 
    representation of pocket occupancy, these values are ignored within the 
    periodic correction so that only the water orientation periodicity is handled
    
    
    Parameters
    ----------
    angle1 : list
        Univariate distribution for a specific feature.

    Returns
    -------
    new_dist : list
        Periodically corrected distribution.

    """
    new_dist=angle1.copy()
    continuous_angles = [angle for angle in new_dist if angle != 10000.0]
    index_cont_angles = [index for index, angle in enumerate(new_dist) if angle != 10000.0]      
    heights=np.histogram(continuous_angles, bins=90, density=True)
    ##if the first bar height is not the minimum bar height
    ##then find the minimum bar and shift everything before that bar by 360
    if heights[0][0] > min(heights[0]):   
        ##define the new periodic boundary for the shifted values as the first minima
        perbound=heights[1][np.where(heights[0] == min(heights[0]))[0][0]+1]
        for angle_index in range(len(continuous_angles)):
            ##if the angle is before the periodic boundary, shift by 2*pi
            ## the boundaries in pyEMMA are in radians. [-pi, pi]
            if continuous_angles[angle_index] < perbound:
                continuous_angles[angle_index]+=2*np.pi
    for index in range(len(index_cont_angles)):
        new_dist[index_cont_angles[index]] = continuous_angles[index]
    
    
    
    return new_dist


def import_distribution(simulation_folder, file_name):
    """
    Import data written to a txt file.

    Parameters
    ----------
    simulation_folder : str
        Path to the folder containing files for import.
    file_name : str
        Name of the file within the simulation_folder to be imported.

    Returns
    -------
    dists : array
        Array of the file data.

    """
    dists = np.loadtxt(simulation_folder + file_name,  delimiter=",")
    return dists
    
    
# this function makes sure that the two simulations are the same length
def match_sim_lengths(sim1,sim2):
    """
    Make two lists the same length by truncating the longer list to match.

    Parameters
    ----------
    sim1 : list
        A one dimensional distribution of a specific feature.
    sim2 : list
        A one dimensional distribution of a specific feature.

    Returns
    -------
    sim1 : list
        A one dimensional distribution of a specific feature.
    sim2 : list
        A one dimensional distribution of a specific feature.

    """
    if len(sim1)!=len(sim2):
        if len(sim1)>len(sim2):
            sim1=sim1[0:len(sim2)]
        if len(sim1)<len(sim2):
            sim2=sim2[0:len(sim1)]  
    return sim1, sim2
        

def get_filenames(folder):  
    """
    Obtain all the names of the files within a certain folder.

    Parameters
    ----------
    folder : str
        Location of the folder to obtain filenames.

    Returns
    -------
    files : list of str
        All filenames within folder.

    """
    files = [file.split(folder)[1] for file in glob.glob(folder + "*", recursive=True)]
    return files
    

#smoothing the kde data so that the extrema can be found without any of the small noise appearing as extrema
def smooth(x,window_len,window=None):
    """
    

    Parameters
    ----------
    x : list
        Distribution to be smoothed.
    window_len : int
        number of bins to smooth over.
    window : str, optional
        Type of window to use for the smoothing. The default is None=Hanning.

    Raises
    ------
    ValueError
        If window argument is not recognised.

    Returns
    -------
    list
        Smoothed distribution.

    """
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
def find_nearest(distr, value):
    """
    Find the nearest value in a distribution to an arbitrary reference value.

    Parameters
    ----------
    distr : list
        The distribution to locate a certain point within.
    value : float
        Reference value for locating within the distribution.

    Returns
    -------
    float
        Closest value to reference value in distribution.

    """
    array = np.array(distr)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


#GAUSSIAN FUNCTIONS
def gauss(x, x0, sigma, a):
    """

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    x0 : TYPE
        DESCRIPTION.
    sigma : TYPE
        DESCRIPTION.
    a : TYPE
        DESCRIPTION.

    Returns
    -------
    gaussian : TYPE
        DESCRIPTION.

    """

    if sigma != 0:
        gaussian = abs(a*np.exp(-(x-x0)**2/(2*sigma**2)))
    return gaussian

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
    """ Eight gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)+gauss(x,mu7,sigma7,A7)+gauss(x,mu8,sigma8,A8)    
def nonamodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5,mu6,sigma6,A6,mu7,sigma7,A7,mu8,sigma8,A8,mu9,sigma9,A9):
    """ Nine gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)+gauss(x,mu7,sigma7,A7)+gauss(x,mu8,sigma8,A8)+gauss(x,mu9,sigma9,A9)      
def decamodal(x,mu1,sigma1,A1,mu2,sigma2,A2,mu3,sigma3,A3,mu4,sigma4,A4,mu5,sigma5,A5,mu6,sigma6,A6,mu7,sigma7,A7,mu8,sigma8,A8,mu9,sigma9,A9,mu10,sigma10,A10):
    """ Ten gaussians """
    return gauss(x,mu1,sigma1,A1)+gauss(x,mu2,sigma2,A2)+gauss(x,mu3,sigma3,A3)+gauss(x,mu4,sigma4,A4)+gauss(x,mu5,sigma5,A5)+gauss(x,mu6,sigma6,A6)+gauss(x,mu7,sigma7,A7)+gauss(x,mu8,sigma8,A8)+gauss(x,mu9,sigma9,A9)+gauss(x,mu10,sigma10,A10)        





def printKclosest(arr,n,x,k): 
    """
    Print K closest values to a specified value. 

    Parameters
    ----------
    arr : list
        The distribution of values.
    n : int
        Search through the first n values of arr for k closest values.
    x : float
        The reference value for which the closest values are sought.
    k : int
        Number of closest values desired.

    Returns
    -------
    a : list
        The closest k values to x.

    """
    a=[]
    # Make a max heap of difference with  
    # first k elements.  
    pq = PriorityQueue() 
    for neighb in range(k): 
        pq.put((-abs(arr[neighb]-x),neighb)) 
    # Now process remaining elements 
    for neighb in range(k,n): 
        diff = abs(arr[neighb]-x) 
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
            pq.put((-diff,neighb))           
    # Print contents of heap. 
    while(not pq.empty()): 
        p,q = pq.get() 
        a.append(str("{} ".format(arr[q])))
    return a

def integral(x, mu, sigma, A):
    """
    Gaussian integral for evaluating state probabilities. Integration between
    negative infinity and x.

    Parameters
    ----------
    x : float
        Upper limit for integral.
    mu : float
        Gaussian mean.
    sigma : float
        Gaussian sigma.
    A : float
        Gaussian amplitude.

    Returns
    -------
    integral : float
        Area under Gaussian from negative infinity to x.

    """
    integral = (A/2) * (1 + math.erf((x - mu) / (sigma * np.sqrt(2))))
    return integral


def gauss_fit(distribution, gauss_bin, gauss_smooth):    
    """
    Obtaining the gaussians to fit the distribution into a Gaussian mix. 
    Bin number is chosen based on 3 degree resolution (120 bins for 360 degrees)

    Parameters
    ----------
    distribution : list
        Distribution of interest for the fitting.
    gauss_bin : int
        Bin the distribution into gauss_bin bins.
    gauss_smooth : int
        Smooth the distribution according to a Hanning window length of gauss_smooth.

    Returns
    -------
    gaussians : list
        y-axis values for the Gaussian distribution.
    xline : list
        x-axis values for the Gaussian distribution.

    """
    histo=np.histogram(distribution, bins=gauss_bin, density=True)
    distributionx=smooth(histo[1][0:-1],gauss_smooth)
    ##this shifts the histo values down by the minimum value to help with finding a minimum
    distributiony=smooth(histo[0]-min(histo[0]),gauss_smooth)
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
    ##for all the extrema, find the 'noc' yval closest to half max yval
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
        FWHM=np.absolute(xsig-mean_xval)
        sigma = FWHM /(2*(np.sqrt(2*np.log(2)))) 
        sig_vals.append(sigma)        
    ##the mean x of the gaussian is the value of x at the peak of y
    mean_vals=[distributionx[np.where(distributiony==extrema)[0][0]] for extrema in corrected_extrema]
    for extr_num in range(len(corrected_extrema)):
        mean_pop.append(mean_vals[extr_num])
        sigma_pop.append(sig_vals[extr_num])
    ##x is the space of angles
    xline=np.linspace(min(distribution),max(distribution),10000)                
    ##choosing the fitting mode
    peak_number=[gauss,bimodal,trimodal,quadmodal,quinmodal,sexmodal,septmodal,octomodal,nonamodal,decamodal]
    mode=peak_number[len(sig_vals)-1]    
    expected=[]
    for param_num in range(len(mean_pop)):
        expected.append(mean_pop[param_num])
        expected.append(sigma_pop[param_num])
        expected.append(corrected_extrema[param_num])    
    # try:
    params,cov=curve_fit(mode,distributionx,distributiony,expected,maxfev=1000000)   
    # except:
        
    gaussians=[]
    gauss_num_space=np.linspace(0,(len(params))-3,int(len(params)/3))    
    # for j in gaussnumber:
    #     gaussians.append(gauss(xline, params[0+int(j)], params[1+int(j)], params[2+int(j)]))
    for gauss_index in gauss_num_space:
        intmax = integral(max(distribution),params[0+int(gauss_index)], params[1+int(gauss_index)], params[2+int(gauss_index)])
        intmin = integral(min(distribution),params[0+int(gauss_index)], params[1+int(gauss_index)], params[2+int(gauss_index)])
        if np.abs(intmax-intmin)>0.02:
            gaussians.append(gauss(xline, params[0+int(gauss_index)], params[1+int(gauss_index)], params[2+int(gauss_index)]))
    return gaussians, xline

##new function to adjust the gaussian parameters until they work
def smart_gauss_fit(distr, gauss_bins=120, gauss_smooth=None, write_name=None):
    
        
    if gauss_smooth is None:
        gauss_smooth = int(gauss_bins*0.1)        
        
    
    trial=0
    attempt_no=0
    bin_origin=gauss_bins
    bin_adjust_up= np.array(range(1,10000))
    bin_adjust_down= bin_adjust_up.copy()*-1
    bin_adjust= np.insert(bin_adjust_up, np.arange(len(bin_adjust_down)), bin_adjust_down)
    
    while trial<1:
        try:
            gaussians, xline = gauss_fit(distr, gauss_bins, gauss_smooth)
            trial+=1
        except:
            attempt_no+=1
            trial=0
            gauss_bins= bin_origin + bin_adjust[attempt_no]
        
    if attempt_no > 0:
        if write_name is None:
            print('Warning: Altered gauss_bins by '+str(bin_adjust[attempt_no])+' for clustering.\nYou might want to check cluster plot.')
        else:
            print('Warning: Altered gauss_bins by '+str(bin_adjust[attempt_no])+' for clustering of '+write_name+'.\nYou might want to check cluster plot.')
  
    return gaussians, xline

def get_intersects(gaussians,distribution,xline, write_plots=None,write_name=None):
    """
    Obtain the intersects of a mixture of Gaussians which have been obtained
    from decomposing a distribution into Gaussians. Additional state limits are
    added at the beginning and end of the distribution.

    Parameters
    ----------
    gaussians : list of lists
        A list of X gaussians.
    distribution : list
        The distribution that Gaussians have been obtained from.
    xline : list
        The x-axis linespace that the distribution spans.
    write_plots : bool, optional
        If true, visualise the states over the raw distribution. The default is None.
    write_name : str, optional
        Filename for write_plots. The default is None.


    Returns
    -------
    all_intersects : list
        All the Gaussian intersects.

    """
    ##adding the minimum angle value as the first boundary
    all_intersects=[min(distribution)]
    mean_gauss_xval=[]
    for gauss_num in range(len(gaussians)):
        mean_gauss_xval.append(xline[list(gaussians[gauss_num]).index(max(gaussians[gauss_num]))])
        
    reorder_indices=[mean_gauss_xval.index(mean) for mean in sorted(mean_gauss_xval)]    
    ##sort gaussians in order of their mean xval and ignore gaussians with maxima below 0.0001 (99% sig)
    reorder_gaussians=[gaussians[gauss_num] for gauss_num in reorder_indices if max(gaussians[gauss_num])>0.0001]
        
    for gauss_index in range(len(reorder_gaussians)-1):    
        ##Find indices between neighbouring gaussians
        idx = np.argwhere(np.diff(np.sign(reorder_gaussians[gauss_index] - reorder_gaussians[gauss_index+1]))).flatten()
        if len(idx)==1:
            all_intersects.append(float(xline[idx][0]) )
        elif len(idx)!=0:
            ##selects the intersect with the maximum probability
            ##to stop intersects occuring when the gaussians trail to zero further right on the plot
            intersect_ymax=max([reorder_gaussians[gauss_index][intersect] for intersect in idx])
            intersect_ymax_index=[item for item in idx if reorder_gaussians[gauss_index][item]==intersect_ymax]            
            all_intersects.append(float(xline[intersect_ymax_index]))
        ##for gaussian neighbours that don't intersect, set state limit as center between maxima
        elif len(idx)==0:            
            gauss_max1=list(reorder_gaussians[gauss_index]).index(max(reorder_gaussians[gauss_index]))
            gauss_max2=list(reorder_gaussians[gauss_index+1]).index(max(reorder_gaussians[gauss_index+1]))
            intersect =  0.5* np.abs(xline[gauss_max2] +  xline[gauss_max1])
            all_intersects.append(float(intersect))
            
    all_intersects.append(max(distribution))  
        
    if write_plots is not None:
        if not os.path.exists('ssi_plots/'):
            os.makedirs('ssi_plots/')
        plt.figure()      
        plt.ion()
        plt.hist(distribution,bins=360, density=True, alpha=0.5)
        for gauss_index in range(len(reorder_gaussians)):
            plt.plot(xline, reorder_gaussians[gauss_index], lw=2)        
        for intersect_index in range(len(all_intersects)):
            plt.axvline(all_intersects[intersect_index],color='k',lw=1,ls='--')   
        plt.xlabel('Radians')
        plt.ylabel('Count')
        plt.title(write_name)        
        plt.ioff()
        plt.savefig('ssi_plots/'+write_name+".png")
    
    return all_intersects
    


def determine_state_limits(distr, gauss_bins=120, gauss_smooth=None, write_plots=None, write_name=None):    
    """
    Cluster a distribution into discrete states with well-defined limits.
    The function handles both residue angle distributions and water 
    distributions. For waters, the assignment of an additional non-angular 
    state is performed if changes in pocket occupancy occur. The clustering
    requires that the distribution can be decomposed to a mixture of Gaussians. 

    Parameters
    ----------
    distr : list
        Distribution for specific feature.
    gauss_bins : int, optional
        Number of histogram bins to assign for the clustering algorithm. 
        The default is 120.
    gauss_smooth : int, optional
        Number of bins to perform smoothing over. The default is 10.
    write_plots : bool, optional
        If true, visualise the states over the raw distribution. The default is None.
    write_name : str, optional
        Filename for write_plots. The default is None.

    Returns
    -------
    list
        State intersects for each cluster in numerical order.

    """
    new_dist=distr.copy()
    distribution=[item for item in new_dist if item != 10000.0]
    ##obtaining the gaussian fit
    gaussians, xline = smart_gauss_fit(distribution,gauss_bins, gauss_smooth, write_name)
    # gaussians, xline = gauss_fit(distribution,gauss_bins,gauss_smooth)            
    ##discretising each state by gaussian intersects       
    intersection_of_states = get_intersects(gaussians, distribution, xline,  write_plots, write_name)   
    if distr.count(10000.0)>=1:
        intersection_of_states.append(20000.0)  
    
    order_intersect=np.sort(intersection_of_states)  
    return list(order_intersect)

def calculate_entropy(state_limits,distribution_list):
    """
    Calculate the Shannon entropy of a distribution as the summation of all 
    -p*log(p) where p refers to the probability of a conformational state. 
    
    Parameters
    ----------
    state_limits : list of lists
        A list of values that represent the limits of each state for each
        distribution.
    distribution_list : list of lists
        A list containing multivariate distributions (lists) for a particular
        residue or water

    Returns
    -------
    entropy : float
        The Shannon entropy value 

    """
    ## subtract 1 since number of states = number of partitions - 1
    mut_prob=np.zeros(([len(state_limits[i])-1 for i in range(len(state_limits))]))     
    entropy=0
    ##iterating over every multidimensional index in the array
    it = np.nditer(mut_prob, flags=['multi_index'])

    while not it.finished:
        arrayindices=list(it.multi_index)
        limit_occupancy_checks=np.zeros((len(arrayindices), len(distribution_list[0])))
        
        for dist_num in range(len(arrayindices)):
            limits=[state_limits[dist_num][arrayindices[dist_num]], state_limits[dist_num][arrayindices[dist_num]+1]]
            distribution=distribution_list[dist_num]
        
            for frame_num in range(len(distribution)):
                limit_occupancy_checks[dist_num][frame_num]= check(distribution[frame_num],limits[0],limits[1]) 
        mut_prob[it.multi_index]= sum(np.prod(limit_occupancy_checks,axis=0)) / len(limit_occupancy_checks[0])
        ##calculating the entropy as the summation of all -p*log(p) 
        
        if mut_prob[it.multi_index] != 0:
            entropy+=-1*mut_prob[it.multi_index]*math.log(mut_prob[it.multi_index],2)
        it.iternext()
    return entropy


##this function requires a list of angles for SSI
##SSI(A,B) = H(A) + H(B) - H(A,B)
def calculate_ssi(distr_a_input, distr_b_input=None, a_states=None, b_states=None,
                  gauss_bins=120, gauss_smooth=None, write_plots=None, write_name=None):
    
    
    if gauss_smooth is None:
        gauss_smooth = int(gauss_bins*0.1)
    
    try:       
        ##calculating the entropy for set_distr_a
        ## if set_distr_a only contains one distributions
        if type(distr_a_input[0]) is not list:
            set_distr_a=[periodic_correction(distr_a_input)]
        ## else set_distr_a is a nested list of multiple distributions (bivariate)
        else:
            set_distr_a=[periodic_correction(distr_a) for distr_a in distr_a_input]
        
        if a_states is None:    
            set_a_states=[]
            for dim_num in range(len(set_distr_a)):
                if write_name is not None:
                    plot_name = write_name + '_dist' + str(dim_num)
                else:
                    plot_name = None
                try:
                    set_a_states.append(determine_state_limits(set_distr_a[dim_num], gauss_bins, gauss_smooth, write_plots, plot_name))
                except:
                    print('Distribution A not clustering properly.\nTry altering Gaussian parameters or input custom states.')
        else:
            set_a_states = a_states
            
        H_a=calculate_entropy(set_a_states,set_distr_a) 
                
        ##calculating the entropy for set_distr_b
        ## if no dist (None) then apply the binary dist for two simulations
        if distr_b_input is None:       
            H_b=1
            set_distr_b=[[0.5]*int(len(set_distr_a[0])/2) + [1.5]*int(len(set_distr_a[0])/2)]
            set_b_states= [[0,1,2]]  
            
        else:
            if type(distr_b_input[0]) is not list:
                set_distr_b=[periodic_correction(distr_b_input)]
            else:
                set_distr_b=[periodic_correction(distr_b) for distr_b in distr_b_input]
            if b_states is None:    
                set_b_states=[]
                for dim_num in range(len(set_distr_b)):
                    if write_name is not None:
                        plot_name = write_name + '_dist' + str(dim_num)
                    else:
                        plot_name = None
                    try:
                        set_b_states.append(determine_state_limits(set_distr_b[dim_num], gauss_bins, gauss_smooth, write_plots, plot_name))
                    except:    
                        print('Distribution B not clustering properly.\nTry altering Gaussian parameters or input custom states.')
                
            else:
                set_b_states = b_states
                
                    
        H_b=calculate_entropy(set_b_states,set_distr_b)
    
        ab_joint_states= set_a_states + set_b_states
        ab_joint_distributions= set_distr_a + set_distr_b
        
        H_ab=calculate_entropy(ab_joint_states,ab_joint_distributions)
    
        SSI = (H_a + H_b) - H_ab
    except:
        SSI = -1
        if write_name is not None:
            print('WARNING: Input error for ' + write_name)
        else:
            print('WARNING: Input error')
            
        print('Default output of SSI= -1.')
        
    return SSI


#CoSSI = H_a + H_b + H_c - H_ab - H_bc - H_ac + H_abc
def calculate_cossi(distr_a_input, distr_b_input, distr_c_input=None, a_states=None, b_states=None,
                    c_states=None, gauss_bins=120, gauss_smooth=None, write_plots=None,write_name=None):

    
    if gauss_smooth is None:
        gauss_smooth = int(gauss_bins*0.1)        
        
    try:
        if type(distr_a_input[0]) is not list:
            set_distr_a=[periodic_correction(distr_a_input)]
        ## else set_distr_a is a nested list of multiple distributions (bivariate)
        else:
            set_distr_a=[periodic_correction(distr_a) for distr_a in distr_a_input]
        
        if a_states is None:    
            set_a_states=[]
            for dim_num in range(len(set_distr_a)):
                if write_name is not None:
                    plot_name = write_name + '_dist' + str(dim_num)
                else:
                    plot_name = None
                try:
                    set_a_states.append(determine_state_limits(set_distr_a[dim_num], gauss_bins, gauss_smooth, write_plots, plot_name))
                except:
                    print('Distribution A not clustering properly.\nTry altering Gaussian parameters or input custom states.')
        else:
            set_a_states = a_states
            
        H_a=calculate_entropy(set_a_states,set_distr_a) 

        ##----------------
        ##calculating the entropy for set_distr_b
        if type(distr_b_input[0]) is not list:
            set_distr_b=[periodic_correction(distr_b_input)]
        else:
            set_distr_b=[periodic_correction(distr_b) for distr_b in distr_b_input]
        if b_states is None:    
            set_b_states=[]
            for dim_num in range(len(set_distr_b)):
                if write_name is not None:
                    plot_name = write_name + '_dist' + str(dim_num)
                else:
                    plot_name = None
                try:
                    set_b_states.append(determine_state_limits(set_distr_b[dim_num], gauss_bins, gauss_smooth, write_plots, plot_name))
                except:    
                    print('Distribution B not clustering properly.\nTry altering Gaussian parameters or input custom states.')
        else:
            set_b_states = b_states
            
        H_b=calculate_entropy(set_b_states,set_distr_b) 
        
        ##----------------
        ##calculating the entropy for set_distr_c
        ## if no dist (None) then apply the binary dist for two simulations
        if distr_c_input is None:
            H_c=1
            set_distr_c=[[0.5]*int(len(set_distr_a[0])/2) + [1.5]*int(len(set_distr_a[0])/2)]
            set_c_states= [[0,1,2]]  
        else:
            if type(distr_c_input[0]) is not list:
                set_distr_c=[periodic_correction(distr_c_input)]
            else:
                set_distr_c=[periodic_correction(distr_c) for distr_c in distr_c_input]
            if c_states is None:    
                set_c_states=[]
                for dim_num in range(len(set_distr_c)):
                    if write_name is not None:
                        plot_name = write_name + '_dist' + str(dim_num)
                    else:
                        plot_name = None
                    try:
                        set_c_states.append(determine_state_limits(set_distr_c[dim_num], gauss_bins, gauss_smooth, write_plots, plot_name))
                    except:    
                        print('Distribution C not clustering properly.\nTry altering Gaussian parameters or input custom states.')
            
            else:
                set_c_states = c_states
        
        H_c=calculate_entropy(set_c_states,set_distr_c)
    
        ##----------------
        ab_joint_states= set_distr_a + set_b_states
        ab_joint_distributions= set_a_states + set_distr_b
        
        H_ab=calculate_entropy(ab_joint_states,ab_joint_distributions)
        ##----------------
        ac_joint_states= set_distr_a + set_c_states 
        ac_joint_distributions= set_a_states + set_distr_c
        
        H_ac= calculate_entropy(ac_joint_states,ac_joint_distributions)
        ##----------------
        bc_joint_states= set_b_states + set_c_states 
        bc_joint_distributions= set_distr_b + set_distr_c
        
        H_bc= calculate_entropy(bc_joint_states,bc_joint_distributions)
        ##----------------
        abc_joint_states= set_distr_a + set_b_states + set_c_states 
        abc_joint_distributions= set_a_states + set_distr_b + set_distr_c
        
        H_abc=calculate_entropy(abc_joint_states,abc_joint_distributions)    
        
        
        SSI = (H_a + H_b) - H_ab
        coSSI = (H_a + H_b + H_c) - (H_ab + H_ac + H_bc) + H_abc 
        
    except:
        SSI = -1
        coSSI = -1
        
        if write_name is not None:
            print('WARNING: Input error for ' + write_name)
        else:
            print('WARNING: Input error')
            
        print('Default output of -1.')   
        
    return SSI, coSSI

