import numpy as np
import math
import os
from pensa.features import *
from .discrete_states import *


# -- Functions to calculate State Specific Information quantities --


def _check(value,x,y):
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
                limit_occupancy_checks[dist_num][frame_num]= _check(distribution[frame_num],limits[0],limits[1]) 
        mut_prob[it.multi_index]= sum(np.prod(limit_occupancy_checks,axis=0)) / len(limit_occupancy_checks[0])
        ##calculating the entropy as the summation of all -p*log(p) 
        
        if mut_prob[it.multi_index] != 0:
            entropy+=-1*mut_prob[it.multi_index]*math.log(mut_prob[it.multi_index],2)
        it.iternext()
    return entropy


def calculate_ssi(distr_a_input, distr_b_input=None, a_states=None, b_states=None,
                  gauss_bins=120, gauss_smooth=None, pbc=True, write_plots=None, write_name=None):
    """
    Calculates the State Specific Information SSI [bits] between two features from two ensembles. 
    By default, the second feature is the binary switch between ensembles.

    SSI(a,b) = H_a + H_b H_ab
    H = Conformational state entropy

    Parameters
    ----------
    distr_a_input : list of lists
        A list containing multivariate distributions (lists) for a particular
        residue or water
    distr_b_input : list of lists, optional
        A list containing multivariate distributions (lists) for a particular
        residue or water. The default is None and a binary switch is assigned.
    a_states : list of lists, optional
        A list of values that represent the limits of each state for each
        distribution. The default is None and state limits are calculated automatically.
    b_states : list of lists, optional
        A list of values that represent the limits of each state for each
        distribution. The default is None and state limits are calculated automatically.
    gauss_bins : int, optional
        Number of histogram bins to assign for the clustering algorithm. 
        The default is 120.
    gauss_smooth : int, optional
        Number of bins to perform smoothing over. The default is ~10% of gauss_bins.
    write_plots : bool, optional
        If true, visualise the states over the raw distribution. The default is None.
    write_name : str, optional
        Filename for write_plots. The default is None.

    Returns
    -------
    SSI : float
        State Specific Information (SSI[bits]) shared between input a and input b (default is binary switch).

    """
    
    if gauss_smooth is None:
        gauss_smooth = int(gauss_bins*0.1)
    
    #try:
    if True:       
        ##calculating the entropy for set_distr_a
        ## if set_distr_a only contains one distributions
        if pbc is True:
            if type(distr_a_input[0]) is not list:
                set_distr_a=[correct_angle_periodicity(distr_a_input)]
            ## else set_distr_a is a nested list of multiple distributions (bivariate)
            else:
                set_distr_a=[correct_angle_periodicity(distr_a) for distr_a in distr_a_input]
        else:
            set_distr_a=distr_a_input        
        
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
            if pbc is True:
                if type(distr_b_input[0]) is not list:
                    set_distr_b=[correct_angle_periodicity(distr_b_input)]
                else:
                    set_distr_b=[correct_angle_periodicity(distr_b) for distr_b in distr_b_input]
            else:
                set_distr_b=distr_b_input
                
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
    else:   
    #except:
        SSI = -1
        if write_name is not None:
            print('WARNING: Input error for ' + write_name)
        else:
            print('WARNING: Input error')
            
        print('Default output of SSI= -1.')
        
    return round(SSI,4)


def calculate_cossi(distr_a_input, distr_b_input, distr_c_input=None, a_states=None, b_states=None,
                    c_states=None, gauss_bins=120, gauss_smooth=None, write_plots=None,write_name=None):
    """
    Calculates the State Specific Information Co-SSI [bits] between three features from two ensembles. 
    By default, the third feature is the binary switch between ensembles.
    
    CoSSI(a,b,c) = H_a + H_b + H_c - H_ab - H_bc - H_ac + H_abc
    
    H = Conformational state entropy
    
    Parameters
    ----------
        

    distr_a_input : list of lists
        A list containing multivariate distributions (lists) for a particular
        residue or water
    distr_b_input : list of lists
        A list containing multivariate distributions (lists) for a particular
        residue or water. 
    distr_c_input : list of lists, optional
        A list containing multivariate distributions (lists) for a particular
        residue or water. The default is None and a binary switch is assigned.
    a_states : list of lists, optional
        A list of values that represent the limits of each state for each
        distribution. The default is None and state limits are calculated automatically.
    b_states : list of lists, optional
        A list of values that represent the limits of each state for each
        distribution. The default is None and state limits are calculated automatically.
    c_states : list of lists, optional
        A list of values that represent the limits of each state for each
        distribution. The default is None and state limits are calculated automatically.
    gauss_bins : int, optional
        Number of histogram bins to assign for the clustering algorithm. 
        The default is 120.
    gauss_smooth : int, optional
        Number of bins to perform smoothing over. The default is ~10% of gauss_bins.
    write_plots : bool, optional
        If true, visualise the states over the raw distribution. The default is None.
    write_name : str, optional
        Filename for write_plots. The default is None.

    Returns
    -------
    SSI : float
        SSI[bits] shared between input a and input b (default is binary switch).
    coSSI : float
        Co-SSI[bits] shared between input a, input b and input c (default is binary switch).

    """

    if gauss_smooth is None:
        gauss_smooth = int(gauss_bins*0.1)              
 
    try:       
        ##calculating the entropy for set_distr_a
        ## if set_distr_a only contains one distributions
        if type(distr_a_input[0]) is not list:
            set_distr_a=[correct_angle_periodicity(distr_a_input)]
        ## else set_distr_a is a nested list of multiple distributions (bivariate)
        else:
            set_distr_a=[correct_angle_periodicity(distr_a) for distr_a in distr_a_input]
        
        if a_states is None:    
            set_a_states=[]
            for dim_num in range(len(set_distr_a)):
                if write_name is not None:
                    plot_name = write_name + '_dist' + str(dim_num)
                else:
                    plot_name = None
                try:
                    set_a_states.append(determine_state_limits(set_distr_a[dim_num], 
                                                               gauss_bins, 
                                                               gauss_smooth, 
                                                               write_plots, 
                                                               plot_name))
                except:
                    print('Distribution A not clustering properly.\nTry altering Gaussian parameters or input custom states.')
        else:
            set_a_states = a_states
            
        H_a=calculate_entropy(set_a_states,set_distr_a)     

        ##----------------
        ##calculating the entropy for set_distr_b
        if type(distr_b_input[0]) is not list:
            set_distr_b=[correct_angle_periodicity(distr_b_input)]
        ## else set_distr_b is a nested list of multiple distributions (bivariate)
        else:
            set_distr_b=[correct_angle_periodicity(distr_b) for distr_b in distr_b_input]
        
        if b_states is None:    
            set_b_states=[]
            for dim_num in range(len(set_distr_b)):
                if write_name is not None:
                    plot_name = write_name + '_dist' + str(dim_num)
                else:
                    plot_name = None
                try:
                    set_b_states.append(determine_state_limits(set_distr_b[dim_num],
                                                               gauss_bins, 
                                                               gauss_smooth, 
                                                               write_plots, 
                                                               plot_name))
                except:
                    print('Distribution A not clustering properly.\nTry altering Gaussian parameters or input custom states.')
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
                set_distr_c=[correct_angle_periodicity(distr_c_input)]
            else:
                set_distr_c=[correct_angle_periodicity(distr_c) for distr_c in distr_c_input]
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
        ab_joint_states = set_a_states + set_b_states
        ab_joint_distributions = set_distr_a + set_distr_b
        
        H_ab = calculate_entropy(ab_joint_states, ab_joint_distributions)
        ##----------------
        ac_joint_states =  set_a_states + set_c_states 
        ac_joint_distributions = set_distr_a + set_distr_c
        
        H_ac = calculate_entropy(ac_joint_states, ac_joint_distributions)
        ##----------------
        bc_joint_states = set_b_states + set_c_states 
        bc_joint_distributions = set_distr_b + set_distr_c
        
        H_bc = calculate_entropy(bc_joint_states, bc_joint_distributions)
        ##----------------
        abc_joint_states = set_a_states + set_b_states + set_c_states 
        abc_joint_distributions = set_distr_a + set_distr_b + set_distr_c
        
        H_abc = calculate_entropy(abc_joint_states, abc_joint_distributions)    
               
        SSI = (H_a + H_b) - H_ab
        coSSI = (H_a + H_b + H_c) - (H_ab + H_ac + H_bc) + H_abc 
        ##conditional mutual info for sanity check
        # con_mut_inf = H_ac + H_bc - H_c - H_abc
        
    except:
        SSI = -1
        coSSI = -1
        
        if write_name is not None:
            print('WARNING: Error for ' + write_name)
        else:
            print('WARNING: Error')
            
        print('Default output of -1.')   
        
    return round(SSI,4), round(coSSI,4)


