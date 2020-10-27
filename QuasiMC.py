#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 16:49:55 2019
@author: Srijoy
Copyright (C) [2019] by [Srijoy Das]
Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, 
but not sell copies of the Software for academic purposes only.

note: sobol_seq.i4_sobol_generate Generates multivariate standard normal
      quasi-random variables using sobol_seq library.
    Parameters:
      Input, integer dim_num is the dimension.
      Input, integer n, the number of points to generate.
      Output, real np array of shape (n, dim_num).

This module was developed to aid coursework related to 
numerical methods for IIQF(Indian Institute of Quantitative Finance).
"""

import QuantLib as ql
import sobol_seq
import numpy as np
from scipy.stats import norm
from matplotlib import pyplot as plt

def ql_sobol_generate_std_normal(dim_num, n, seed=0):
    
    """
    ql_sobol_generate_std_normal generates multivariate 
    standard normal quasi-random variables
    using quantlib
    
    """
    
    sbl = ql.SobolRsg(dim_num, seed)
    
    sobols = np.zeros((n, dim_num), dtype=np.float64)
    
    for i in range(n):
        
        x = sbl.nextSequence().value()
        normals = norm.ppf(x)
        sobols[i,:] = normals
    
    return sobols

def ql_sobol_generate_uniform(dim_num, n, seed=0):
    
    """
    ql_sobol_generate_uniform generates multivariate 
    uniform (0,1) quasi-random variables using 
    quantlib
    
    """
    
    sbl = ql.SobolRsg(dim_num, seed)
    
    sobols = np.zeros((n, dim_num), dtype=np.float64)
    
    for i in range(n):
        
        sobols[i,:] = sbl.nextSequence().value()
    
    return sobols


def i4_sobol_generate_uniform(dim_num, n):
    sobols = sobol_seq.i4_sobol_generate(dim_num, n)
    return sobols


def i4_sobol_generate_std_normal(dim_num, n):
    sobols = sobol_seq.i4_sobol_generate(dim_num, n)
    normals = norm.ppf(sobols)
    return normals

def main():
    
    np.random.seed(6345345)
    # make 2000 random numbers and use half as X coordinate
    # and the other half as Y coordinate for 1000 points
    X = np.random.uniform(size=(2000,1))
    
    #sobol_X = i4_sobol_generate_uniform(2, 500)
    sobols = ql_sobol_generate_uniform(2, 1000)
    
    # Plot the distributions.
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10,5))
    ax1.scatter(X[:1000], X[1000:], color="blue")
    ax2.scatter(sobols[:,0], sobols[:,1], color="red")
  
    ax1.set_title("Random")
    ax2.set_title("Sobol")
    plt.show()
    
if __name__ == "__main__":
    main()
    
    