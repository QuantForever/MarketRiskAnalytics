#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 15:33:59 2019
@author: Srijoy
Copyright (C) [2019] by [Srijoy Das]
Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, 
but not sell copies of the Software for academic purposes only.

This module was developed to aid coursework related to 
numerical methods for IIQF(Indian Institute of Quantitative Finance).
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()
np.random.seed(1)

def Cholesky(cor_matrix):
    Chol = np.linalg.cholesky(cov_matrix)
    return Chol

def Gen_mvn(dim, no_obs):
     obs = np.random.normal(0, 1, size=(dim, no_obs)) 
     return obs

if __name__ == '__main__':

    # Number of observations per column
    no_obs = 1000     
    no_cols = 3         
    # Mean values of each column
    # Number of columns
    means = [1, 2, 3]         
    sds =   [1, 2, 3]         
    sd = np.diag(sds)
    # The correlation matrix [3 x 3]
    cor_matrix = np.array([[1.0, -0.6, -0.9],
                           [-0.6, 1.0, 0.5],
                           [-0.9, 0.5, 1.0]])    
    
    # Multi-dimensional normal random vector draws N(0,1) in [3 x 1,000]
    observations = Gen_mvn(no_cols, no_obs)
    
    # Create the covariance matrix
    cov_matrix = np.dot(sd, np.dot(cor_matrix, sd))   
    
    # Cholesky factorisation
    A = Cholesky(cor_matrix)                         
    
    # Generating random MVN (0, cov_matrix)
    x = np.dot(A, observations)               
    s = x.transpose() + means               
    transforms = s.transpose()                          
    
    # Checking correlation consistency.
    np.set_printoptions(precision=2)
    print ('\n reproduce the correlation matrix:\n')
    print(np.corrcoef(transforms))                       
    
    data = observations
    g = sns.jointplot(data[0,:], data[1,:], height=4, color='r')
    g.fig.suptitle("Independent normals [X, Y]", fontsize=12)
    plt.show()
    
    data = transforms
    g = sns.jointplot(data[0,:], data[1,:], height=4, color='g')
    g.fig.suptitle("Correlated normals [X, Y] from Cholesky Decomp.", fontsize=12)
    
    plt.tight_layout()
    plt.show()