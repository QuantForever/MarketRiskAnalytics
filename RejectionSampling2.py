#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 16:02:44 2019
@author: Srijoy
Copyright (C) [2019] by [Srijoy Das]
Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, 
but not sell copies of the Software for academic purposes only.

References: https://wiseodd.github.io/techblog/2015/10/21/rejection-sampling/

This module was developed to aid coursework related to 
numerical methods for IIQF(Indian Institute of Quantitative Finance).
"""

import numpy as np
import scipy.stats as st
import seaborn as sb
import matplotlib.pyplot as plt

sb.set()

def target(x):
    return st.beta.pdf(x, 2, 2)

def rejection_sampling(n=100, k=1.0):
    samples = []
    for i in range(n):
        z = np.random.uniform(0, 1)
        u = np.random.uniform(0, 1)*k
        if u <= target(z):
            samples.append(z)

    return np.array(samples)
   
def main():
    
    k = max(target(np.arange(0, 1+0.01, 0.01)))
    
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5,5))
    
    x = np.arange(0, 1+0.01, 0.01)
    
    ax[0].plot(x, target(x))
    ax[0].plot(x, [k for item in x])
    ax[0].set_title("Rejection-sampling of Beta distribution")
    
    s = rejection_sampling(10000, k)
    sb.distplot(s, color='red', ax=ax[1])
    plt.tight_layout()
    plt.show()

    print("\nvalue of k: %0.4f" %k)
    print("acceptance percentage is: %0.2f" %(100*len(s)/10000))
    
    
if __name__ == '__main__':
    
    main()
    
    