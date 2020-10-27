#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 02:27:43 2020
@author: Srijoy

RANDU is a linear congruential pseudorandom number generator (LCG) 
of the Park–Miller type, which has been used since the 1960s.
It is defined by the recurrence:
    
    V(i+1) = 65539 * V(i) mod 2**31   
"""

import matplotlib.pyplot as plt 
import seaborn as sb
import numpy as np

sb.set()

def main():
    
    seed = 2
    x_i = [seed]
    a = 65539 #multiplier
    m = 2**31
    size = 1000
    
    for k in range(size):
        x_next = (a * x_i[k]) % m
        x_i.append(x_next)
    
    u_i = np.array(x_i)/m
        
    fig, ax = plt.subplots(nrows=2, figsize=(5,5))
    
    ax[0].scatter(x_i[:-1], x_i[1:], color="blue")
    ax[0].set_xlabel("X(n-1)"); ax[0].set_ylabel("X(n)")
    ax[0].set_title("Spectral Test-RANDU (m=2^31 a = 65539)")
    
    sb.distplot(u_i, color='green', ax=ax[1])
    ax[1].set_xlim(0.0, 1.0)
    ax[1].set_title("Uniform Distribution Plot")
    
    plt.tight_layout()
    plt.show()
    
if __name__ == '__main__':
    
    main()
    

