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

This module was developed to aid coursework related to 
numerical methods for IIQF(Indian Institute of Quantitative Finance).
"""
import numpy as np
import math
import seaborn as sb
import matplotlib.pyplot as plt
import time

sb.set()

np.random.seed(1)

def marsaglia_polar():
    
    while True:
        
        x = (np.random.uniform() * 2) - 1
        y = (np.random.uniform() * 2) - 1
        #z = np.random.uniform()
        s = x * x + y * y
        if s < 1:
            t = math.sqrt((-2) * math.log(s))
            return [x * t/math.sqrt(s), y * t/math.sqrt(s)]

def box_muller():
    
    u1 = np.random.uniform()
    u2 = np.random.uniform()

    t = math.sqrt((-2) * math.log(u1))
    v = 2 * math.pi * u2
    
    return [t * math.cos(v), t * math.sin(v)]

def box_muller_vectorised(n=1):
    
    u1 = np.random.rand(n)
    u2 = np.random.rand(n)

    t = np.sqrt((-2) * np.log(u1))
    v = 2 * math.pi * u2
    
    return [t * np.cos(v), t * np.sin(v)]


def marsaglia_vectorized(n=1):
    
    def accept_reject():
        
        while True:
            x = (np.random.uniform() * 2) - 1
            y = (np.random.uniform() * 2) - 1
            s = x * x + y * y
            if s < 1:
                return [x, y]
    
    w1 = np.zeros(n)
    w2 = np.zeros(n)
    
    for i in range(n):
        w1[i], w2[i] = accept_reject()
    
    s = w1**2 + w2**2
    t = np.sqrt(-2*np.divide(np.log(s), s))
    
    z1 = w1*t
    z2 = w2*t
    
    return [z1, z2]


def marsaglia_bray(n=1):
    
    p = math.pi/4; aux = p*(1-p)
    
    x = (3*math.sqrt(aux) + math.sqrt(9*aux + p*n))/p
    N = math.ceil(x*x)
        
    w1 = np.random.rand(N) * 2 - 1
    w2 = np.random.rand(N) * 2 - 1
    s = w1 * w1 + w2 * w2
    
    index = s<1
            
    w1 = w1[index][:n]
    w2 = w2[index][:n]
    s = s[index][:n]
        
    t = np.sqrt(-2*np.divide(np.log(s), s))
    
    z1 = w1*t
    z2 = w2*t
        
    return [z1, z2]

def main():
    
    print ()

    start = time.time()
    w = []
    for i in range(10000):
        w.append(box_muller()[0])
    end = time.time()
    print ("Box-Muller method took %0.2f ms" % ((end - start) * 1000.0))
    
    start = time.time()
    r = []
    for i in range(10000):
        r.append(marsaglia_polar()[0])
    end = time.time()
    print ("Marsaglia-Polar method took %0.2f ms" % ((end - start) * 1000.0))
    
    start = time.time()
    v = marsaglia_vectorized(10000)[0]
    end = time.time()
    print ("Marsaglia-vectorized method took %0.2f ms" % ((end - start) * 1000.0))
    
    start = time.time()
    t = marsaglia_bray(10000)[0]
    end = time.time()
    print ("Marsaglia-bray method took %0.2f ms" % ((end - start) * 1000.0))
    
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(5,5))
    
    sb.distplot(w, axlabel='Box-Muller', color='blue', ax=ax[0,0])
    sb.distplot(r, axlabel='Polar', color='yellow', ax=ax[0,1])
    sb.distplot(v, axlabel='Marsaglia-vectorized', color='red', ax=ax[1,0])
    sb.distplot(t, axlabel='Marsaglia-bray', color='green', ax=ax[1,1])
    
    plt.tight_layout()
    plt.show()
 

if __name__ == '__main__':
    
    main()
    
    