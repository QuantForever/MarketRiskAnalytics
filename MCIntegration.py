#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 16:46:49 2019
Copyright (C) [2019] by [Srijoy Das]
Permission is hereby granted, free of charge, to any person obtaining 
a copy of this software and associated documentation files (the "Software"), 
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, 
but not sell copies of the Software for academic purposes only.

The MC Integration is used to approximate the value of the integral:
    f = -x**2 + x : 0 <= x <= 1.
Monte Carlo Integration:  1) Volume estimation/"Hit or Miss" method
                          2) Sample-Mean estimation method
                          3) Using quasi monte carlo (Sobol sequence)
"""

from scipy import integrate
import numpy as np
import QuasiMC as lds
import matplotlib.pyplot as plt

np.random.seed(1)

#func = lambda x: np.exp(x)
#func = lambda x: x**2
func = lambda x: -x**2 + x

""" 
Using numerical quadrature
that are elementary versions such as the
trapezoidal and Simpson's rules 
"""
def numerical_integral():
    val, abs_err = integrate.quad(func, 0, 1)
    return val

def volume_method(pts_x, pts_y, n, vol):   
    accept_ratio = np.sum(pts_y < func(pts_x))/n
    sol = vol * accept_ratio
    return sol

def sample_mean_method(pts_x, n):
    pts_y = func(pts_x)
    sol = sum(pts_y)/n
    return sol

def main():

   print ('\nActual value is: %.5f' % numerical_integral())
   
   # Monte Carlo approximation
   
   print ('\nSample-Mean method\n')
   
   for n in 10**np.array([2, 4, 6]):
       #Simulate the uniform r.v
       pts_x = np.random.uniform(0, 1, n)
       sol = sample_mean_method(pts_x, n)
       print ('%10d %.5f' % (n, sol))
   
   print ('\nVolume method\n')
   
   c_max = np.max([func(i) for i in np.arange(0,1.01,0.01)])
   
   for n in 10**np.array([2, 4, 6]):
       #Simulate the uniform r.v
       pts_x = np.random.uniform(0, 1, n)
       pts_y = np.random.uniform(0, 1, n)*c_max
       sol =   volume_method(pts_x, pts_y, n, c_max*1.0)
       print ('%10d %.5f' % (n, sol))

   # Simulate the uniform r.v using quasi MC (low discrepancy)
   print ('\nVolume method using Sobol sequence\n')
   
   for n in 10**np.array([2, 3, 4, 5]):
       sobols = lds.ql_sobol_generate_uniform(2,n)
       pts_x = sobols[:,0] ; pts_y = sobols[:,1]*c_max
       sol = volume_method(pts_x, pts_y, n, c_max*1.0)
       print ('%10d %.5f' % (n, sol))
   
   # Plot the distributions
   x = np.linspace(0,1,101)
   fig, axs = plt.subplots(1, 2, figsize=(6,4))
   
   normals = np.random.uniform(0, 1, (2, 10**3))
   mc_x = normals[0, :]
   mc_y = normals[1, :] * c_max
   points_under = [True if mc_y[i] <= func(mc_x[i]) else False 
                for i in range(len(mc_x))]
   
   axs[0].plot(x, func(x), linewidth=2, c='k')
   axs[0].set_title('using numpy.random')
   axs[0].set_xlim(0,1)
   axs[0].set_ylim(0,1)
   axs[0].scatter(mc_x[points_under], mc_y[points_under],
           c='r', s=15)
   axs[0].scatter(mc_x[np.logical_not(points_under)], 
            mc_y[np.logical_not(points_under)], 
            c='b', s=15)
   
   
   sobols = lds.ql_sobol_generate_uniform(2,10**3)
   mc_x = sobols[:, 0]
   mc_y = sobols[:, 1] * c_max
   points_under = [True if mc_y[i] <= func(mc_x[i]) else False 
                for i in range(len(mc_x))]
   
   axs[1].plot(x, func(x), linewidth=2, c='k')
   axs[1].set_title('using sobol sequence',)
   axs[1].set_xlim(0,1)
   axs[1].set_ylim(0,1)
   axs[1].scatter(mc_x[points_under], mc_y[points_under],
           c='r', s=15)
   axs[1].scatter(mc_x[np.logical_not(points_under)], 
            mc_y[np.logical_not(points_under)], 
            c='b', s=15)
   
   plt.show()
   
   
if __name__ == '__main__':
    main()