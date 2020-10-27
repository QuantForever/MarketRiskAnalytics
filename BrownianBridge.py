#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 16:38:21 2020

References: Robert Sedgewick, Kevin Wayne, and Robert Dondero.

The value of H determines what kind of process the fBm is:

if H = 1/2 then the process is in fact a Brownian motion or Wiener process
if H > 1/2 then the increments of the process are positively correlated
if H < 1/2 then the increments of the process are negatively correlated


Generate brownian bridge in two ways:
    a) using the SDE form
    b) using the interpolation technique

"""

import matplotlib.pyplot as plt
import math
import numpy as np

np.random.seed(4)

# Uses the SDE form of BB to generate paths

def bb_sde_method(M, N, a=0.0, b=0.0):
    
    dt = 1.0 / N; t = 0.0                                                         
    B = np.zeros((M, N+1), dtype=np.float32)
    T = np.zeros(N+1, dtype=np.float32)
    B[:, 0] = a
    
    for n in range(1, N):
         t += dt
         z = np.random.normal(0, math.sqrt(dt), M)
         B[:, n] = B[:, n-1] + ((b - B[:, n-1]) * (dt / (1 - t))) + z
         T[n] = t
         
    B[:, -1] = b
    T[-1] = 1.0
    
    return [B, T]                                                            

# Uses the interpolation form of BB to generate paths and uses recursion
# parameter k is stopping criterion (i.e. after k partitions)
    
def bb_interpolate_method(x0, y0, x1, y1, variance, k, path, t):
 
    if (x1 - x0) <= 2**(-k):
        #plt.plot([x0, x1], [y0, y1])
        return 
    
    xm = (x0 + x1) / 2.0
    ym = (y0 + y1) / 2.0
    
    delta = np.random.normal(0, math.sqrt(variance))
    
    path.append(ym + delta); t.append(xm)
    
    bb_interpolate_method(x0, y0, xm, ym+delta, variance/2.0, k, path, t)
    
    bb_interpolate_method(xm, ym+delta, x1, y1, variance/2.0, k, path, t)

    return

def main():
    
    # Generate generalized bb paths from t=0 to 1.0
    bb_paths, t = bb_sde_method(10, 50, 0.0, 0.0)
    
    plt.title ("brownian-bridge path by SDE method")
    plt.plot(t, bb_paths.T)
    plt.show()
    
    # Generate interpolated bb paths from x=0 to 1.0
    k = 3; x0 = 0; xT = 1.0; y0 = 0.0; yT = 0.0; variance = (xT-x0)/2.0
    
    path = []; t = []
    
    bb_interpolate_method(x0, y0, xT, yT, variance/2.0, k, path, t)
    
    sorted_time = [x0]; sorted_path = [y0]
    
    for t_,x_ in sorted(zip(t,path)):
        sorted_time.append(t_)
        sorted_path.append(x_)
    
    sorted_time.append(xT); sorted_path.append(yT)
    
    plt.title ("brownian-bridge path by interpolation")
    plt.plot(sorted_time, sorted_path, color='red', marker='.')
    plt.show()

if __name__ == '__main__':

   main()

