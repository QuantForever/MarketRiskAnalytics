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

Problem: Suppose we wish to estimate the value of a call option 
using Monte Carlo methods which is well “out of the money”, 
one with a strike price K far above the
current price of the stock S0.

So that if we were to attempt to evaluate this
option using crude Monte Carlo, 
the majority of the terminal stock price values randomly generated
would fall below the strike (K)
and contribute zero to the option price. 

Solution: One possible remedy is to use 
importance sampling by adjusting the mean:
mu = log(K/So), as demonstrated below.

This module was developed to aid coursework related to 
numerical methods for IIQF(Indian Institute of Quantitative Finance).
"""

import pandas as pd
import numpy as np
from StockPriceSim import BSClosedForm, GBM, payoff, daily_return
from scipy.stats import norm
import seaborn as sb
import matplotlib.pyplot as plt

sb.set()

np.random.seed(1)

def main():
    
    panel_data = pd.read_csv('GOOG.csv', header=0, index_col=0)
    
    adj_close = panel_data['Close']
    
    returns = daily_return(adj_close)

    #mu = np.mean(returns)*252.0          # drift coefficient
    sig = np.std(returns)*np.sqrt(252.0)  # diffusion coefficient
    #print (mu, sig)
   
    So = adj_close[0]; 
    Strike = 1.5*So # Moneyness
    rate = 0.02
    T = 1.0
   
    # Price an OTM european call using importance sampling
    # strike (moneyness) = 1.5
    
    print ('\nSpot price: %.2f and Strike: %0.2f' % (So, Strike))
    price = BSClosedForm(So, Strike, rate, T, sig, True)
    print('\nAnalytical price of OTM option is %.2f' %(price))
    
    v1 = []; 
    
    steps = 250; num_sim = 100;
    
    normals = np.random.normal(0, 1, (num_sim, steps))
    
    paths = GBM(So, rate, sig, T, steps, normals, num_sim)
    
    for i in np.arange(num_sim):
        v1.append(payoff(paths[i,-1], Strike, rate, T))
    
    price = np.mean(v1)
    
    print ('\nMC price is: %.2f \n' % price)
    mc_err = np.std(v1)/np.sqrt(num_sim)
    print ('[Sampling err is: %.4f [N = %d]] \n' %(mc_err, num_sim))
    
    """deep OTM European option pricing with importance sampling
       estimate the radon-nikodyn derivative (lambd)
    """
    v1 = []
    
    mu = np.log(Strike/So)
    
    paths2 = GBM(So, mu, sig, T, steps, normals, num_sim)
    
    for i in np.arange(num_sim):
        v1.append(payoff(paths2[i,-1], Strike, rate, T))
    
    z = np.log(paths2[:,-1]/So)
    
    lambd = np.divide(norm.pdf(z, (rate - 0.5*(sig*sig))*T, sig*(T**0.5) ), \
                norm.pdf(z, (np.log(Strike/So) - 0.5*(sig*sig))*T, sig*(T**0.5) ))
                
    v2 = np.multiply(v1, lambd)
    
    price = np.mean(v2)
    print ('\nMC price using Importance Sampling: %.2f \n' % price) 
    mc_err = np.std(v2)/np.sqrt(num_sim)
    print ('[Sampling err is: %.4f [N = %d]] \n' %(mc_err, num_sim))
    
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
    
    sb.distplot(paths[:,-1], label='p1(x): mu=rate', color='blue', ax=ax)
    sb.distplot(paths2[:, -1], label='p2(x): mu=log(K/So)', color='red', ax=ax)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
if __name__ == "__main__":
    main()