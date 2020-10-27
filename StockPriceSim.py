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

Topic:  a) Stock price simulation and forecasting (using brownian bridge)
        b) Price an European call and Asian call using crude MC
        c) Price an European call and Asian call using control variates
        d) Price an European call and Asian call using Sobol (QMC)

This module was developed to aid coursework related to 
numerical methods for IIQF(Indian Institute of Quantitative Finance).
"""

from scipy.stats import norm 
#from pandas_datareader import data
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import QuasiMC as mc
from BrownianBridge import bb_interpolate_method

np.random.seed(1)

def BSClosedForm(S0, K, r, T, sigma, is_call):
    # call or put
    d1 = ((r + 0.5 * sigma**2) * T + np.log(S0 /K )) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    p = 0.0
    
    if  is_call:
        p =  S0 * norm.cdf(d1) - np.exp(- r * T) *  K * norm.cdf(d2) 
    else:               
        p = np.exp(- r * T) *  K * norm.cdf(-d2) -  S0 * norm.cdf(-d1)
        
    return p

def daily_return(adj_close):
    returns = []
    for i in range(0, len(adj_close)-1):
        today = adj_close[i+1]
        yesterday = adj_close[i]
        daily_return = (today - yesterday)/yesterday
        returns.append(daily_return)
    return returns


def GBM_BB(So, Sf, mu, sigma, T, k=8):    

    path = [So]; t = [0]; sorted_path = [So]
    
    sorted_time = []; sorted_bb_path = []; 
    
    b_f = (np.log(Sf/So) - (mu - 0.5*sigma**2)*T)/sigma
    
    bb_interpolate_method(0, 0, T, b_f, T/4.0, k, path, t)
    
    for t_,x_ in sorted(zip(t,path)):
        sorted_time.append(t_)
        sorted_bb_path.append(x_)
    
    for i in range(1, len(sorted_time)):
        s_ = So*np.exp((mu - 0.5*sigma**2)*sorted_time[i] + sigma*sorted_bb_path[i])
        sorted_path.append(s_)

    sorted_time.append(T)
    sorted_path.append(Sf)
    
    return sorted_path, sorted_time


def GBM(So, mu, sigma, T, N, normal, num_sim):    
    
    dt = float(T/N)
    paths = np.zeros((num_sim, N+1))
    paths [:,0] = So
    for i in np.arange(1, int(N+1)):
        drift = (mu-0.5*sigma**2)*dt
        diffusion = sigma * np.sqrt(dt)
        paths[:,i] = paths[:,i-1] * np.exp(drift + diffusion*normal[:,i-1])

    return paths


def payoff(S_T, Strike, rate, Mat=1.0):
    payoff = np.exp(-rate*Mat) * max(S_T - Strike, 0)
    return payoff


def control_variates(c, co, f):
    lambd = np.mean(np.multiply(c-np.mean(c), f-np.mean(f)))/np.var(c)
    f2 = f - lambd * (c - co)
    return f2


def main():
    
    # We would like all available data from start to end date
    #start_date = '2019-01-01'
    #end_date = '2019-12-31'

    # User pandas_reader.data.DataReader to load the desired data. As simple as that.
    #panel_data = data.DataReader('GOOG', 'yahoo', start_date, end_date)
    
    panel_data = pd.read_csv('GOOG.csv', header=0, index_col=0)
    adj_close = panel_data['Close']
    
    returns = daily_return(adj_close)

    mu = np.mean(returns)*252.0          # drift coefficient
    sig = np.std(returns)*np.sqrt(252.0)  # diffusion coefficient
    #print (mu, sig)
   
    So = adj_close[0]; Sf = adj_close[-1]
    Strike = 0.9*So # Moneyness
    rate = 0.05
    T = 1.0
    N = 252
    Asian_N = 60 #Last 60 days average

    fig, ax = plt.subplots()
    time = np.linspace(0, T, N+1)
    ax.plot(time, adj_close, label = 'Actual', color='k')
    #forecasted_ts, t = GBM(So, mu, sig, T, N)      
    forecasted_ts, t = GBM_BB(So, Sf, mu, sig, T, 8) # K=8 -- 256 steps
    ax.plot(t, forecasted_ts, label = 'GBM-Brownian Bridge', ls='--', color='orange')

    ax.set_ylabel('GOOG Stock Price, $')
    ax.set_xlabel('time (yrs)')
    ax.set_title('Geometric Brownian Motion - GOOG')
    ax.legend(loc = 'upper left')
    plt.show()

    # Price an european call and asian call under risk-neutral measure
    # strike (moneyness) = 0.9
    
    price = BSClosedForm(So, Strike, rate, T, sig, True)
    print('\nAnalytical price of european option is %.2f\n' %(price))
    
    v1 = []; 
    
    steps = 250; num_sim = 1000;
    
    normals = np.random.normal(0, 1, (num_sim, steps))
    
    paths = GBM(So, rate, sig, T, steps, normals, num_sim)
    
    for i in np.arange(num_sim):
        v1.append(payoff(paths[i,-1], Strike, rate, T))
    
    price = np.mean(v1)
    
    print ('\nMC price of call on GOOG is: %.2f \n' % price)
    mc_err = np.std(v1)/np.sqrt(num_sim)
    print ('[Sampling err is: %.4f [N = %d]] \n' %(mc_err, num_sim))
    
    """European option pricing with control variate - 
       discounted stock price (which is a martingale) 
    """ 
    S_T = paths[:,-1]*np.exp(-rate*T)
    
    f1 = control_variates(S_T, So, np.array(v1))
    
    price = np.mean(f1)
    print ('\nMC price using control variate is: %.2f \n' % price)
    mc_err = np.std(f1)/np.sqrt(num_sim)
    print ('[Sampling err (using control variate): %.4f [N = %d]] \n' %(mc_err, num_sim))
    
    #Generate the random numbers using quasi monte-carlo and compare.
    steps = 120;
    
    sobols = mc.ql_sobol_generate_std_normal(steps, num_sim)
    #Expanded the sobol sequence by bb technique (quantlib)
    paths2 = GBM(So, rate, sig, T, steps, sobols, num_sim)
    
    v2 = [];
    for i in np.arange(num_sim):
        v2.append(payoff(paths2[i,-1], Strike, rate, T))

    call_price = np.mean(v2)
    print ('\nMC price using Sobol: %.2f \n' % call_price) 
    mc_err = np.std(v2)/np.sqrt(num_sim)
    print ('[Sampling err is: %.4f [M, N] = [%d, %d]] \n' %(mc_err, steps, num_sim))
    
    v3 = [];
    for i in np.arange(num_sim):
        v3.append(payoff(np.mean(paths[i,-Asian_N:]), Strike, rate, T))
    
    f3 = control_variates(S_T, So, np.array(v3))
    
    asian_price = np.mean(f3)
    print ('\nMC price of Asian call on GOOG is: %.2f  \n' % asian_price)
    mc_err = np.std(f3)/np.sqrt(num_sim)
    print ('[Sampling err is: %.4f [N = %d]] \n' %(mc_err, num_sim))
    
    # Plot convergence results [TBD]
    
if __name__ == "__main__":
    
    main()
    
    
    