# library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from math import *
from scipy.optimize import fmin_bfgs
#from mpl_toolkits.mplot3d import axes3d  

# Importation
import os
os.getcwd()
os.chdir("C:\\Users\\Halem\\Desktop\\PFE\\doc_important")

pd.set_option("display.max_row",5)
pd.set_option("display.max_column",60)
option_book = pd.read_excel("OptionBook.xlsx") 
option = option_book.copy()
# option.shape 
# option.dtypes
# option.dtypes.value_counts()
# option.isna().sum()  missing value


# class blackscholes

class BlackScholes:
    
        
    def bs_pricing(self, Option_type,S,K,T,r,rf,sigma):
        
        global d1
        global d2
        d1 = (log(S/K) + (r - rf + sigma**2/2)*T) / sigma*sqrt(T)
        d2 = d1 - sigma* sqrt(T)
        
        if Option_type == "Call":
            option_price = S* exp(-rf*T)* norm.cdf(d1) - K * exp(-r*T)*norm.cdf(d2)
            return option_price
        elif Option_type == "Put":
            option_price = K* exp(-r*T)* norm.cdf(-d2) - S * exp(-rf*T)*norm.cdf(-d1)
            return option_price
        else:
            return "incorrect parameters"
       
   
    def delta(self, Option_type,S,K,T,r,rf,sigma):
        if Option_type == "Call":
            delta = exp(-rf*T) * norm.cdf(d1)   
        elif Option_type == "Put":
            delta = exp(-rf*T) * (norm.cdf(d1)-1)
        return delta
    
    
    def gamma(self, Option_type,S,K,T,r,rf,sigma):
        return (norm.pdf(d1)*exp(-rf*T) )/ ( S*sigma*sqrt(T) )
    
    def vega(self, Option_type,S,K,T,r,rf,sigma):
        return S*sqrt(T)*norm.pdf(d1)*exp(-rf*T)
    
    def theta(self, Option_type,S,K,T,r,rf,sigma):
        a = (-S *norm.pdf(d1) *sigma* exp(-rf*T)) /(2*sqrt(T))
        b = rf * S * norm.cdf(d1) * exp(-rf*T) 
        c = r * K * exp(-r*T)* norm.cdf(d2)
        b2 = rf * S * norm.cdf(-d1) * exp(-rf*T) 
        c2 = r * K * exp(-r*T)* norm.cdf(-d2)
        if Option_type == "Call":
             theta = a+b-c   
        elif Option_type == "Put":
            theta = a-b2+c2 
        return theta
    
    """
    
    def greeks(self, Option_type,S,K,T,r,rf,sigma):
        print("delta = ", self.delta(Option_type,S,K,T,r,rf,sigma) ,'\n')
        print("gamma = ", self.gamma(Option_type,S,K,T,r,rf,sigma) ,'\n')
        print("theta = ", self.theta(Option_type,S,K,T,r,rf,sigma) ,'\n')
        print("vega = ", self.vega(Option_type,S,K,T,r,rf,sigma) ,'\n')
    
    """
    
    def __init__(self,Option_type,S,K,T,r,rf,sigma):
    
        self.asset_price = S
        self.strike = K
        self.time_to_exp = T
        self.d_rate = r
        self.f_rate = rf
        self.asset_volatility = sigma
        
        self.price = self.bs_pricing(Option_type,S,K,T,r,rf,sigma)
        
        
        self.delta = self.delta(Option_type,S,K,T,r,rf,sigma)
        self.gamma = self.gamma(Option_type,S,K,T,r,rf,sigma)
        self.theta = self.theta(Option_type,S,K,T,r,rf,sigma)
        self.vega = self.vega(Option_type,S,K,T,r,rf,sigma)
        #self.greek = self.greeks(Option_type,S,K,T,r,rf,sigma)

""" 
# test
Option_type = "Call"
S = 300 #stock price S_{0}
K = 250 # strike
T = 1 # time to maturity
r = 0.03 # domestic interest rate in annual %
rf = 0.00 # foreign interest rate in annual %
sigma = 0.15 # annual volatility in %
    
bs = BlackScholes("Call",S,K,T,r,rf,sigma)
print( bs.price)
"""


# Input 
Option_type = option["C/P"]
S = option['Spot']
K = option['Strike']
T1 = option['Maturity']
T2 = option['Value Date']
r = option['Rate Cur1']
rf = option['Rate Cur2']
sigma = option['Volat.']
Notional = option['Notional']
    
# Convert datetime to integer
import time
from datetime import datetime
    
option['Date'] = 0 # new column 
date = pd.to_datetime(T1) - pd.to_datetime(T2)

for i in range(0,len(date)):
    option['Date'][i] = date[i].days
T = option['Date']
T = T/360    
    

## BS price and greeks
d = []
    
for i in range(0,len(S)):
    d.append(BlackScholes(Option_type[i],S[i],K[i],T[i],r[i],rf[i],sigma[i]).price)
    #option['BS_price'] = d
    
        
## Monte carlo pricing

def geo_paths(S, T, r, rf, sigma, steps, N):
    """
    Inputs
    #S = Current currency Price
    #K = Strike Price
    #T = Time to maturity 1 year = 1, 1 months = 1/12
    #r = domestic interest rate in annual %
    #rf = foreign interest rate in annual %
    # sigma = volatility 
    
    Output
    # [steps,N] Matrix of asset paths 
    """
    dt = T/steps
    ST = np.log(S) +  np.cumsum(((r - rf - sigma**2/2)*dt + sigma*np.sqrt(dt) * np.random.normal(size=(steps,N))),axis=0)
    return np.exp(ST)

"""
# EXAMPLE
S = 0.12 #currency price S_{0}
K = 0.10 # strike
T = 1/2 # time to maturity
r = 0.05 # domestic interest rate in annual %
rf = 0.02 # foreign interest rate in annual %
sigma = 0.25 # annual volatility in %
steps = 252 # time steps
N = 1000 # number of trials

paths= geo_paths(S,T,r,rf,sigma,steps,N)

plt.plot(paths);
plt.xlabel("Time Increments")
plt.ylabel("Currency Price")
plt.title("Geometric Brownian Motion")
"""

    
def monte_carlo_pricing(Option_type, S,K, T, r, rf, sigma, steps, N):
    
    option_price = 0
    paths= geo_paths(S, T, r, rf,sigma, steps, N)
    if Option_type == "Call":
        payoffs = np.maximum(paths[-1]-K, 0)
        option_price = np.exp(-r*T)*np.mean(payoffs) #discounting back to present value
        
    elif Option_type == "Put":
        payoffs = np.maximum(K-paths[-1], 0)
        option_price = np.exp(-r*T)*np.mean(payoffs) #discounting back to present value

    else:
        print("incorrect parameters")
    
    return option_price

# Monte carlo pricing
m =[]
for i in range(0,len(S)):
    m.append( monte_carlo_pricing(Option_type[i],S[i],K[i],T[i],r[i],rf[i],sigma[i], steps=100, N=1000) )
    # option['Monte_carlo_price'] = m       
    

## Implied volatility 
def implied_volatility(price , Option_type, S, K, T, r, rf):
    Objective = lambda x: abs( price - BlackScholes(Option_type,S,K,T,r,rf,x).price )**2    
    return fmin_bfgs(Objective, 0.5, disp = False)[0]

k =[]
for i in range(0,len(S)):
    k.append( implied_volatility(price[i] , Option_type[i], S[i], K[i], T[i], r[i], rf[i]) )
    # option['Implied volatility'] = k      
    
##
    




