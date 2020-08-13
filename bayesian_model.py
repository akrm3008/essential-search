import pymc3 as pm
import math
import datetime
import pandas as pd
import numpy as np
from geo import getDistanceDurationBayes
import datetime

#import theano
#import theano.tensor as T
#from theano.compile.ops import as_op
#class GasStation():    
#def __init__(self,)



#V = [0,1]
#P = [0]
#T = ['09-16-2020 9:00:00', '09-16-2020 12:00:00', 
#                  '09-16-2020 15:00:00', '09-16-2020 18:00:00', '09-16-2020 21:00:00',
#                  '109-16-2020 22:00:00', '09-16-2020 23:00:00']


#with pm.Model() as model:
#    
#    Obs = {(i,t) : pm.Bernoulli(name='obs'+ str(i) + str(t), p = 0.9) for i in V for t in range(len(T))}
#    Lam = pm.Deterministic('lam', pm.math.switch(Obs[(0,0)], 1, 0))
#    Mu = pm.Deterministic('mu', pm.math.switch(Obs[(0,0)], 6, 0))
#    Distance_lag = {(p,i) : pm.Exponential(name ='dist' + str(p) + str(i), lam = Lam) for p in P for i in V}
#    Time_lag = {(p,i,t) : pm.Exponential(name ='time' + str(p) + str(i) + str(t), lam = Mu)  
#                       for p in P for i in V for t in range(len(T))}
#    
#
#
#x1 = np.random.exponential(scale = 1, size = 1)[0]
#x2 = np.random.exponential(scale = 4, size = 4)[0]
#x3 =  np.random.exponential(scale = 1, size = 1)[0]
#x4 = np.random.exponential(scale = 4, size = 4)[0]
#Distance_data = np.array([x1, x2, x3, x4])
#Ditance_data = observed_data.reshape(2,2)
#Time_lag_data = Distance_data



def RetreiveData(Post_data, Gas_stations, Date = None, Obs_times = None):
    
    nV = Gas_stations.shape[0]
    
    if pd.isnull(Date):
        Post_data_rel = Post_data
              
    else:
        Post_data_rel = Post_data[Post_data['DATE'] == Date]
        
    if pd.isnull(Obs_times):
        nT = Post_data_rel.shape[0]
        Obs_times = Post_data_rel['DATE_TIME'].to_numpy()
    else:
        nT = len(Obs_times)
        
    Post_time = Post_data_rel['DATE_TIME'].to_numpy()
    
    nP = Post_data_rel.shape[0]
    
    Gas_stations_locations = Gas_stations['Location'].to_numpy()
    Post_data_SNO = Post_data_rel['SNO'].to_numpy()

    Gas_stations_dict = {i : Gas_stations_locations[i] for i in range(len( Gas_stations_locations))}
    Post_dict = {p : Post_data_SNO[p] for p in range(len(Post_data_SNO))}
    Obs_times_dict = {t : Obs_times[t] for t in range(len(Obs_times))}
    
 
    TravelDuration_data, Distance_data = getDistanceDurationBayes(Gas_stations, Post_data_rel)
        
        
    Time_lag = {(i,t,p) : 0 for i in range(Gas_stations.shape[0]) for t in range(len(Obs_times)) for p in range(len(Post_data_rel))}
    
    
    for i in range(Gas_stations.shape[0]) :
        for t in range(len(Obs_times)) :
            for p in range(len(Post_data_rel)):
                if i!= p:
                    if (Post_time[p] - Obs_times[t])/np.timedelta64(1, 's')/60 < int(TravelDuration_data[(i,p)]) :
                        Time_lag[(i,t,p)] = math.inf
                    else: 
                        Time_lag[(i,t,p)] = (Post_time[p] - Obs_times[t])/np.timedelta64(1, 's')/60 
       
    return  nV, nT, nP, Gas_stations_dict, Post_dict, Obs_times_dict, TravelDuration_data, Distance_data, Time_lag

    



def CalcProsteriorProbs(nV, nT, nP, Time_lag_data, Distance_data): 

    with pm.Model() as model: 
    
        Obs = pm.Bernoulli('Obs', p = 0.5, shape = (nV, nT))
    
        x= (1-Obs).prod()
        
        Posts_p = pm.Deterministic('Posts_p', 0.5 - 0.5*x)
        Posts = pm.Bernoulli(name = 'Posts', p = Posts_p, shape = 1)
        
        Lams = pm.Deterministic('Lams', 0.05 + Obs[...,None]*Posts*0.95)
        Mus = pm.Deterministic('Mus', 0.05 + Obs[...,None]*Posts*0.95)
         
        Time_lag = {(i,t,p) : pm.Exponential(name ='Time_lag' + str(i) + str(t) + str(p), lam = Lams[i][t][p],
                              observed = Time_lag_data[i][t][p]) for i in range(nV) for t in range(nT) for p in range(nP)}
        Distance = {(i,t,p) : pm.Exponential(name ='Distance_lag' + str(i) + str(t) + str(p), lam = Mus[i][t][p],
                              observed = Distance_data[(i,p)]) for i in range(nV) for t in range(nT) for p in range(nP)}
        
        trace = pm.sample(2000)
        pm.traceplot(trace)
        pm.summary(trace, varnames=['Obs', 'Posts', 'Post_p', 'Lams', 'Mus'])
        
    return pm
            
                    

                

