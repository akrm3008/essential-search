import pymc3 as pm
import math
import datetime
import pandas as pd
import numpy as np
from locations_and_distance import getDistanceDuration
import datetime

#import theano
#import theano.tensor as T
#from theano.compile.ops import as_op
#class GasStation():    
#def __init__(self,)



V = [0,1]
P = [0]
T = ['09-16-2020 9:00:00', '09-16-2020 12:00:00', 
                  '09-16-2020 15:00:00', '09-16-2020 18:00:00', '09-16-2020 21:00:00',
                  '109-16-2020 22:00:00', '09-16-2020 23:00:00']


with pm.Model() as model:
    
    Obs = {(i,t) : pm.Bernoulli(name='obs'+ str(i) + str(t), p = 0.9) for i in V for t in range(len(T))}
    Lam = pm.Deterministic('lam', pm.math.switch(Obs[(0,0)], 1, 0))
    Mu = pm.Deterministic('mu', pm.math.switch(Obs[(0,0)], 6, 0))
    Distance_lag = {(p,i) : pm.Exponential(name ='dist' + str(p) + str(i), lam = Lam) for p in P for i in V}
    Time_lag = {(p,i,t) : pm.Exponential(name ='time' + str(p) + str(i) + str(t), lam = Mu)  
                       for p in P for i in V for t in range(len(T))}
    


x1 = np.random.exponential(scale = 1, size = 1)[0]
x2 = np.random.exponential(scale = 4, size = 4)[0]
x3 =  np.random.exponential(scale = 1, size = 1)[0]
x4 = np.random.exponential(scale = 4, size = 4)[0]
Distance_data = np.array([x1, x2, x3, x4])
Ditance_data = observed_data.reshape(2,2)
Time_lag_data = Distance_data


#def calcShortageProb()



def RetreiveData(Obs_times = False , Post_data, Gas_stations):
    
    nV = Gas_staions.shape[0]
    
    if not Obs_times:
        nT = Post_data.shape[0]
        Obs_times = Post_data['DATE_TIME'].to_numpy().tolist()
    else:
        nT = len(Obs_times)
    nP = Post_data.shape[0]
    
    
    
    Gas_stations_locations = Gas_stations['Locations'].to_numpy().tolist()
    Post_data_SNO = Post_data['SNO'].to_numpy().tolist()

    
    Gas_stations_dict = {i : Gas_stations[i] for i in range(len( Gas_stations_locations))}
    Post_dict = {p : Post_data_SNO[p] for p in range(len(Gas_stations_locations))}
    Obs_times_dict = {t : Obs_times[t] for t in range(len(Obs_times))}
    
    TravelDuration_data, Distance_data = getDistanceDurationBayes(Gas_stations, Post_data)
    Time_lag = {(i,t,p) : 0 for i in range(Gas_stations) for t in range(Obs_times.shape[0]) for p in range(Post_data.shape[0])}
    
    
    for i in range(Gas_stations) :
        for t in range(Obs_times.shape[0]) :
            for p in range(Post_data.shape[0]):
                if ((Post_time[p] - Obs_times[t]).seconds)/60 < int(TravelDuration_Data[(i,p)]) :
                    Time_lag[(i,t,p)] = math.inf
                else: 
                    Time_lag[(i,t,p)] = ((Post_time[p] - Obs_times[t]).seconds)/60 
   
    return  nV, nT, nP, Gas_stations_dict, Post_dict, Obs_times_dict, TravelDuration_data, Distance_data, Time_lag

    



def CalcProsteriorProbs(nV, nT, nP, Time_lag_data, Distance_data): 

    with pm.Model() as model: 
    
        Obs = pm.Bernoulli('Obs', p = 0.5, shape = (nV, nT))
    
        x= (1-Obs).prod()
        
        Posts_p = pm.Deterministic('Tweets_p', 0.5 - 0.5*x)
        Posts = pm.Bernoulli(name = 'Tweets', p = Tweets_p, shape = 1)
        
        Lams = pm.Deterministic('Lams', 0.05 + Obs[...,None]*Posts*0.95)
        Mus = pm.Deterministic('Mus', 0.05 + Obs[...,None]*Posts*0.95)
         
        Time_lag = {(i,t,p) : pm.Exponential(name ='Time_lag' + str(i) + str(t) + str(p), lam = Lams[i][t][p],
                              observed = Time_lag_data[i][t][p]) for i in range(nV) for t in range(nT) for p in range(nP)}
        Distance = {(i,t,p) : pm.Exponential(name ='Distance_lag' + str(i) + str(t) + str(p), lam = Mus[i][t][p],
                              observed = Distance_data[(i,p)]) for i in range(nV) for t in range(nT) for p in range(nP)}
        
        trace = pm.sample(2000)
        pm.traceplot(trace)
        pm.summary(trace, varnames=['Obs', 'Posts', 'Post_p', 'Lams', 'Mus'])
            
                    

## Following code tests the functions we deifined in this script
        
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)    
    
Path1 = "/Users/abhinavkhare/Documents/Phd project/Results/results_with_final_data/gas_tweets_till15th_v1.csv"
Path2 = "/Users/abhinavkhare/Documents/Phd Project/Data/Data/tweets.csv"
df_label = pd.read_csv(Path1, encoding = "ISO-8859-1")
df  = pd.read_csv(Path2, encoding = "ISO-8859-1")

df = df.rename(columns = {'Unnamed: 0': 'SNO', 'TWEET_TEXT' : 'TWEET_TEXT_MAIN' })
df_merged  = df.merge(df_label, on = 'SNO')[['SNO', 'TWEET_TEXT', 'DATE', 'TIME', 'TIMEZONE', 'LATITUDE', 'LONGITUDE', 'label_s']]
post_data = df_merged[df_merged['LATITUDE'].notnull() & df_merged['label_s'] == 1]
post_data['DATE_TIME'] = post_data['DATE'] + " " + post_data['TIME']
post_data['DATE_TIME'] = post_data['DATE_TIME'].apply(lambda x: x.replace("2017", "17"))
post_data['DATE_TIME'] =  post_data['DATE_TIME'].apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%y %H:%M:%S'))




