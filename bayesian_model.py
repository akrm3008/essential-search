import pymc3 as pm
import math
import datetime
import pandas as pd
import numpy as np
from geo import getDistanceDuration

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
        OBs_times = Post_data['Time']
    else:
        nT = len(Obs_times)
    nP = Post_data.shape[0]
    
   TravelDuration_data, Distance_data = getDistanceDurationBayes(Gas_stations, Post_data)

    



pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)    
    
Path1 = "/Users/abhinavkhare/Documents/Phd project/Results/results_with_final_data/gas_tweets_till15th_v1.csv"
Path2 = "/Users/abhinavkhare/Documents/Phd Project/Data/Data/tweets.csv"
df_label = pd.read_csv(Path1, encoding = "ISO-8859-1")
df  = pd.read_csv(Path2, encoding = "ISO-8859-1")

df = df.rename(columns = {'Unnamed: 0': 'SNO', 'TWEET_TEXT' : 'TWEET_TEXT_MAIN' })
df_merged  = df.merge(df_label, on = 'SNO')[['SNO', 'TWEET_TEXT', 'DATE', 'TIME', 'TIMEZONE', 'LATITUDE', 'LONGITUDE', 'label_s']]
post_data = df_merged[df_merged['LATITUDE'].notnull() & df_merged['label_s'] == 1]


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
                          observed = Distance_data[i][p]) for i in range(nV) for t in range(nT) for p in range(nP)}
    
    trace = pm.sample(2000)
    pm.traceplot(trace)
    pm.summary(trace, varnames=['Obs', 'Posts', 'Post_p', 'Lams', 'Mus'])
        
                
    
Obs.random(size = 100)      
Time.random(point = {Obs, Tweets, Tweets_p, Lams}, size = 100)



with pm.Model() as model:
    
    Obs1 = pm.Bernoulli(name='Obs1', p = 0.50)
    Obs2 = pm.Bernoulli(name='Obs2', p = 0.50)
    
        
    Obs3 = pm.Bernoulli(name='Obs3', p = 0.50)
    Obs4 = pm.Bernoulli(name='Obs4', p = 0.50)
#    
    
   
    #Tweet_p = pm.Deterministic('Tweet_p', pm.math.switch(Obs1,pm.math.switch(Obs2, 0.50, 0.50), pm.math.switch(Obs2, 0.50, 0)))
#    Tweet_p = pm.Deterministic('Tweet_p', 0.5 - 0.5*(1-Obs1)*(1-Obs2))
    Tweet_p = pm.Deterministic('Tweet_p', 0.5 - 0.5*(1-Obs1)*(1-Obs2)*(1-Obs3)*(1-Obs4))
    
    
    Tweet = pm.Bernoulli(name = 'Tweet', p = Tweet_p)
    
    Lam1 = pm.Deterministic('Lam1', 0.05 + Obs1*Tweet*0.95)
    Lam2 = pm.Deterministic('Lam2', 0.05 + Obs2*Tweet*0.95)
    Lam3 = pm.Deterministic('Lam3', 0.05 + Obs3*Tweet*0.95)
    Lam4 = pm.Deterministic('Lam4', 0.05 + Obs4*Tweet*0.95)
#   Lam1 = pm.Deterministic('Lam1', pm.math.switch(Obs1,pm.math.switch(Tweet, 1, 0.05), 
#                                                pm.math.switch(Tweet, 0.05, 0.05)))
#   Lam2 = pm.Deterministic('Lam2', pm.math.switch(Obs2,pm.math.switch(Tweet, 1, 0.05), 
#                                                pm.math.switch(Tweet, 0.05, 0.05)))
    
#   Mu1 = pm.Deterministic('Mu1', pm.math.switch(Obs1,pm.math.switch(Tweet, 1, 0.05), 
#                                                pm.math.switch(Tweet, 0.05, 0.05)))
#   Mu2 = pm.Deterministic('Mu2', pm.math.switch(Obs1,pm.math.switch(Tweet, 1, 0.05), 
#                                                pm.math.switch(Tweet, 0.05, 0.05)))
    
    Mu1 = pm.Deterministic('Mu1', 0.05 + Obs1*Tweet*0.95)
    Mu2 = pm.Deterministic('Mu2', 0.05+ Obs2*Tweet*0.95)
    Mu3 = pm.Deterministic('Mu3', 0.05 + Obs3*Tweet*0.95)
    Mu4 = pm.Deterministic('Mu4', 0.05+ Obs4*Tweet*0.95)
    
    
   
    Time1 = pm.Exponential(name ='Time1', lam = Lam1, observed = x1)
    Time2 = pm.Exponential(name ='Time2', lam = Lam2, observed = x2)
    Time3 = pm.Exponential(name ='Time3', lam = Lam3, observed = x3)
    Time4 = pm.Exponential(name ='Time4', lam = Lam4, observed = x4)
    
       
#    Time1 = pm.Exponential(name ='Time1', lam = Lam1)
#    Time2 = pm.Exponential(name ='Time2', lam = Lam2)
#    Time3 = pm.Exponential(name ='Time3', lam = Lam3)
#    Time4 = pm.Exponential(name ='Time4', lam = Lam4)
    
    #Dist1 = pm.Exponential(name ='Dist1', lam = Mu1, observed=np.random.exponential(scale = 1, size=3))
    #Dist2 = pm.Exponential(name ='Dist2', lam = Mu2, observed=np.random.exponential(scale = 1, size=3))
    #Dist3 = pm.Exponential(name ='Dist3', lam = Mu3, observed=np.random.exponential(scale = 4, size=3))
    #DIst4 = pm.Exponential(name ='Dist4', lam = Mu4, observed=np.random.exponential(scale = 4, size=3))
    
    
    
#    Dist1 = pm.Exponential(name ='Dist1', lam = Lam1, observed=np.random.exponential(scale = 1, size=3))
#    Dist2 = pm.Exponential(name ='Dist2', lam = Lam1, observed=np.random.exponential(scale = 1, size=3))
    #Distance = pm.Exponential(name ='Distance', lam = 0)
    #Distance =pm.Normal(name ='Distance', mu = Lam , sigma = 2, observed=np.random.randn(100))
    #Time = pm.Exponential(name ='Time', lam = 6*Lam)
    #deltaT = 6
    #alpha = 0.2
    #Fut_short = pm.Bernoulli(name = 'Pred', p = Obs + alpha*deltaT)
    trace = pm.sample(2000)
    pm.traceplot(trace)
    pm.summary(trace, varnames=['Obs1', 'Obs2','Obs3', 'Obs4', 'Tweet', 'Tweet_p','Lam1', 'Lam2',
                                'Lam3', 'Lam4'])
#with pm.Model() as model:
#    x =pm.Bernoulli(name='x', p = 0.9, shape = (len(V),len(T)))
    pm.summary(trace, varnames=['Obs1', 'Obs2', 'Tweet_p', 'Lam1','Lam2' ])
    #pm.summary(trace,varnames=['Distance'])
    pm.summary(trace, varnames=['Obs1', 'Obs2', 'Tweet_p'])
#  pm.summary(trace, varnames=['Obs', 'Distance','Time'])



#
#def logp(obs1, lam, value):
#    if obs1 == 0:
#        return math.inf
#    if obs1 == 1:
#        return math.log(lam) - lam*value
#
#
#with pm.Model() as model:
#    
#    Obs1 = pm.Bernoulli(name='Obs1', p = 0.51)
##    Lam1 = pm.Deterministic('Lam1', pm.math.switch(Obs1,1,0))
##    Time1 = pm.Exponential(name ='Time1', lam = Lam1, observed=np.random.exponential(scale = 1, size=10))
#    lam = 1 
#    Time1 = pm.DensityDist('Time1', logp, observed=dict(obs1 = 1 , lam = lam, value = 0.5))
#    
#    
#    
#    trace = pm.sample(200)
#    pm.traceplot(trace)
#    pm.summary(trace, varnames=['Obs1'])
#
#
#









Obs.random(size = 100)      
Distance.random(point = {Obs,Lam}, size = 100)
Time1.random(point = {Obs, Mu}, size = 100)





tweet_datetimes = ['09-16-2020 21:00:00']




Obs.random(size = 100)      
Distance.random(point = {Obs,Lam}, size = 100)
Time.random(point = {Obs, Mu}, size = 100)








