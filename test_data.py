import math 
import numpy as np
import itertools




class created_data: 
    
    def __init__(self, n, p, xc=[], yc=[],  f=[], mi=[], t={}, d={}, s={}):
        self.n = n # number of gas stations
        # Sets 
        self.N = [x for x in range(1,self.n+1)] # Set of gas stations
        self.V =  self.N + [0]  # Set of all nodes 
        self.VO = [(i,k)  for i in self.V for k in range(1,len(self.N)+2)] # the vertex-order ser
        self.AO = [(i,j,k)  for i in self.V for j in self.V if i !=j for k in range(1,len(self.N)+2)]  # the arc-order set 
        self.AO2 = [(i,j)  for i in self.V for j in self.V if i !=j ]
        self.xc = xc
        self.yc = yc
        self.p = p 
        self.f = f # fuel 
        self.mi = mi # milege 
        self.p_log = {j: math.log(self.p[j]) for j in self.V}
        self.p_log2 = {j: math.log(1 - self.p[j]) for j in self.V}
        self.paths = []
        self.t = t
        self.d = d 
        self.s = s
        
        # Parameters
    def generate_parmaters(self):
        self.d = {(i,j): round(np.hypot(self.xc[i]-self.xc[j], self.yc[i]- self.yc[j]),2) for i in 
                  self.V for j in self.V} # distance between nodes (of arcs)
        for i in self.V:
            for j in self.V:
                if j==0:
                    self.d[i,j]=0
        self.s = {(i,j): 1 for i in self.V for j in self.V} #  average speeds at each arcs 
        self.t = {(i,j): self.d[i,j]/self.s[i,j] for i in self.V for j in self.V}  #  average travel time between nodes
         # probability of finding gas at each gas station
     

        
    def create_paths(self) :
        arc_orders = list(itertools.permutations(self.N))
        arc_sets = []
        for i in range(2,len(self.N)):
            arc_sets = list(itertools.combinations(self.N, i))
            for x in arc_sets:
                arc_orders = arc_orders + list(itertools.permutations(x))
        arc_orders = [[0]+ list(x) for x in arc_orders]
        self.paths = [[(x[j],x[j+1]) for j in range(0,len(x)) if j!=len(x)-1] for x in arc_orders]
        
        
class random_data:
    
    def __init__(self,n, seed):
        self.n = n
        self.seed = seed 
        self.rnd = np.random
        self.rnd.seed(self.seed)
       
        # Sets 
        self.N = [x for x in range(1,n+1)] # Set of gas stations
        self.V =  self.N + [0]  # Set of all nodes 
        #A = [(i,j) for i in V for j in V if i!=j]  # Set of arcs 
        self.VO = [(i,k)  for i in self.V for k in range(1,len(self.N)+2)]
        self.AO = [(i,j,k)  for i in self.V for j in self.V if i !=j for k in range(1,len(self.N)+2)]  # the arc-order set 
        self.AO2 = [(i,j)  for i in self.V for j in self.V if i !=j ]
        #VO = [(j,k)  for j in V for k in range(1,len(N)+1)] 
        
        self.xc = self.rnd.rand(self.n+1)*200
        self.yc = self.rnd.rand(self.n+1)*100
        
        # Parameters
        self.d = {(i,j): round(np.hypot(self.xc[i]-self.xc[j], self.yc[i]- self.yc[j]),2) for i in self.V for j in self.V} # distance between nodes (of arcs)
        for i in self.V:
            for j in self.V:
                if j==0:
                    self.d[i,j]=0
        self.s = {(i,j): round(self.rnd.rand(1)[0]*10,2) for i in self.V for j in self.V} #  average speeds at each arcs 
        self.t = {(i,j): self.d[i,j]/self.s[i,j] for i in self.V for j in self.V}  #  average travel time between nodes
        self.p = {j: round(self.rnd.rand(1)[0],2) for j in self.N}  # probability of finding gas at each gas station
        self.p.update({0 : 0.0000001}) # Probability of finding gas at the depot is zero (~0 cause of log in objective )
        # milege 
        
        self.f = 10  # fuel 
        self.mi = 2 # milege
        
        self.p_log = {j: math.log(self.p[j]) for j in self.V}
        self.p_log2 = {j: math.log(1 - self.p[j]) for j in self.V}
        
        self.paths = []
        
    def create_paths(self) :
        arc_orders = list(itertools.permutations(self.N))
        arc_sets = []
        for i in range(2,len(self.N)):
            arc_sets = list(itertools.combinations(self.N, i))
            for x in arc_sets:
                arc_orders = arc_orders + list(itertools.permutations(x))
        arc_orders = [[0]+ list(x) for x in arc_orders]
        self.paths = [[(x[j],x[j+1]) for j in range(0,len(x)) if j!=len(x)-1] for x in arc_orders]
    
    

n = 3   
xc = np.array([0, 0, 1, 1])  
yc = np.array([0, 1, 0, 1])  
p = {0: 0.0000001,
     1: 0.3,    
     2: 0.2,
     3: 0.7}
f = 2
mi =  10 

N = [x for x in range(1,n+1)] # Set of gas stations
V =  N + [0]  # Set of all nodes 
#A = [(i,j) for i in V for j in V if i!=j]  # Set of arcs 
AO = [(i,j,k)  for i in V for j in V if i !=j for k in range(1,len(N)+2)]  # the arc-order set 

d = {(i,j): round(np.hypot(xc[i]- xc[j], yc[i]- yc[j]),2) for i in 
          V for j in V} # distance between nodes (of arcs)
for i in V:
    for j in V:
        if j==0:
            d[i,j]=0
s = {(i,j): 1 for i in V for j in V} #  average speeds at each arcs 
t = {(i,j): d[i,j]/s[i,j] for i in V for j in V}




