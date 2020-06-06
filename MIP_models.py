# Maximise the probability of getting gas
from gurobipy import Model, GRB, quicksum
import copy


## Depot is needed as dummy node in our formulation as we have the  constraint 4 and constraint 1
## Big M required in objective as all nodes or arcs dont need to be visited. In that case 


def minimise_expected_time(data): 
        
    # Getting all the parameters
    data1 = copy.deepcopy(data)
    
    n = data1.n
    V= data1.V
    N= data1.N
    AO= data1.AO
    p_log = data1.p_log
    p_log2 = data1.p_log2
    d = data1.d
    f = data1.f
    mi = data1.mi
    t = data1.t
    
    # Initiating Model
    m = Model('Gasoline Search')
    
    # Defining variable
    x = m.addVars(AO, vtype = GRB.BINARY)
    #y = m.addVars(VO, vtype = GRB.BINARY)
    
    
    # Defining the objective function                
    #m.setObjective(quicksum(t[i,j]*x[i,j,k] for i in V for j in V if j != i for k in A), "minimize")
    m.setObjective(quicksum((p_log[j])*x[i,j,k] + (p_log2[j])*x[i,j,m] 
                            for k in range(1,n+1) for m in range(1,k) for i in V for j in N if j != i )
                             + quicksum(t[i,j]*x[i,j,l] for k in range(1,n+1) for l in range(1,k+1) for i in V for j in N if j != i)
                             - quicksum(1000*x[i,j,k] for k in range(1,n+1) for i in V for j in N if j != i), GRB.MINIMIZE)
        
    # Defining constraints
    
    # Constraint 1 : Length of the path constrained by the fuel left
    m.addConstr(quicksum(d[i,j]*x[i,j,k] for k in range(1,n+1)for i in V for j in V if j != i ) <= f*mi)
    
    
    
    # Constraint 2 : Only one arc can be the kth arc in the path  
    m.addConstrs(quicksum(x[i,j,k] for i in V for j in V if j != i ) <= 1 for k in range(1,n+2))
    
    # Constraint 3: Each node should be visited only once 
    m.addConstrs(quicksum(x[i,j,k] for k in range(1,n+2) for i in V if i!= j) <= 1 for j in V )
    
     # Constraint 4 : Destination of the kth arch is the origin of the k+1 the arc
    m.addConstrs(quicksum(x[i,j,k] - x[j,i,k+1] for i in V if i!=j) == 0 for k in range(1,n+1)
                                                                for j in N)
        
    # Constraint 5: 1st arc starts at the origin
    m.addConstr(quicksum(x[0,j,1] for j in V if j != 0) == 1)
    
  
    # Constraint 5 : Constraint for vertex order variable
    #m.addConstrs(y[j,k] - quicksum(x[i,j,k] for i in V if j != i ) == 0 for j in N for k in range(1,n+1))
    
    # Constraint 6: Depot/Dummy is the last node to be visited 
    m.addConstr(quicksum(x[i,0,n+1] for i in N) == 1)
    
    # optimizing the model
    m.optimize()
    
    # Active Arc-orders
 
    arc_orders = [a for a in AO if x[a].x >0.99]
    arc_orders = [a  for j in N for a in arc_orders if a[2]== j]
    

    return arc_orders


# minimise expected_time without distance constraint

def minimise_expected_time2(data): 
    
    data1 = copy.deepcopy(data)
        
    # Getting all the parameters
    n = data1.n
    V= data1.V
    N= data1.N
    AO= data1.AO
    p_log = data1.p_log
    p_log2 = data1.p_log2
    #d = data1.d
    #f = data1.f
    #mi = data1.mi
    t = data1.t
    
    # Initiating Model
    m = Model('Gasoline Search')
    
    # Defining variable
    x = m.addVars(VO, vtype = GRB.BINARY)
    #y = m.addVars(VO, vtype = GRB.BINARY)
    
    
    # Defining the objective function                
    #m.setObjective(quicksum(t[i,j]*x[i,j,k] for i in V for j in V if j != i for k in A), "minimize")
    m.setObjective(quicksum((p_log[j])*x[i,j,k] + (p_log2[j])*x[i,j,m] 
                            for k in range(1,n+1) for m in range(1,k) for i in V for j in N if j != i )
                             + quicksum(t[i,j]*x[i,j,l] for k in range(1,n+1) for l in range(1,k+1) for i in V for j in N if j != i)
                             - quicksum(1000*x[i,j,k] for k in range(1,n+1) for i in V for j in N if j != i), GRB.MINIMIZE)
        
    # Defining constraints
    
     #Constraint 1 : Lenght of the path constrained by the fuel left
    #m.addConstr(quicksum(d[i,j]*x[i,j,k] for k in range(1,n+1)for i in V for j in V if j != i ) <= f*mi)
    
    # Constraint 2 : Only one arc can be the kth arc in the path  
    m.addConstrs(quicksum(x[i,j,k] for i in V for j in V if j != i ) <= 1 for k in range(1,n+1))
        
    # Constraint 3 : 1st arc starts at the origin
    m.addConstr(quicksum(x[0,j,1] for j in V if j != 0) == 1)
    
    # Constraint 4 : Destination of the kth arch is the origin of the k+1 the arc
    m.addConstrs(quicksum(x[i,j,k] - x[j,i,k+1] for i in V if i!=j) == 0 for k in range(1,n+1)
                                                                for j in N)
    # Constraint 5 : Constraint for vertex order variable
    #m.addConstrs(y[j,k] - quicksum(x[i,j,k] for i in V if j != i ) == 0 for j in N for k in range(1,n+1))
    
    # Constraint 5: Each node should be visited only once 
    m.addConstrs(quicksum(x[i,j,k] for k in range(1,n+2) for i in V if i!= j) <= 1 for j in V )
    
    # Constraint 6: Depot/Dummy is the last node to be visited 
    m.addConstr(quicksum(x[i,0,n+1] for i in N) == 1)
    
    # optimizing the model
    m.optimize()
    
    # Active Arc-orders
 
    arc_orders = [a for a in AO if x[a].x >0.99]
    arc_orders = [a  for j in N for a in arc_orders if a[2]== j]
    

    return arc_orders


def maximise_probability(data): 
    
    data1 = copy.deepcopy(data)
    
    # Getting all the parameters
    n = data1.n
    V= data1.V
    N= data1.N
    AO= data1.AO
    p_log = data1.p_log
    p_log2 = data1.p_log2
    d = data1.d
    f = data1.f
    mi = data1.mi
    t = data1.t
    
    # Initiating Model
    m = Model('Gasoline Search')
    
    # Defining variable
    x = m.addVars(AO, vtype = GRB.BINARY)
    #y = m.addVars(VO, vtype = GRB.BINARY)
    
    
    # Defining the objective function                
    #m.setObjective(quicksum(t[i,j]*x[i,j,k] for i in V for j in V if j != i for k in A), "minimize")
    m.setObjective(quicksum((p_log[j])*x[i,j,k] + (p_log2[j])*x[i,j,m] 
                            for k in range(1,n+1) for m in range(1,k) for i in V for j in N if j != i )
                             + quicksum(t[i,j]*x[i,j,l] for k in range(1,n+1) for l in range(1,k+1) for i in V for j in N if j != i)
                             - quicksum(1000*x[i,j,k] for k in range(1,n+1) for i in V for j in N if j != i), GRB.MINIMIZE)
        
    # Defining constraints
    
    # Constraint 1 : Lenght of the path constrained by the fuel left
    m.addConstr(quicksum(d[i,j]*x[i,j,k] for k in range(1,n+1)for i in V for j in V if j != i ) <= f*mi)
    
    # Constraint 2 : Only one arc can be the kth arc in the path  
    m.addConstrs(quicksum(x[i,j,k] for i in V for j in V if j != i ) <= 1 for k in range(1,n+2))
        
    # Constraint 3 : 1st arc starts at the origin
    m.addConstr(quicksum(x[0,j,1] for j in V if j != 0) == 1)
    
    # Constraint 4 : Destination of the kth arch is the origin of the k+1 the arc
    m.addConstrs(quicksum(x[i,j,k] - x[j,i,k+1] for i in V if i!=j) == 0 for k in range(1,n+1)
                                                                for j in N)
    # Constraint 5 : Constraint for vertex order variable
    #m.addConstrs(y[j,k] - quicksum(x[i,j,k] for i in V if j != i ) == 0 for j in N for k in range(1,n+1))
    
    # Constraint 5: Each node should be visited only once 
    m.addConstrs(quicksum(x[i,j,k] for k in range(1,n+2) for i in V if i!= j) <= 1 for j in V )
    
    # Constraint 6: Depot/Dummy is the last node to be visited 
    m.addConstr(quicksum(x[i,0,n+1] for i in N) == 1)
    
    # optimizing the model
    m.optimize()
    
    # Active Arc-orders
 
    arc_orders = [a for a in AO if x[a].x >0.99]
    arc_orders = [a  for j in N for a in arc_orders if a[2]== j]
    
    
    return arc_orders





