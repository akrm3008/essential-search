
# This function returns expected time time to find gas given we find gas on a decided path 
def expected_time_find_gas(arc_orders, data): 
    
    expected_time = 0
    total_prob_find = 0
    
    for k in range(1,len(arc_orders)+1):
        prob_term = 1 
        time_term = 0 
        for i in range(0,k): 
            if i== k-1:
                prob_term = prob_term*data.p[arc_orders[i][1]]
                time_term = time_term + data.t[arc_orders[i][0],arc_orders[i][1]]

    
            else: 
                prob_term = prob_term*(1 - data.p[arc_orders[i][1]])
                time_term = time_term + data.t[arc_orders[i][0],arc_orders[i][1]]
                
            
        total_prob_find = total_prob_find + prob_term
    
                
        expected_time = expected_time + prob_term*time_term
    
    expected_time = expected_time/total_prob_find 
        
    return expected_time





# This function returns expected time of travel given a decided path

def expected_time_travel(arc_orders, data): 
    
    expected_time = 0
    
    for k in range(1,len(arc_orders)+1):
        prob_term = 1 
        time_term = 0 

        for i in range(0,k): 
            if i== k-1:
                prob_term = prob_term*data.p[arc_orders[i][1]]
                time_term = time_term + data.t[arc_orders[i][0],arc_orders[i][1]]
                
            else: 
                prob_term = prob_term*(1 - data.p[arc_orders[i][1]])
                time_term = time_term + data.t[arc_orders[i][0],arc_orders[i][1]]        
                
        expected_time = expected_time + prob_term*time_term 
         
    k = len(arc_orders)
    prob_term = 1
    time_term = 0
    for i in range(0,k):  
        prob_term = prob_term*(1 - data.p[arc_orders[i][1]])
        time_term = time_term + data.t[arc_orders[i][0],arc_orders[i][1]]
    
    expected_time = expected_time + prob_term*time_term 
    
    return expected_time







# function returns the probabilty of fidning the entity on the path
def probability(arc_orders, data):
    
    total_prob_find = 0
    for k in range(1,len(arc_orders)+1):
        prob_term = 1
        for i in range(0,k): 
            if i== k-1:
                prob_term = prob_term*data.p[arc_orders[i][1]]
        
            else: 
                 prob_term =  prob_term*(1 - data.p[arc_orders[i][1]])
                 
        total_prob_find = total_prob_find + prob_term
                 
    return total_prob_find


# function returns the distanc travelled on the path
def distance(arc_orders, data): 
    distance = 0
    for x in arc_orders:
        distance = distance + data.d[x[0],x[1]]                  
    return distance

