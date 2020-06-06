import operator
import copy

## Function for the greedy heurestic

def greedy1(data):
    
    data1 = copy.deepcopy(data)

    n = data1.n
    V= data1.V
    N= data1.N
    d = data1.d
    p= data1.p
    #d = data1.d
    f = data1.f
    mi = data1.mi
    t = data1.t
    ratios = {i : {(i,j): t[i,j]/p[j] for j in N if i!=j} for i in V}
    d1= {i: {(i,j): d[i,j] for j in N if i!=j} for i in V}
    arc_orders = []
    distance_left = f*mi
    #visited_nodes=[0]*(data1.n+1)
    i = 0
    V1 = V
    
    while len(arc_orders) != n:
        arc_options = ratios[i]
        sorted_arcs= sorted(arc_options.items(), key=operator.itemgetter(1))
        shortest_distance = sorted(d1[i].items(), key=operator.itemgetter(1))[0][1]
        #sorted(d.items(), key=operator.itemgetter(1))[n+1][1]
        if distance_left > shortest_distance:
            arc_orders.append(sorted_arcs[0][0])
            #count = len(sorted_arcs)
            distance_left = distance_left - d1[i][sorted_arcs[0][0]]
            ratios.pop(i)
            d1.pop(i)
            V1.remove(i)
            #i = sorted_arcs[0][0][1]
            if i !=0 : 
                for j in V1: 
                    ratios[j].pop((j,i))
                    d1[j].pop((j,i))
            i = sorted_arcs[0][0][1]
        else: 
            break

    return arc_orders
        
            
 







