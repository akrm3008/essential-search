import numpy as np
import pandas as pd
from test_data import created_data, random_data
from model import minimise_expected_time, maximise_probability 
from plot import plot_coordinates, plot_path
from objectives import expected_time_find_gas, expected_time_travel, probability, distance 
from heurestic import greedy1


# Created data to  test 
n = 3    
xc = np.array([0, 0, 1, 1])  
yc = np.array([0, 1, 0, 1])  
p = {0: 0.0000001,
     1: 0.3,    
     2: 0.2,
     3: 0.7}
f = 2
mi =  10 


test_data = created_data(n, p, xc, yc, f, mi)
test_data.generate_parmaters()
test_data.create_paths()


exp_time_find_gas =[]
exp_time_travel = []
probability_find_gas =[]
path_distance = []



# Comparing the objectives of different paths (Brute force approach)

for x in test_data.paths:
    exp_time_find_gas.append(expected_time_find_gas(x, test_data))
    exp_time_travel.append(expected_time_travel(x, test_data))
    probability_find_gas.append(probability(x, test_data))   
    path_distance.append(distance(x,test_data))

path_quality = pd.DataFrame(list(zip(test_data.paths, path_distance ,exp_time_travel, exp_time_find_gas, probability_find_gas)),
                           columns = ['path', 'path_distance', 'expected_time_travel', 'expected_time_findgas',
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   'proabability_findgas'] )

path_quality.to_csv()


# Modelling and soliving models using Gurobi slover and Heurestic 
    
arc_orders = minimise_expected_time(test_data)
arc_orders2 = maximise_probability(test_data)
arc_orders3 = greedy1(test_data)



# Plotting 

plot_coordinates(xc, yc)

plot_path(xc, yc, arc_orders)
plot_path(xc, yc, arc_orders2)
plot_path(xc, yc, arc_orders3)


expected_time_find_gas(arc_orders, test_data)
expected_time_travel(arc_orders, test_data)
probability(arc_orders, test_data)


# Random data testing 

n = [5]
seeds = [x for x in range(1,2)]
data = [[i, j, random_data(i,j)]  for i in n for j in seeds]
for x in data:
    x[2].create_paths()
    
    
comparison_list = [[x[0], x[1], y, distance(y,x[2]), expected_time_find_gas(y,x[2]), expected_time_travel(y,x[2]), probability(y,x[2])] 
                    for x in data for y in x[2].paths]



comparison_df = pd.DataFrame(comparison_list, columns = ['n','seed', 'path', 'path_distance', 'expected_time_travel', 'expected_time_findgas',
                                       'proabability_findgas'])




