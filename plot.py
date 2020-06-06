import matplotlib.pyplot as plt


def plot_coordinates(xc, yc):

    plt.plot(xc[0], yc[0], c='r', marker = 's')
    plt.scatter(xc[1:], yc[1:], c = 'b')
    
    
    
def plot_path(xc, yc, arc_orders):
    # Plotting the path 
    if len(arc_orders[1]) ==3 : 
        for i,j,k in arc_orders:
            if j!=0:
                plt.plot([xc[i], xc[j]],[yc[i],yc[j]], c='g')
    else :
         for i,j in arc_orders:
            if j!=0:
                plt.plot([xc[i], xc[j]],[yc[i],yc[j]], c='g')
    
    plt.plot(xc[0], yc[0], c='r', marker = 's')
    plt.scatter(xc[1:], yc[1:], c = 'b')