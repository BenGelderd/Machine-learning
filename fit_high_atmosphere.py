import numpy as np
import matplotlib.pyplot as plt 
import scipy.optimize

#read and unpack data into altitude and density parameters
data = np.genfromtxt("data/AthmDensity.csv", delimiter=",", skip_header=1)
altitude, density = zip(*data)
altitude = np.array(altitude)
density = np.array(density)
log_density = np.log(density)

#provided constants 
B = 8.82e7 
h0 = 1. 

"""
Developing function which we will use to generate our parameters and plot our
logarithmic fit with.
We will use the log of the data and function F to plot our fit by raising it to
the exponential for the plot, by doing this we better fit our data for larger
altitudes.

"""
def F(h, a, l, sigma):
    phi = a * np.exp(-1 * h * l) + B * ( (h/h0)**(-1 * sigma) ) 
    return phi

def f2(h, a, l, sigma):
    e = np.log(a * np.exp(-1 * h * l) + B * ( (h/h0)**(-1 * sigma)) )
    return e

#fit the data converted into kms with given parameters for a,l and sigma
popt, pcov, = scipy.optimize.curve_fit(f2, altitude, log_density, p0 = \
                                       [1.96, 0.15, 7.57])
popt1, pcov1 = scipy.optimize.curve_fit(F, altitude, density, \
                                        p0=[1.946, 0.15, 7.57]) 

#print values for a,l and sigma
print("a = ", popt1[0], "l = ", popt1[1], "s = ", popt1[2])
plt.title("Atmopheric density - Altitude")
plt.xlabel("h (km)", fontsize=14) #labelling graph
plt.ylabel("dens (kg/m^3)", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout(rect=[0, 0, 1, 1]) # Ensure the full figure is visible
plt.grid(True, which="both", linestyle="--", linewidth=0.5) #creates grid

#plot logartihmic graph for the data and fitted line
plt.plot(altitude, density, "b*")               # Plot data
plt.plot(altitude, np.exp(f2(altitude, *popt)), 'r--') #plot fitted curve
plt.semilogy()
plt.tight_layout()
plt.show()