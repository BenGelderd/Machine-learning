import numpy as np
import matplotlib.pyplot as plt
import math


# Constants in correct units
B = 4.63e30 
sigma = 7.57
G = 6.673e-11  
M_e = 5.97e24  
R_e = 6.37e6  


h = np.linspace(3e5, 1e6, 500)  # Altitudes in m
h_conv = np.linspace(300, 1000, 500)


# Re-entry time function
def reentry_time(M_A):
    """
    Compute re-entry time in years for a given mass-to-area ratio m/A.
    This is calculated in metres according to the constant units as 
    such we will need to convert account for h into km.
    m/A can be treated as a constant as agrees with our parameter units.
    """
    
    t = (h**(sigma + 1)) / (4 * B * math.sqrt(G * M_e * (R_e))*(sigma + 1)) * M_A 
    
    return (t / 3.1536e7)  # Convert seconds to years and h into km


# Compute for different cases
case_1 = reentry_time(27)  # Bolt
case_2 = reentry_time(54)  # Rod
case_3 = reentry_time(5.4)  # Plate
case_4 = reentry_time(544.66)  # Gemini

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(h_conv, case_1, "k--", label="Bolt (M/A=27)")
plt.plot(h_conv, case_2, "g--", label="Rod (M/A=54)")
plt.plot(h_conv, case_3, "b--", label="Plate (M/A=1.35)")
plt.plot(h_conv, case_4, "r--", label="Gemini (M/A=544.66)")
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.xlabel("Altitude (km)")
plt.ylabel("re-entry time (years)")
plt.title("Re-entry Time vs Altitude for Different Debris Types")
plt.semilogy()  # Logarithmic scale for time
plt.legend()
plt.tight_layout()
plt.savefig("time_plots.png")
plt.show()