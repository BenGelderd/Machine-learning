import gravity as gr
import math
import numpy as np
import matplotlib.pyplot as plt
from spacecraft import Spacecraft
import random


random_colour = ["r", "b", "g"]

def process(Fr, Ftheta, t_thrust, col):
    dt = 0.1
    gdt = 1
    s = Spacecraft([0,0,0,0],dt,0, 4e5,4000) # initial condition set later
    tmax= 5400
    Fr = Fr
    Ftheta = Ftheta
    t_thrust = t_thrust
    
    tmin = s.min_dist_to_target(Fr,Ftheta,t_thrust,tmax,dt,gdt)[0]
    dmin = s.min_dist_to_target(Fr, Ftheta, t_thrust, tmax, dt)[1]
    print("Fr={} Ftheta={} t_thrust={}".format(Fr,Ftheta,t_thrust))
    print("dmin={}, tmin={} Fuel={}"\
      .format(dmin ,tmin,(abs(Fr)+abs(Ftheta))*t_thrust))
    s.plot(1,3, col + '-');
    plt.xlabel('phi',fontsize=20)
    plt.ylabel('z',fontsize=20)
    plt.plot([0],[0], 'rx');


process(1.89, 100., 73.3, "b")
process(32., 100., 266., "g")
plt.tight_layout()
plt.savefig("orbits.png")



plt.show()

