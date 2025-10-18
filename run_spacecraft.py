import gravity as gr
import math
import numpy as np
import matplotlib.pyplot as plt
from spacecraft import Spacecraft

def process(Fr, Ftheta, t_thrust, col):
    dt = 0.1
    gdt = 1
    s = Spacecraft([0,0,0,0],dt,0,4e5,4000.) # initial condition set later
    tmax= 4000
    Fr = Fr
    Ftheta = Ftheta
    t_thrust = t_thrust
    
    tmin = s.min_dist_to_target(Fr,Ftheta,t_thrust,tmax,dt,gdt)[0]
    dmin = s.min_dist_to_target(Fr, Ftheta, t_thrust, tmax, dt)[1]
    print("Fr={} Ftheta={} t_thrust={}".format(Fr,Ftheta,t_thrust))
    print("dmin={}, tmin={} Fuel={}"\
      .format(dmin ,tmin,(abs(Fr)+abs(Ftheta))*t_thrust))
    s.plot(1,3,col + '-');
    plt.xlabel('phi',fontsize=20)
    plt.ylabel('z',fontsize=20)
    plt.plot([0],[0],col + '+');

process(50, 100., 100, "b")
process(25, 50., 200, "g")
process(10, 20., 500, "r")
plt.show()