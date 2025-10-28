import gravity as gr
import math
import numpy as np
import matplotlib.pyplot as plt

class Spacecraft(gr.Gravity):

      def __init__(self,V0=[0], dt=0.1, t0=0,h=0.0,m=1.0):
         super().__init__(V0, dt, t0, h)
         self.m = m # When F is non null the mass plays a role
         
      def F(self,t,v):
          """ Equation to solve: 
              v[0] is z .
              v[1] is dz/dt (V_z in essay notation).
              v[2] is phi .
              v[3] is dphi/dt (V_phi in essay notation).
              
              V_z_p is the radial veloctity and equivalent to the second 
              diff of z.
              V_phi_p is equivalent to the sond diff of phi.
          """
          

          
          #Making force time dependent
          if t > self.t_thrust:
              self.F_r = 0
              self.F_theta = 0
          
          #equations of motion
          
          V_z_p = -1.0 * self.G*self.M_E/((self.r_0 + v[0])**2) \
              + (self.r_0 + v[0])*((self.omega_0+v[3])**2) + self.F_r/self.m    
                  
          V_phi_p = -2.0 * (self.omega_0 + v[3]) * v[1] / (self.r_0 + v[0]) \
              + self.F_theta/(self.m * (self.r_0 + v[0]))
              
            
          #equations generated from question 3 as array for computation
          return np.array([v[1], V_z_p, v[3], V_phi_p], dtype='float64')
  

      def min_dist_to_target(self, Fr, Ftheta, t_thrust, tmax, dt, gdt=1):
        """ Integrate equation and determine min distance to target.
          : param Fr     : radial thrust (perpendicular to orbit).
          : param Ftheta : horizontal thrust (parralet to orbit).
          : param t_thrust : duration of thrust.
          : param tmax : duration of integration.
          : param dt : integration time step.
        
        Computes the trajectory and determines the minimum distance to 
        the debris.
        """
        # Store thrust parameters as class variables
        self.F_r = Fr
        self.F_theta = Ftheta
        self.t_thrust = t_thrust

        z0 = -1000
        v_z0 = 0
        phi0 = -2000.0 / self.r_0
        v_phi0 = math.sqrt(self.G * self.M_E / ((self.r_0 + z0) ** 3)) \
            - self.omega_0

        V0 = [z0, v_z0, phi0, v_phi0] #initial position of spacecraft
        self.reset(V0, dt, t0=0)
        self.iterate(tmax)

        distance = self.min_min(t_after=0) #obtain the minimum distance and time
        if distance is None:
            return None, None

        t_min , d_min = distance #unpacking the list
        fuel = (abs(Fr) + abs(Ftheta)) * t_thrust
        return t_min, d_min, fuel

if __name__ == "__main__":
    """
    Output of the 3 examples given.
    """
    def process(Fr, Ftheta, t_thrust, col):
        dt = 0.1
        gdt = 1
        s = Spacecraft([0,0,0,0],dt,0,4e5, 4000.) # initial condition set later
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
        plt.plot([0],[0],col + 'x')
        

    process(50, 100., 100, "b")
    process(25, 50., 200, "g")
    process(10, 20., 500, "r")
    plt.tight_layout()
    plt.show()
