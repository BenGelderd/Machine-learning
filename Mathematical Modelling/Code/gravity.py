from ode_rk4 import ODE_RK4
import math
import numpy as np

class Gravity(ODE_RK4):
  def __init__(self,V0=[0], dt=0.1, t0=0, h=0.0):
      """ V0 : initial conditions for z, v_z, phi, v_phi.
          dt : integration time step
          t0 : initial time
          h : altitude of the target orbit
      """
      super().__init__(V0,dt,t0)
      self.h = h
      self.G = 6.673e-11
      self.M_E = 5.97e24
      self.R_E = 6370e3
      self.r_0 = self.R_E+h
      self.omega_0 = math.sqrt(self.G*self.M_E/self.r_0**3)
      self.F_r = 0
      self.F_theta = 0
      self.m = 1
      self.t_last = 0
      self.d_last = -1
      self.d_butlast = -1
      self.d_min = []
      self.d_max = []

  def reset(self, V0, dt, t0=0):
      """ Reset the integration parameters; see __init__ for more info."""
      super().reset(V0, dt, t0)
      self.t_last = 0
      self.d_last = -1
      self.d_butlast = -1
      self.d_min = []
      self.d_max = []

  def F(self,t,v):
      """ Equation to solve: 
          v[0] is z 
          v[1] is dz/dt (V_z in essay notation)
          v[2] is phi 
          v[3] is dphi/dt (V_phi in essay notation)
          
          V_z_p is the radial veloctity and equivalent to the second diff of z
          V_phi_p is equivalent to the sond diff of phi
      """
      
      #equations of motion
      V_z_p = -1.0 * self.G*self.M_E/((self.r_0 + v[0])**2) \
          + (self.r_0 + v[0])*((self.omega_0+v[3])**2) + self.F_r/self.m    
              
      V_phi_p = -2.0 * (self.omega_0 + v[3]) * v[1] / (self.r_0 + v[0]) \
          + self.F_theta/(self.m * (self.r_0 + v[0]))
          
      #equations generated from question 3 as array for computation
      return np.array([v[1], V_z_p, v[3], V_phi_p], dtype='float64')

      
  def dist_2_reference(self):
    """
    Calculate the distance between spacecraft and reference trajectory.
    takes the Euclidean distance in polar coordinates.
    """
    r = self.r_0 + self.V[0]  # Orbital radius
    d = math.sqrt(self.V[0]**2 + (r * self.V[2])**2 + 2*r*self.V[0]*\
                  (1 - math.cos(self.V[2]))) 
    return d
  
  def post_integration_step(self):
    """
     Isolating distances into a list of minima 

    """
    new_d = self.dist_2_reference()

    if self.d_butlast > self.d_last < new_d:
        self.d_min.append([self.t_last, self.d_last])
    
    self.d_butlast = self.d_last  # Update distances
    self.d_last = new_d
    self.t_last = self.t  # Update time
    pass
    
  def min_min(self, t_after=0):
      """
      Filtering minima and returning the closest minima to debris possible

      """
      
      self.min_val = [
              [t, d] for t, d in self.d_min if t > t_after 
          ]
      if not self.min_val:
          return None
      
      min_lst = self.min_val[0]
      for sublist in self.min_val:
          if sublist[1] < min_lst[1]:
              min_lst = sublist
              
      return min_lst
  
