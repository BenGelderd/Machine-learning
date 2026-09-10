
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd


class Schwarzschild_Photon:
    """
    Class to integrate photon trajectories in the equatorial plane 
    of a Schwarzschild black hole.
    """
    def __init__(self, M=1.0, r_cam=800.0, R_sky=1000.0, beta=0.0):
        self.M = M
        self.r_cam = r_cam
        self.R_sky = R_sky
        self.beta = beta
        self.E = 1.0 # Set Energy to 1 (affine parameter scaling)

        # For L we use special relaitivty, E = c|p| for a photon
        # we account for its tangential component, if we were to account for redshift we would scale by gamma
        self.L = self.r_cam * np.sin(self.beta) / np.sqrt(1.0 - 2.0 * self.M / self.r_cam)
        self.solution = None

    def get_initial_conditions(self):
        """Returns [r(0), s(0), phi(0)] based on the V^2 potential equation."""
        #V^2 = (1 - 2M/r) * L^2/r^2
        v2 = (1.0 - 2.0 * self.M / self.r_cam) * (self.L**2 / self.r_cam**2)
        v_r_sq = self.E**2 - v2
        
        # Ray starts moving inward (towards the black hole), so radial velocity s is negative
        s0 = -np.sqrt(v_r_sq) if v_r_sq > 0 else 0.0
        return [self.r_cam, s0, 0.0]

    def rhs(self, tau, y):
        """
        EOMs
        y = [r, s, phi] where s = r_dot
        """
        r, s, phi = y
        
        # Prevent division by zero at singularity
        if r <= 0: return [0, 0, 0] 
        
        dr_dtau = s
        ds_dtau = - (self.L**2 * (3.0 * self.M - r)) / (r**4)
        dphi_dtau = self.L / (r**2)
        
        return [dr_dtau, ds_dtau, dphi_dtau]

    # --- Events ---
    def event_horizon(self, tau, y):
        return y[0] - 2.0 * self.M
    event_horizon.terminal = True #occurs when return vale = 0
    event_horizon.direction = -1 #determines whether the value is decreasing when it hits 0

    def event_escape(self, tau, y):
        return y[0] - self.R_sky
    event_escape.terminal = True
    event_escape.direction = 1

    def integrate(self, max_tau=3000.0, max_step=5.0):
        """Runs the solve_ivp integration."""
        y0 = self.get_initial_conditions()
        self.solution = solve_ivp(
            fun=self.rhs, 
            t_span=(0, max_tau), 
            y0=y0,
            events=[self.event_horizon, self.event_escape],
            max_step=max_step, # Keeps trajectory points dense enough for smooth plotting
            rtol=1e-5, atol=1e-5
        )
        return self.solution

#------------------------------------------

def modulo_pi(angle):
    """Normalize angle to [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


if __name__ == '__main__':
    # Configuration
    M_BH = 20.0
    R_CAMERA = 800.0
    R_CELESTIAL_SKY = 1000.0
    
    # Setup angles: number of rays between -45 and 45 degrees
    beta_angles = np.linspace(-np.pi/3, np.pi/3, 100).tolist()
    
    # Calculate critical capture angle dynamically to ensure we get a highlighted ray
    # Critical impact parameter b_crit = 3*sqrt(3)*M
    b_crit = 3.0 * np.sqrt(3.0) * M_BH
    sin_beta_crit = b_crit / R_CAMERA
    beta_crit_deg = np.degrees(np.arcsin(sin_beta_crit))
    

    
    # Inject a ray just slightly outside the capture zone to orbit the photon sphere
    beta_highlight = np.deg2rad(beta_crit_deg + 0.005)
    beta_angles.append(beta_highlight)
    beta_angles.sort()

    results_data = []
    escaped_paths = []

    print(f"Tracing {len(beta_angles)} rays from camera at r={R_CAMERA}...")

    # Run the integration for each angle
    for beta in beta_angles:
        photon = Schwarzschild_Photon(M=M_BH, r_cam=R_CAMERA, R_sky=R_CELESTIAL_SKY, beta=beta)
        sol = photon.integrate()
        
        # Check termination events
        if sol.status == 1 and len(sol.t_events[1]) > 0:
            # Event 1 triggered: Escaped to Celestial Sphere
            r_path, s_path, phi_path = sol.y
            final_phi_deg = phi_path[-1]
            
            results_data.append({"beta": beta, "phi": final_phi_deg})
            
            # Save coordinates for Plot 1
            x_path = r_path * np.cos(phi_path)
            y_path = r_path * np.sin(phi_path)
            escaped_paths.append({"beta": beta, "x": x_path, "y": y_path})
            
        else:
            # Event 0 triggered: Captured by Event Horizon
            results_data.append({"beta": beta, "phi": np.nan})
            
    # Save data to CSV
    df = pd.DataFrame(results_data)
    df.to_csv("2d_data.csv", index=False)
    print("Data saved to '2d_data.csv'.")

#------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

 # --- PLOT 1: 2D Spacetime Visualization ---
ax1.set_title(f"2D Ray Tracing (M={M_BH})", fontsize=14)
ax1.set_xlabel("x / M")
ax1.set_ylabel("y / M")
ax1.set_aspect('equal')


# Draw Boundaries
ax1.add_patch(plt.Circle((0, 0), 2.0*M_BH, color='black', zorder=5, label='Event Horizon (2M)'))
ax1.add_patch(plt.Circle((0, 0), 3.0*M_BH, color='red', fill=False, ls='--', lw=1.5, zorder=4, label='Photon Sphere (3M)'))
ax1.add_patch(plt.Circle((0, 0), R_CELESTIAL_SKY, color='red', fill=False, ls=':', alpha=0.3, zorder=1, label='Celestial Sphere'))

#Add Quadrant Axes and Angle Labels 
ax1.axhline(0, color='white', linestyle='--', alpha=0.2, zorder=0)
ax1.axvline(0, color='white', linestyle='--', alpha=0.2, zorder=0)

# Add text labels just outside the celestial sphere
    # 0 degrees (Right)
ax1.text(R_CELESTIAL_SKY + 0.05, 0, r'$\Phi = 0^\circ$', color='black', 
             ha='left', va='center', fontsize=11, fontweight='bold')
    # 90 degrees (Top)
ax1.text(0, R_CELESTIAL_SKY + 0.05, r'$\Phi = 90^\circ$', color='black', 
             ha='center', va='bottom', fontsize=11, fontweight='bold')
    # 180 degrees (Left)
ax1.text(-R_CELESTIAL_SKY - 0.15, -(R_CELESTIAL_SKY * 0.12), r'$\Phi = \pm 180^\circ$', color='black', 
            ha='right', va='center', fontsize=11, fontweight='bold')
    # -90 degrees (Bottom)
ax1.text(0, -R_CELESTIAL_SKY - 0.05, r'$\Phi = -90^\circ$', color='black', 
             ha='center', va='top', fontsize=11, fontweight='bold')
    # ------------------------------------------------------

ax1.scatter(R_CAMERA, 0, color='blue', marker='D', s=30, zorder=6, label=f'Camera ({R_CAMERA}M)')

    # Draw Rays
for path in escaped_paths:
    if np.isclose(path["beta"], beta_highlight):
        ax1.plot(path["x"], path["y"], color='red', lw=2.0, zorder=3, label=f'Highlighted Ray ($\\beta \\approx {path["beta"]:.3f}^\\circ$)')
    else:
        ax1.plot(path["x"], path["y"], color='orange', lw=0.5, alpha=0.6, zorder=2)

ax1.set_xlim(-R_CELESTIAL_SKY*1.1, R_CELESTIAL_SKY*1.1)
ax1.set_ylim(-R_CELESTIAL_SKY*1.1, R_CELESTIAL_SKY*1.1)
ax1.legend(loc="upper left")

    # --- PLOT 2: Beta vs Phi Angle Map ---
valid_df = df.dropna() # Omit captured rays (NaNs)

ax2.set_title("Deflection Angle ($\\Phi$) vs. Observation Angle ($\\beta$)", fontsize=14)
ax2.set_xlabel("Observation Angle $\\beta$ at Camera (degrees)")
ax2.set_ylabel("Deflection Angle $\\Phi$ on Celestial Sphere (degrees)")
ax2.grid(True, linestyle='--', alpha=0.7)

    #Plot ALL valid rays as individual dots (linestyle='' stops them from linking)
ax2.plot(np.degrees(modulo_pi(valid_df["beta"])), np.degrees(modulo_pi(valid_df["phi"])), marker='o', linestyle='', color='cyan', 
             markeredgecolor='blue', zorder=3, label='Standard Rays')

    # Extract the specific phi value for our highlighted ray
    # Use np.isclose because float values are rarely exactly equal
phi_highlight = valid_df.loc[np.isclose(valid_df["beta"], beta_highlight), "phi"]

    # Plot that single point in red right on top of the graph!
if not phi_highlight.empty:
    ax2.plot(np.degrees(modulo_pi(beta_highlight)), np.degrees(modulo_pi(phi_highlight)), marker='o', linestyle='', color='red', 
                markeredgecolor='darkred', markersize=8, zorder=4, 
                label=f'Highlighted Ray ($\\beta \\approx {np.degrees(beta_highlight):.3f}^\\circ$)')
    
#Add cutoff lines 
# Add vertical capture boundaries at +/- beta_crit
ax2.axvline(beta_crit_deg, color='black', linestyle='--', alpha=0.5, label='Capture Boundary $\\approx 3\\sqrt{3}M$')
ax2.axvline(-beta_crit_deg, color='black', linestyle='--', alpha=0.5) # No label so it doesn't duplicate in legend

ax2.legend(loc="upper left")

#plt.savefig("2d_plot_Schwarzschild.png", dpi=300) # Save the figure with high resolution

plt.tight_layout()
plt.savefig("2d_plot_Schwarzschild.png", dpi=300) # Save the figure with high resolution
plt.show()

