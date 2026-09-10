import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import pandas as pd
from sympy import beta

def modulo_pi(angle):
#Normalize angle to [-pi, pi].
    return (angle + np.pi) % (2 * np.pi) - np.pi

class Kerr_Photon:
    """
    Class to integrate photon trajectories in the equatorial plane 
    of a Kerr black hole.
    """
    def __init__(self, M=100.0, a_star=0.99, r_cam=800.0, R_sky=1000.0, beta=0.0):
        self.M = M
        self.a = a_star * self.M  
        self.r_cam = r_cam
        self.R_sky = R_sky
        self.beta = beta
        
        # Constants of Motion
        self.L_z = self.r_cam * np.sin(self.beta) 
        self.K = (self.L_z - self.a)**2
        
        # Event Horizon
        self.r_horizon = self.M + np.sqrt(self.M**2 - self.a**2)

    def get_initial_conditions(self):
        R_K = ((self.r_cam**2 + self.a**2) - self.a * self.L_z)**2 - \
              (self.r_cam**2 - 2.0*self.M*self.r_cam + self.a**2) * self.K
        p_r_0 = -np.sqrt(max(R_K, 0.0))
        return [self.r_cam, 0.0, p_r_0]

    def rhs(self, lam, state):
        r, phi, p_r = state
        Delta = r**2 - 2.0*self.M*r + self.a**2
        
        dr_dlam = p_r
        dphi_dlam = (self.a / Delta) * (r**2 + self.a**2 - self.a*self.L_z) - (self.a - self.L_z)
        dp_r_dlam = 2.0*r*(r**2 + self.a**2 - self.a*self.L_z) - (r - self.M)*self.K
        
        return [dr_dlam, dphi_dlam, dp_r_dlam]

    def integrate(self):
        def event_horizon(lam, state):
            return state[0] - self.r_horizon * 1.01
        event_horizon.terminal = True
        
        def event_sky(lam, state):
            # Stop integrating when the ray hits the Celestial Sphere
            return state[0] - self.R_sky 
        event_sky.terminal = True
        
        sol = solve_ivp(
            self.rhs,
            (0, 1e5), 
            self.get_initial_conditions(),
            events=[event_horizon, event_sky],
            max_step=5.0,
            rtol=1e-8, atol=1e-8 
        )
        return sol

# --- Main Execution and Plotting ---
if __name__ == "__main__":
    M = 50.0
    a_star = 0.99
    r_cam = 900.0
    R_sky = 1000.0

    r_prograde = 2*M * (1 + np.cos((2/3) * np.arccos(-a_star)))
    r_retrograde = 2*M * (1 + np.cos((2/3) * np.arccos(a_star)))

    # Fire a fan of rays from -60 to +60 degrees
    angles = np.linspace(-np.pi/4, np.pi/4, 100).tolist()

    b_crit_prograde = (r_prograde**2 * (r_prograde - 3*M) + (a_star*M)**2 * (r_prograde+ M)) / (a_star*M * (M - r_prograde))
    b_crit_retrograde = (r_retrograde**2 * (r_retrograde - 3*M) + (a_star*M)**2 * (r_retrograde + M)) / (a_star*M * (M - r_retrograde))
    # Inject a ray just slightly outside the capture zone to orbit the photon sphere
    sin_beta_1 = np.clip(b_crit_prograde / r_cam, -1.0, 1.0)
    sin_beta_2 = np.clip(b_crit_retrograde / r_cam, -1.0, 1.0)

    beta_crit_prograde = np.arcsin(sin_beta_1)
    beta_crit_retrograde = np.arcsin(sin_beta_2)

    beta_highlight_prograde = beta_crit_prograde + 0.0001  # Slightly outside the prograde capture zone
    angles.append(beta_highlight_prograde)
    beta_highlight_retrograde = beta_crit_retrograde - 0.0001  # Slightly outside the retrograde capture zone
    angles.append(beta_highlight_retrograde)
    angles.sort()

    results_data = []
    escaped_paths = []
    # 4. Integrate and plot the rays
    for angle in angles:
        photon = Kerr_Photon(M=M, a_star=a_star, r_cam=r_cam, R_sky=R_sky, beta=angle)
        sol = photon.integrate()
        
        # Check termination events
        if sol.status == 1 and len(sol.t_events[1]) > 0:
            # Event 1 triggered: Escaped to Celestial Sphere
            r_traj = sol.y[0]
            phi_traj = sol.y[1]
            fina_phi = phi_traj[-1]
        
            x = r_traj * np.cos(phi_traj)
            y = r_traj * np.sin(phi_traj)
            
            results_data.append({"beta": angle, "phi": (fina_phi % (2 * np.pi)) })
            color = 'orange'
            alpha =  1.0

            escaped_paths.append({"beta": angle, "x": x, "y": y})
            
        else:
            # Event 0 triggered: Captured by Event Horizon
            results_data.append({"beta": angle, "phi": np.nan})
        
        #plt.plot(x, y, color=color, alpha=alpha, linewidth=1.0)
    
    df = pd.DataFrame(results_data)
    #df.to_csv("2d_data_Kerr.csv", index=False)
    print("Data saved to '2d_data_Kerr.csv'.")
    
    print(f"Firing {len(angles)} rays from camera...")

    """"""""""""""""""""""""""""""""""""""""""""""""
    #Plotting
    """"""""""""""""""""""""""""""""""""""""""""""""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    ax1.set_title(f"Kerr Lensing (M={M}, a/M={a_star})", fontsize=14)
    
    # 1. Plot the Event Horizon (Solid Black)
    r_H = M + np.sqrt(M**2 - (a_star*M)**2)
    horizon = plt.Circle((0, 0), r_H, color='black', zorder=2, label="Event Horizon")
    ax1.add_patch(horizon)

    # 2. Plot the Kerr Photon Rings (Red Dashed)
    ph1 = plt.Circle((0, 0), r_prograde, color='blue', linestyle='--', fill=False, zorder=6, label="Prograde Photon Ring")
    ph2 = plt.Circle((0, 0), r_retrograde, color='blue', linestyle='--', fill=False, zorder=6, label="Retrograde Photon Ring")
    ax1.add_patch(ph1)
    ax1.add_patch(ph2)

    # 3. Plot the Celestial Sphere (Red Dashed)
    sky = plt.Circle((0, 0), R_sky, color='red', linestyle='--', fill=False, zorder=2, label="Celestial Sphere")
    ax1.add_patch(sky)

    for path in escaped_paths:
        if np.isclose(path["beta"], beta_highlight_prograde) or np.isclose(path["beta"], beta_highlight_retrograde):
            ax1.plot(path["x"], path["y"], color='red', alpha = 0.8, lw=2.0, label=f'Highlighted Ray ($\\beta \\approx {np.rad2deg(path["beta"]):.3f}^\\circ$)')
        else:
            ax1.plot(path["x"], path["y"], color=color, lw=1.0, alpha=alpha, zorder=3)

    # Plot Camera Location
    ax1.scatter([r_cam], [0], color='gold', edgecolor='black', s=100, zorder=10, label="Camera")

    ax1.set_xlabel("X (M)")
    ax1.set_ylabel("Y (M)")
    
    # Expand limits to fit the entire celestial sphere
    ax1.set_xlim(-R_sky * 1.05, R_sky * 1.05)
    ax1.set_ylim(-R_sky * 1.05, R_sky * 1.05)
    
    ax1.set_aspect('equal')
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc="upper right")

    #Add Quadrant Axes and Angle Labels 
    ax1.axhline(0, color='white', linestyle='--', alpha=0.2, zorder=0)
    ax1.axvline(0, color='white', linestyle='--', alpha=0.2, zorder=0)

    # Add text labels just outside the celestial sphere
        # 0 degrees (Right)
    ax1.text(R_sky + 0.05, 0, r'$\Phi = 0^\circ$', color='black', 
                ha='left', va='center', fontsize=11, fontweight='bold')
        # 90 degrees (Top)
    ax1.text(0, R_sky + 0.05, r'$\Phi = 90^\circ$', color='black', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        # 180 degrees (Left)
    ax1.text(-R_sky - 0.15, -(R_sky * 0.12), r'$\Phi = \pm 180^\circ$', color='black', 
                ha='right', va='center', fontsize=11, fontweight='bold')
        # -90 degrees (Bottom)
    ax1.text(0, -R_sky - 0.05, r'$\Phi = -90^\circ$', color='black', 
                ha='center', va='top', fontsize=11, fontweight='bold')
        # ------------------------------------------------------

    valid_df = df.dropna()
    #highlighted ray on the beta vs phi plot
    # We use np.isclose because float values are rarely exactly equal
    phi_highlight_prograde = valid_df.loc[np.isclose(valid_df["beta"], beta_highlight_prograde), "phi"]
    phi_highlight_retrograde = valid_df.loc[np.isclose(valid_df["beta"], beta_highlight_retrograde), "phi"]

    # 3. Plot that single point in red right on top of the graph!
    if not phi_highlight_prograde.empty:
        ax2.plot(np.degrees(modulo_pi(beta_highlight_prograde)), np.degrees(modulo_pi(phi_highlight_prograde)), marker='o', linestyle='', color='red', 
                    markeredgecolor='darkred', markersize=8, zorder=4, 
                    label=f'Highlighted Ray ($\\beta \\approx {np.degrees(beta_highlight_prograde):.3f}^\\circ$)')

    if not phi_highlight_retrograde.empty:
        ax2.plot(np.degrees(modulo_pi(beta_highlight_retrograde)), np.degrees(modulo_pi(phi_highlight_retrograde)), marker='o', linestyle='', color='red', 
                    markeredgecolor='darkred', markersize=8, zorder=4, 
                    label=f'Highlighted Ray ($\\beta \\approx {np.degrees(beta_highlight_retrograde):.3f}^\\circ$)')

    ax2.set_title("Deflection Angle vs. Incoming Angle", fontsize=14)
    ax2.set_xlabel("Incoming Angle $\\beta$ (degrees)")
    ax2.set_ylabel("Deflection Angle $\\phi$ (degrees)")
    ax2.plot(np.rad2deg(modulo_pi(df["beta"])), np.rad2deg(modulo_pi(df["phi"])), 
             color='blue', marker='o', linestyle='', markersize=5, label="Deflection Data")
    # Add vertical capture boundaries at +/- beta_crit
    ax2.axvline(np.rad2deg(beta_crit_prograde), color='red', linestyle='--', alpha=0.5, label='Prograde Capture Boundary')
    ax2.axvline(np.rad2deg(beta_crit_retrograde), color='black', linestyle='--', alpha=0.5, label='Retrograde Capture Boundary') 
    ax2.legend()
    ax2.grid(True, linestyle=':', alpha=0.6)    

    #plt.savefig("2d_plot_Kerr.png", dpi=300)
    print("Plot saved as '2d_plot_Kerr.png'.")
    plt.show()