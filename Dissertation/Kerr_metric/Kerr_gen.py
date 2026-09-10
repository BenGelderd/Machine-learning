import numpy as np
from scipy.integrate import solve_ivp
from PIL import Image
import concurrent.futures
import time
import sys

"""
Use a similar class initialisation adn function to the 2D plot but with the added parameters
required for image generation.
"""
class Kerr_Raytracer:
    def __init__(self, M=1.0, a=0.99, r_cam=50.0, R_sky=1000.0, alpha=0.0, beta=0.0):
        self.M = M
        self.a = a
        self.r_cam = r_cam
        self.R_sky = R_sky
        self.alpha = alpha
        self.beta = beta
        
        # Bardeen Constants
        self.L_z = -self.alpha
        self.eta = self.beta**2
        self.K = self.eta + (self.L_z - self.a)**2
        self.r_horizon = self.M + np.sqrt(self.M**2 - self.a**2)

    def get_initial_conditions(self):
        R_K = ((self.r_cam**2 + self.a**2) - self.a * self.L_z)**2 - \
              (self.r_cam**2 - 2.0*self.M*self.r_cam + self.a**2) * self.K
        p_r_0 = -np.sqrt(max(R_K, 0.0))
        return [self.r_cam, np.pi/2.0, 0.0, p_r_0, self.beta]

    def equations_of_motion(self, lam, state):
        r, theta, phi, p_r, p_theta = state
        
        sin_theta = max(np.abs(np.sin(theta)), 1e-8)
        cos_theta = np.cos(theta)
        sin2 = sin_theta**2
        Delta = r**2 - 2.0*self.M*r + self.a**2

        dr_dlam = p_r
        dtheta_dlam = p_theta
        dphi_dlam = (self.a / Delta) * (r**2 + self.a**2 - self.a*self.L_z) - (self.a - self.L_z / sin2)
        dp_r_dlam = 2.0*r*(r**2 + self.a**2 - self.a*self.L_z) - (r - self.M)*self.K
        dp_theta_dlam = cos_theta * ((self.L_z**2)/(sin_theta**3) - self.a**2 * sin_theta)
        
        return [dr_dlam, dtheta_dlam, dphi_dlam, dp_r_dlam, dp_theta_dlam]

    def integrate(self):
        def event_horizon(lam, state):
            return state[0] - self.r_horizon * 1.01
        event_horizon.terminal = True
        
        def event_sky(lam, state):
            return state[0] - self.R_sky
        event_sky.terminal = True
        
        sol = solve_ivp(
            self.equations_of_motion,
            (0, 1e5), # Relaxed Mino limit for extreme bending
            self.get_initial_conditions(),
            events=[event_horizon, event_sky],
            method='LSODA',
            first_step=1e-5,
            rtol=1e-5, atol=1e-5 # Balanced for speed and visual accuracy
        )
        return sol

# =============================================================================
# Pixel Mapping Function (Runs on every CPU Core)
# =============================================================================
def trace_and_color_pixel(args):
    x_idx, y_idx, alpha, beta, M, a, r_cam, R_sky, bg_pixels = args
    
    # 1. Trace the Ray
    photon = Kerr_Raytracer(M=M, a=a, r_cam=r_cam, R_sky=R_sky, alpha=alpha, beta=beta)
    sol = photon.integrate()
    
    # 2. If it hit the horizon, return Black
    if sol.t_events[0].size > 0 or sol.t_events[1].size == 0:
        return (x_idx, y_idx, np.array([0, 0, 0], dtype=np.uint8))
        
    # 3. If it escaped, get final angles
    final_theta = sol.y[1, -1]
    final_phi = sol.y[2, -1]
    
    # 4. Direct UV Mapping to Background Image (No Interpolation!)
    bg_height, bg_width, _ = bg_pixels.shape
    
    U = (final_phi + np.pi) / (2.0 * np.pi)
    V = final_theta / np.pi
    
    U = U % 1.0
    V = np.clip(V, 0.0, 1.0)
    
    bg_x = int(U * (bg_width - 1))
    bg_y = int(V * (bg_height - 1))
    
    # Return the exact pixel coordinate and its new color
    return (x_idx, y_idx, bg_pixels[bg_y, bg_x])


if __name__ == "__main__":
    # --- Configuration ---
    M = 80.0
    a_star = 0.99
    a = a_star * M        
    r_cam = 900.0   
    R_sky = 1000.0
    
    # --- Resolution ---
    # 80x270 = 129,000 rays. Render time significant for higher resolution
    RES_X = 1280
    RES_Y = 720
    
    # --- Camera ---
    FOV_DEG = 90.0
    aspect_ratio = RES_X / RES_Y
    max_angle_rad = np.radians(FOV_DEG / 2.0)
    FOV_MAX = r_cam * np.sin(max_angle_rad) 
    
    # --- Load Background Image ---
    print("Loading background image...")
    original_image = "Inter.png"
    try:
        bg_img = Image.open(original_image).convert('RGB')
        bg_pixels = np.array(bg_img)
    except Exception as e:
        print(f"Error loading {original_image}.")
        sys.exit()

    # --- Setup Screen Grid ---
    print(f"Setting up {RES_X}x{RES_Y} camera grid...")
    angles_x = np.linspace(-max_angle_rad * aspect_ratio, max_angle_rad * aspect_ratio, RES_X)
    angles_y = np.linspace(max_angle_rad, -max_angle_rad, RES_Y)

    alpha = r_cam * np.sin(angles_x) / np.sqrt(1.0 - 2.0*M/r_cam)
    beta  = r_cam * np.sin(angles_y) / np.sqrt(1.0 - 2.0*M/r_cam)
    
    # Package jobs for the CPU
    ray_jobs = []
    for y_idx in range(RES_Y):
        for x_idx in range(RES_X):
            ray_jobs.append((x_idx, y_idx, alpha[x_idx], beta[y_idx], M, a, r_cam, R_sky, bg_pixels))
            
    total_pixels = len(ray_jobs)
    
    # --- Create Final Image Array ---
    final_image_data = np.zeros((RES_Y, RES_X, 3), dtype=np.uint8)
    
    # --- Raytrace! ---
    print(f"Igniting engines. Tracing {total_pixels} rays across all CPU cores...")
    start_time = time.time()
    
    # Use ProcessPool for heavy CPU math
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for i, result in enumerate(executor.map(trace_and_color_pixel, ray_jobs)):
            x_idx, y_idx, color = result
            final_image_data[y_idx, x_idx] = color
            
            if (i + 1) % 5000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                rem_time = (total_pixels - (i + 1)) / rate
                print(f"Rendered {i + 1}/{total_pixels} pixels... (Est. remaining: {rem_time/60:.1f} min)")

    # --- Save Output ---
    print("Saving realistic render...")
    final_img = Image.fromarray(final_image_data)
    final_img.save("Kerr_Realistic_test.png", quality=100)
    
    total_time = time.time() - start_time
    print(f"Done! Image saved as 'Kerr_Realistic_test.png' in {total_time / 60:.1f} minutes.")