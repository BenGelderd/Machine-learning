import numpy as np
import pandas as pd
from PIL import Image
from scipy.interpolate import interp1d

# =============================================================================
# 1. Configuration & Data Loading
# =============================================================================
# Load the 2D raytracing data
data = pd.read_csv("2d_data.csv")

# FIX: Drop NaNs and keep only positive angles to prevent "NaN poisoning"
valid_data = data.dropna()
valid_data = valid_data[valid_data['beta'] >= 0]

# Convert degrees to radians
beta_data = valid_data['beta'].values
phi_data = valid_data['phi'].values

# Create an interpolation function.
# - If a pixel's beta is LESS than our minimum valid beta, it falls in the shadow (returns NaN -> black pixel).
# - If a pixel's beta is GREATER than our maximum data, it returns NaN (black pixel).
get_phi = interp1d(beta_data, phi_data, kind='linear', bounds_error=False, fill_value=np.nan)

# Load the background image
bg_img = Image.open("Inter.png")
bg_pixels = np.array(bg_img)
bg_height, bg_width, _ = bg_pixels.shape

# Setup final render resolution
RENDER_WIDTH = 1920
RENDER_HEIGHT = 1080
FOV_DEG = 90.0

# =============================================================================
# 2. Camera & Screen Setup (The 90-Degree FOV)
# =============================================================================
print("Setting up camera grid...")
# For a 90 degree horizontal FOV, the focal length is exactly half the screen width
focal_length = RENDER_WIDTH / (2.0 * np.tan(np.radians(FOV_DEG / 2.0)))

# Create a 2D grid of pixel coordinates
# Screen center is (0,0). Y is flipped so positive is UP.
x_coords = np.linspace(-RENDER_WIDTH/2, RENDER_WIDTH/2, RENDER_WIDTH)
y_coords = np.linspace(RENDER_HEIGHT/2, -RENDER_HEIGHT/2, RENDER_HEIGHT)
X, Y = np.meshgrid(x_coords, y_coords)

# =============================================================================
# 3. 3D Vector Math & Bending (Vectorized for all pixels at once)
# =============================================================================
print("Calculating ray deflections...")

# Define the Optical Axis (u) pointing straight from camera to black hole
# We will say the black hole is down the +X axis.
# Right is +Y, Up is +Z
u = np.array([1.0, 0.0, 0.0])

# Calculate initial pixel vector (v) for every pixel
# Shape is (1080, 1920, 3)
V = np.stack((np.full_like(X, focal_length), X, Y), axis=-1)

# Normalize V to get direction vectors
V_norm = np.linalg.norm(V, axis=-1, keepdims=True)
V_dir = V / V_norm

# Calculate the initial angle Beta for every pixel using the dot product
# Since u is [1,0,0], the dot product is just the X component of V_dir
cos_beta = V_dir[:, :, 0]
beta_grid = np.arccos(cos_beta)

# Lookup the final swept angle Phi for every pixel
phi_grid = get_phi(beta_grid)

# Find the perpendicular normal vector (n = u x v) for the rotational plane
N = np.cross(u, V_dir)
N_norm = np.linalg.norm(N, axis=-1, keepdims=True)
# Avoid division by zero for the dead-center pixel
N_norm[N_norm == 0] = 1.0 
N_dir = N / N_norm

# Find the tangent vector (w = n x u)
W = np.cross(N_dir, u)

# Calculate the final deflected 3D ray vector!
phi_grid_mapped = (phi_grid - np.pi) 

phi_grid_expanded = phi_grid_mapped[..., np.newaxis]
r_final = np.cos(phi_grid_expanded) * u + np.sin(phi_grid_expanded) * W

# =============================================================================
# 4. Equirectangular UV Mapping
# =============================================================================
print("Mapping pixels to background sphere...")

# Extract the final x, y, z components
r_x = r_final[:, :, 0]
r_y = r_final[:, :, 1]
r_z = r_final[:, :, 2]

# Calculate spherical coordinates (Longitude and Latitude)
# arctan2 perfectly handles the full 360 degree wrap around the camera
longitude = np.arctan2(r_y, r_x) 
latitude = np.arcsin(np.clip(r_z, -1.0, 1.0)) # Clip prevents tiny floating point errors

# Normalize to UV coordinates (0.0 to 1.0)
U = (longitude + np.pi) / (2.0 * np.pi)
V_uv = (latitude + (np.pi / 2.0)) / np.pi

# Convert UV back to background image pixel coordinates
bg_x = (U * (bg_width - 1)).astype(int)
# V=0 is bottom, V=1 is top. Image arrays usually have Y=0 at the top, so we invert it.
bg_y = ((1.0 - V_uv) * (bg_height - 1)).astype(int)

# =============================================================================
# 5. Image Generation
# =============================================================================
print("Generating final image...")

# Create an empty black image array
final_image_data = np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.uint8)

# Create a mask to find only the rays that escaped (Phi is not NaN)
escaped_mask = ~np.isnan(phi_grid)

# Map the colors from the background image to the final image, but only where rays escaped
final_image_data[escaped_mask] = bg_pixels[bg_y[escaped_mask], bg_x[escaped_mask]]

# Save the final image
final_img = Image.fromarray(final_image_data)
file_name = "lensed_image.jpg"
final_img.save(file_name, quality=95)

print(f"Render complete! Saved as {file_name}")