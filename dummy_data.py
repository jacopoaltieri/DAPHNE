import numpy as np
import config
import matplotlib.pyplot as plt
from scipy.special import exp1

def get_analytical_solution(t, x, z, z_s, X_SOURCE_POS, Q_PRIME, ALPHA, KAPPA, sigma):
    """
    Analytical solution of the 2D heat equation for a Gaussian line source (wire along y-axis):

        ∂T/∂t -  alpha ∇²T = (Q' / (2π  sigma ²)) * exp(-((x - x_s)² + (z - z_s)²) / (2 sigma ²))

    Solution:
        ΔT(x,z,t) = (Q' / (4π κ)) * E₁(r² / (4 alpha t + 2 sigma ²))

    Parameters:
        t : time [s]
        x, z : coordinates of evaluation point [mm]
        z_s : source depth [mm]
        X_SOURCE_POS : source x position [mm]
        Q_PRIME : line power per unit length [W/mm]
        ALPHA : thermal diffusivity [mm²/s]
        KAPPA : thermal conductivity [W/(mm·K)]
        sigma : standard deviation of Gaussian source [mm]
    """
    t = np.maximum(t, 1e-9)  # avoid t=0 singularity
    r_sq = (x - X_SOURCE_POS)**2 + (z - z_s)**2
    denom = 4 * ALPHA * t + 2 * sigma**2
    arg = r_sq / denom
    delta_T = (Q_PRIME / (4 * np.pi * KAPPA)) * exp1(arg)
    return  config.GLOBAL_MIN_TEMP + delta_T


print("Phase 1: Generating training data...")
z_sources_train = config.Z_S_TRAIN
all_train_inputs = []
all_train_outputs = []

# --- Axes for gridded data ---
t_axis = np.linspace(0, config.T, config.T)
x_axis = np.linspace(0, config.X, config.X)
y_axis = np.linspace(0, config.Y, config.Y)

# === Shared Initial Condition points (t = 0, T = minimum temp) ===
N_ic = config.N_INITIAL  
t_ic = np.zeros((N_ic, 1))
x_ic = np.random.rand(N_ic, 1) * config.X
y_ic = np.random.rand(N_ic, 1) * config.Y
z_ic = np.random.rand(N_ic, 1) * config.Z
T_ic_flat = np.full((N_ic, 1), config.GLOBAL_MIN_TEMP)  # initial temp

for z_s in z_sources_train:
    print(f"Generating data for source depth z_s={z_s} mm...")    

    # --- Surface data generation (z=0) ---
    t_grid, x_grid, y_grid = np.meshgrid(t_axis, x_axis, y_axis, indexing='ij')
    z_grid = np.zeros_like(t_grid)
    z_s_grid = np.full_like(t_grid, z_s)
    
    T_data = get_analytical_solution(
        t_grid, x_grid, z_grid, z_s,
        config.X_SOURCE_POS, config.Q_PRIME, config.ALPHA, config.KAPPA, sigma=config.SIGMA
    )
    
    # Flatten surface data
    t_data_flat = t_grid.flatten()[:, np.newaxis]
    x_data_flat = x_grid.flatten()[:, np.newaxis]
    y_data_flat = y_grid.flatten()[:, np.newaxis] 
    z_data_flat = z_grid.flatten()[:, np.newaxis]
    z_s_data_flat = z_s_grid.flatten()[:, np.newaxis]
    T_data_flat = T_data.flatten()[:, np.newaxis]
    
    data_inputs_combined = np.hstack([t_data_flat, x_data_flat, y_data_flat, z_data_flat, z_s_data_flat])

    # --- Collocation points (for PDE residual) ---
    t_phys = np.random.rand(config.N_COLLOCATION, 1) * config.T
    x_phys = np.random.rand(config.N_COLLOCATION, 1) * config.X
    y_phys = np.random.rand(config.N_COLLOCATION, 1) * config.Y 
    z_phys = np.random.rand(config.N_COLLOCATION, 1) * config.Z
    z_s_phys = np.full_like(x_phys, z_s)
    
    phys_inputs_combined = np.hstack([t_phys, x_phys, y_phys, z_phys, z_s_phys])

    # --- Attach shared IC points (duplicate z_s for this source) ---
    z_s_ic = np.full_like(x_ic, z_s)
    ic_inputs_combined = np.hstack([t_ic, x_ic, y_ic, z_ic, z_s_ic])

    # --- Combine all input/output sets ---
    inputs_combined = np.vstack([
        data_inputs_combined,
        phys_inputs_combined,
        ic_inputs_combined
    ])
    outputs_combined = np.vstack([
        T_data_flat,
        np.zeros((config.N_COLLOCATION, 1)),  # dummy targets for PDE points
        T_ic_flat                              # zero-temperature IC targets
    ])

    all_train_inputs.append(inputs_combined)
    all_train_outputs.append(outputs_combined)

print("Data generation complete.")

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Pick the first source depth
    z_s = z_sources_train[0]

    # --- Use 2D grids for the 2D analytical solution ---
    t_axis_plot = np.linspace(0, config.T, config.T) # Renamed to avoid confusion
    x_plot_axis = np.linspace(0, config.X, config.X)
    z_plot_axis = np.linspace(0, config.Z, config.Z)
    y_plot_axis = np.linspace(0, config.Y, config.Y) # Need this for tiling

    t_idxs = [0, config.T // 2, config.T - 1]

    for t_idx in t_idxs:
        t_val = t_axis_plot[t_idx]

        # Create (z, x) grids for T(z, x)
        Z_plot_grid, X_plot_grid = np.meshgrid(z_plot_axis, x_plot_axis, indexing='ij')

        # --- Calculate the 2D temperature field T(z, x) ---
        # Pass the scalar z_s (e.g., 10) as the 'z_s' argument
        T_field_2D = get_analytical_solution(
            t_val,
            X_plot_grid,         # 'x' argument
            Z_plot_grid,         # 'z' argument
            z_s,                 # 'z_s' argument (the scalar)
            config.X_SOURCE_POS,
            config.Q_PRIME, config.ALPHA, config.KAPPA,
            sigma=config.SIGMA
        )
        # T_field_2D now has shape (len(z), len(x))

        # --- Plot three views ---
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Temperature at t={t_val:.2f} s (source z_s={z_s} mm)")

        # Top view (x-y plane at z=top, z=0)
        # Solution is y-independent, so tile the z=0 slice
        T_surface_slice = T_field_2D[0, :]  # T(z=0, x)
        T_top_view = np.tile(T_surface_slice, (len(y_plot_axis), 1))
        im0 = axes[0].imshow(
            T_top_view, origin='lower', extent=[0, config.X, 0, config.Y], aspect='auto'
        )
        axes[0].set_title("Top view (x-y)")
        axes[0].set_xlabel("x [mm]")
        axes[0].set_ylabel("y [mm]")
        fig.colorbar(im0, ax=axes[0])

        # Side view (x-z plane)
        # This is just the 2D field we calculated
        im1 = axes[1].imshow(
            T_field_2D, origin='upper', extent=[0, config.X, 0, config.Z], aspect='auto'
        )
        axes[1].set_title("Side view (x-z)")
        axes[1].set_xlabel("x [mm]")
        axes[1].set_ylabel("z [mm]")
        fig.colorbar(im1, ax=axes[1])

        # Front view (y-z plane at x=center)
        # Solution is y-independent, so tile the x=center slice
        x_center_idx = config.X // 2
        T_front_slice = T_field_2D[:, x_center_idx] # T(z, x=center)
        # Tile T(z) along the y-axis
        T_front_view = np.tile(T_front_slice[:, np.newaxis], (1, len(y_plot_axis)))
        im2 = axes[2].imshow(
            T_front_view, origin='upper', extent=[0, config.Y, 0, config.Z], aspect='auto'
        )
        axes[2].set_title("Front view (y-z)")
        axes[2].set_xlabel("y [mm]")
        axes[2].set_ylabel("z [mm]")
        fig.colorbar(im2, ax=axes[2])

        plt.tight_layout()
        plt.show()