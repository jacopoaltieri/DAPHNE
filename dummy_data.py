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

    # radial distance squared in x-z plane
    r_sq = (x - X_SOURCE_POS)**2 + (z - z_s)**2

    # modified argument including Gaussian width
    denom = 4 * ALPHA * t + 2 * sigma**2
    arg = r_sq / denom

    # temperature rise
    delta_T = (Q_PRIME / (4 * np.pi * KAPPA)) * exp1(arg)

    # return absolute temperature
    return config.GLOBAL_MIN_TEMP + delta_T



print("Fase 1: Generazione Dati di Addestramento...")
# Dati di addestramento da sorgenti a profondità note
z_sources_train = config.Z_S_TRAIN
all_train_inputs = []
all_train_outputs = []

# --- Creazione degli assi per la griglia dati ---
t_axis = np.linspace(0, config.T, config.T)
x_axis = np.linspace(0, config.X, config.X)
y_axis = np.linspace(0, config.Y, config.Y)

for z_s in z_sources_train:
    print(f"  Generazione griglia dati 3D (t,x,y) per z_s = {z_s} mm...")
    
    # --- Dati di superficie (per L_data) su GRIGLIA COMPLETA ---
    t_grid, x_grid, y_grid = np.meshgrid(t_axis, x_axis, y_axis, indexing='ij')
    z_grid = np.zeros_like(t_grid) # Superficie z=0
    z_s_grid = np.full_like(t_grid, z_s)
    
    # Calcola la temperatura sulla griglia (la funzione ignora y_grid,
    # il che è corretto per un filo parallelo all'asse y)
    T_data = get_analytical_solution(
        t_grid, x_grid, z_grid, z_s,
        config.X_SOURCE_POS, config.Q_PRIME, config.ALPHA,config.KAPPA, sigma=config.SIGMA
    )
    
    # Appiattisci le griglie per l'input della rete
    t_data_flat = t_grid.flatten()[:, np.newaxis]
    x_data_flat = x_grid.flatten()[:, np.newaxis]
    y_data_flat = y_grid.flatten()[:, np.newaxis] # Aggiunta coordinata Y
    z_data_flat = z_grid.flatten()[:, np.newaxis]
    z_s_data_flat = z_s_grid.flatten()[:, np.newaxis]
    T_data_flat = T_data.flatten()[:, np.newaxis]
    
    # Combina gli input di superficie (ora 5 colonne)
    data_inputs_combined = np.hstack([t_data_flat, x_data_flat, y_data_flat, z_data_flat, z_s_data_flat])

    # --- Punti di collocazione (per L_phys) ---
    # Questi possono rimanere campionati casualmente (è più efficiente)
    # Ma ora devono includere la coordinata y
    t_phys = np.random.rand(config.N_COLLOCATION, 1) * config.T
    x_phys = np.random.rand(config.N_COLLOCATION, 1) * config.X
    y_phys = np.random.rand(config.N_COLLOCATION, 1) * config.Y # Aggiunta coordinata Y
    z_phys = np.random.rand(config.N_COLLOCATION, 1) * config.Z
    z_s_phys = np.full_like(x_phys, z_s)
    
    # Combina gli input di fisica (ora 5 colonne)
    phys_inputs_combined = np.hstack([t_phys, x_phys, y_phys, z_phys, z_s_phys])
    
    # --- Unisci Dati e Fisica ---
    inputs_combined = np.vstack([data_inputs_combined, phys_inputs_combined])
    outputs_combined = np.vstack([T_data_flat, np.zeros((config.N_COLLOCATION, 1))])

    all_train_inputs.append(inputs_combined)
    all_train_outputs.append(outputs_combined)

print("Generazione dati completata.")


# --- NUOVO: Plot della sequenza di frame DATG ---
if __name__ == "__main__":
    print("\nGenerazione plot di esempio della sequenza DATG...")

    n_frames_to_plot = 5
    plot_times = np.linspace(1e-9, config.T, n_frames_to_plot) # Iniziamo da t>0
    plot_z_s = config.Z_S_TRAIN[3] # Usiamo la prima sorgente per l'esempio
    
    # Griglia 2D (x,y) per il plot
    plot_nx, plot_ny = 300, 300
    x_plot_axis = np.linspace(0, config.X, plot_nx)
    y_plot_axis = np.linspace(0, config.Y, plot_ny)
    X_plot_grid, Y_plot_grid = np.meshgrid(x_plot_axis, y_plot_axis)
    Z_plot_grid = np.zeros_like(X_plot_grid) # z=0

    # Trova la T massima (che per questo modello è a t=0) per la normalizzazione
    T_initial_frame = get_analytical_solution(
        1e-9, X_plot_grid, Z_plot_grid, plot_z_s,
        config.X_SOURCE_POS, config.Q_PRIME, config.ALPHA,config.KAPPA, sigma=config.SIGMA
    )
    vmax = config.GLOBAL_MAX_TEMP
    vmin = config.GLOBAL_MIN_TEMP
    
    print(f"Plotting 5 frame con T_max={vmax:.4f}°C (all'inizio)")

    fig, axes = plt.subplots(1, n_frames_to_plot, figsize=(20, 5), sharey=True)
    
    for i, t in enumerate(plot_times):
        # Calcola il frame
        T_frame = get_analytical_solution(
            np.full_like(X_plot_grid, t), # t costante
            X_plot_grid, 
            Z_plot_grid, 
            plot_z_s,
            config.X_SOURCE_POS, config.Q_PRIME, config.ALPHA,config.KAPPA, sigma=config.SIGMA
        )
        
        # Plotta il frame
        ax = axes[i]
        im = ax.imshow(
            T_frame, 
            cmap='hot', 
            vmin=vmin, 
            vmax=vmax,
            extent=[0, config.X, 0, config.Y], # Assi x, y
            origin='lower'
        )
        ax.set_title(f"t = {t:.1f} s")
        ax.set_xlabel("x (mm)")
        ax.axvline(config.X_SOURCE_POS, color='cyan', linestyle='--', lw=1, label='Filo')
        if i == 0:
            ax.set_ylabel("y (mm)")
            ax.legend()

    fig.suptitle(f"Simulazione DATG: Sorgente a Impulso (z_s={plot_z_s} mm)", fontsize=16)
    fig.colorbar(im, ax=axes.ravel().tolist(), label="Aumento Temp. (°C)", shrink=0.75)
    # plt.tight_layout()
    plt.show()