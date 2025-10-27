import numpy as np
import config

def get_analytical_solution(t, x, z, z_s, X_SOURCE_POS, Q_PRIME, ALPHA, sigma):
    """
    Soluzione analitica dell'equazione del calore con sorgente gaussiana:
    
    ∂T/∂t - alpha∇²T = Q(x,z;x_s,z_s)
    
    con:
        Q(x,z;x_s,z_s) = (q' / (2π sigma²)) * exp(-((x-x_s)² + (z-z_s)²) / (2 sigma²))
    
    Parametri:
        t: tempo [s]
        x, z: coordinate del punto di valutazione [m]
        z_s: quota della sorgente [m]
        X_SOURCE_POS: coordinata x della sorgente [m]
        Q_PRIME: intensità di sorgente [W/m]
        ALPHA: diffusività termica [m²/s]
        sigma: deviazione standard spaziale della sorgente [m]
    """
    
    t = np.maximum(t, 1e-9)  # evita t = 0
    
    # distanza al quadrato dal centro della sorgente
    r_sq = (x - X_SOURCE_POS)**2 + (z - z_s)**2
    
    # varianza effettiva: diffusione + estensione iniziale della sorgente
    sigma_eff_sq = sigma**2 + 2 * ALPHA * t
    
    # soluzione analitica
    delta_T = (Q_PRIME / (2 * np.pi * sigma_eff_sq * 1000 * 4186)) * \
              np.exp(-r_sq / (2 * sigma_eff_sq))
    
    return delta_T


print("Fase 1: Generazione Dati di Addestramento...")
# Dati di addestramento da sorgenti a profondità note
z_sources_train = config.Z_S_TRAIN  # profondità delle sorgenti [mm]
N_DATA_PER_SOURCE = config.X * config.Y *config.T  # punti di dati per sorgente
all_train_inputs = []
all_train_outputs = []

for z_s in z_sources_train:
    # Dati di superficie (per L_data)
    t_data = np.random.rand(N_DATA_PER_SOURCE, 1) * config.T
    x_data = np.random.rand(N_DATA_PER_SOURCE, 1) * config.X
    z_data = np.zeros_like(x_data)
    z_s_data = np.full_like(x_data, z_s)
    T_data = get_analytical_solution(t_data, x_data, z_data, z_s,config.X_SOURCE_POS, config.Q_PRIME, config.ALPHA, sigma=config.SIGMA)  # temperatura [°C]
    
    # Punti di collocazione (per L_phys)
    t_phys = np.random.rand(config.N_COLLOCATION, 1) * config.T
    x_phys = np.random.rand(config.N_COLLOCATION, 1) * config.X
    z_phys = np.random.rand(config.N_COLLOCATION, 1) * config.Z
    z_s_phys = np.full_like(x_phys, z_s)
    
    inputs_combined = np.vstack([
        np.hstack([t_data, x_data, z_data, z_s_data]),
        np.hstack([t_phys, x_phys, z_phys, z_s_phys])
    ])
    outputs_combined = np.vstack([T_data, np.zeros((config.N_COLLOCATION, 1))])

    all_train_inputs.append(inputs_combined)
    all_train_outputs.append(outputs_combined)
