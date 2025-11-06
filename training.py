import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time

import config
import dummy_data
from model import pinn_model, source_term_Q

print(f"using device: {config.DEVICE}")

# === Normalization constants ===
T_scale = config.GLOBAL_MAX_TEMP - config.GLOBAL_MIN_TEMP
L_scale = config.X
alpha = config.ALPHA
rho_c = config.RHO * config.C  # volumetric heat capacity [J/mm^3/C]
q_prime = config.Q_PRIME  # heat source intensity [W/mm]
tau_scale = L_scale**2 / alpha  # time scale


def normalize_inputs(inputs):
    """Normalize inputs [t, x, y, z, z_s] nondimensionally."""
    t = inputs[:, 0:1] / tau_scale
    x = inputs[:, 1:2] / L_scale
    y = inputs[:, 2:3] / L_scale
    z = inputs[:, 3:4] / L_scale
    z_s = inputs[:, 4:5] / L_scale
    return torch.cat([t, x, y, z, z_s], dim=1)


# === Data preparation ===
train_inputs = torch.tensor(
    np.vstack(dummy_data.all_train_inputs), dtype=torch.float32
).to(config.DEVICE)
train_outputs = torch.tensor(
    np.vstack(dummy_data.all_train_outputs), dtype=torch.float32
).to(config.DEVICE)

print(f"Training inputs shape: {train_inputs.shape}")
print(f"Training outputs shape: {train_outputs.shape}")

optimizer = torch.optim.Adam(pinn_model.parameters(), lr=config.LR)
loss_fn = nn.MSELoss()
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
# === Training loop ===
loss_data_hist, loss_phys_hist, loss_total_hist = [], [], []

start_time = time.time()
pinn_model.train()

for epoch in range(config.N_EPOCHS):
    optimizer.zero_grad()

    # --- Masks ---
    is_ic_point = (train_inputs[:, 0] < 1e-9) & (
        train_inputs[:, 3] < 1e-9
    )  # t ~ 0 and z ~ 0
    # Boundary condition (bottom plate, z=Z)
    is_bc_point = train_inputs[:, 3] >= config.Z
    # Surface data points (z=0, t > 0)
    is_surface_point = (train_inputs[:, 3] < 1e-9) & (~is_ic_point)
    # PDE (collocation) points are everything else
    is_pde_point = ~is_ic_point & ~is_bc_point & ~is_surface_point

    # === Normalize all inputs ===
    norm_inputs = normalize_inputs(train_inputs)
    norm_outputs = (train_outputs - config.GLOBAL_MIN_TEMP) / T_scale
    
    # --- Data loss (superficial points, t > 0) ---
    data_inputs = norm_inputs[is_surface_point]
    data_outputs = norm_outputs[is_surface_point]

    T_pred_data = pinn_model(data_inputs)
    T_pred_data_clamped = torch.clamp(T_pred_data, 0.0, 1.0)  # Clamp to [0, 1] range
    loss_data = loss_fn(T_pred_data_clamped, data_outputs)

    # --- Initial condition loss (t = 0) ---
    ic_inputs = norm_inputs[is_ic_point]
    ic_outputs = norm_outputs[is_ic_point]
    T_pred_ic = pinn_model(ic_inputs)
    loss_ic = loss_fn(T_pred_ic, ic_outputs)

    # --- boundary condition loss (z = Z) ---
    bc_inputs = norm_inputs[is_bc_point]
    bc_outputs = norm_outputs[is_bc_point]
    T_pred_bc = pinn_model(bc_inputs)
    loss_bc = loss_fn(T_pred_bc, bc_outputs)

    # --- Physics loss (PDE residual) ---
    inputs_phys = norm_inputs.clone()
    t_norm, x_norm, y_norm, z_norm, z_s_norm = inputs_phys.split(1, dim=1)

    # Enable gradients
    t_norm.requires_grad_()
    x_norm.requires_grad_()
    y_norm.requires_grad_()
    z_norm.requires_grad_()

    T_pred_phys = pinn_model(
        torch.cat([t_norm, x_norm, y_norm, z_norm, z_s_norm], dim=1)
    )
    ones = torch.ones_like(T_pred_phys, device=config.DEVICE)

    grads = torch.autograd.grad(
        T_pred_phys,
        [t_norm, x_norm, y_norm, z_norm],
        grad_outputs=ones,
        create_graph=True,
    )
    dT_dt_norm, dT_dx_norm, dT_dy_norm, dT_dz_norm = grads

    dT_dxx_norm = torch.autograd.grad(
        dT_dx_norm, x_norm, grad_outputs=ones, create_graph=True
    )[0]
    dT_dyy_norm = torch.autograd.grad(
        dT_dy_norm, y_norm, grad_outputs=ones, create_graph=True
    )[0]
    dT_dzz_norm = torch.autograd.grad(
        dT_dz_norm, z_norm, grad_outputs=ones, create_graph=True
    )[0]
    laplacian_norm = dT_dxx_norm + dT_dyy_norm + dT_dzz_norm

    # Reconstruct physical coordinates for Q(x,z,z_s)
    x_phys = x_norm * L_scale
    z_phys = z_norm * L_scale
    z_s_phys = z_s_norm * L_scale

    Q_phys = source_term_Q(x_phys, z_phys, z_s_phys)

    Q_norm = Q_phys * (tau_scale / T_scale)

    residual = dT_dt_norm - laplacian_norm - Q_norm
    residual_pde_only = residual[is_pde_point]
    loss_phys = loss_fn(
        residual_pde_only, torch.zeros_like(residual_pde_only, device=config.DEVICE)
    )

    if epoch % 20 == 0:   
        with torch.no_grad():
            print("=== TERM MAGNITUDES ===")
            print("dT_dt_norm: ", dT_dt_norm.min().item(), dT_dt_norm.max().item(), dT_dt_norm.abs().mean().item())
            print("laplacian_norm: ", laplacian_norm.min().item(), laplacian_norm.max().item(), laplacian_norm.abs().mean().item())
            print("Q_norm: ", Q_norm.min().item(), Q_norm.max().item(), Q_norm.abs().mean().item())
            print("residual: ", residual.min().item(), residual.max().item(), residual.abs().mean().item())



    # --- Total loss ---
    total_loss = (
        loss_data
        + config.LAMBDA_PHYSICS * loss_phys
        + config.LAMBDA_CONDITIONS * (loss_ic + loss_bc)
    )

    total_loss.backward()
    optimizer.step()
    scheduler.step()

    # === Track losses ===
    loss_data_hist.append(loss_data.item())
    loss_phys_hist.append(loss_phys.item())
    loss_total_hist.append(total_loss.item())

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:04d}: "
            f"Total={total_loss.item():.3e}, "
            f"Data={loss_data.item():.3e}, "
            f"Phys={loss_phys.item():.3e}"
        )

train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} s.")

# === Plot training losses ===
plt.figure(figsize=(8, 5))
plt.plot(loss_total_hist, label="Total Loss", color="black", linewidth=1.5)
plt.plot(loss_data_hist, label="Data Loss", color="blue", linestyle="--")
plt.plot(loss_phys_hist, label="Physics Loss", color="red", linestyle=":")
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PINN Training Losses")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

# === Inference: Surface temperature map (z=0) ===
pinn_model.eval()
print("\nRunning surface inference (T(x, y, z=0))...")

z_s_plot = config.Z_S_TRAIN[0]
t_plot = config.T / 2
z_surface = 0.0

nx, ny = 200, 200
x_axis = np.linspace(0, config.X, nx)
y_axis = np.linspace(0, config.Y, ny)
X_grid, Y_grid = np.meshgrid(x_axis, y_axis)

# Build inference tensor (physical)
t_grid = np.full_like(X_grid, t_plot)
z_grid = np.full_like(X_grid, z_surface)
z_s_grid = np.full_like(X_grid, z_s_plot)

# Normalize for network input
inputs_infer_norm = np.stack(
    [
        t_grid / tau_scale,
        X_grid / L_scale,
        Y_grid / L_scale,
        z_grid / L_scale,
        z_s_grid / L_scale,
    ],
    axis=-1,
)

inputs_infer_torch = torch.tensor(
    inputs_infer_norm.reshape(-1, 5), dtype=torch.float32
).to(config.DEVICE)

# Predict temperature
with torch.no_grad():
    T_pred_absolute = (
        (pinn_model(inputs_infer_torch).cpu().numpy().reshape(ny, nx) * T_scale) + config.GLOBAL_MIN_TEMP
    )

print(
    f"T_pred_absolute range: {T_pred_absolute.min():.3e} to {T_pred_absolute.max():.3e}"
)
print(
    f"vmin={config.GLOBAL_MIN_TEMP}, vmax={np.max([config.GLOBAL_MAX_TEMP, T_pred_absolute.max()])}"
)

# === Plot surface temperature ===
plt.figure(figsize=(7, 5))
im = plt.imshow(
    T_pred_absolute,
    extent=[0, config.X, 0, config.Y],
    origin="lower",
    cmap="hot",
    # vmin=config.GLOBAL_MIN_TEMP,
    # vmax=np.max([config.GLOBAL_MAX_TEMP, T_pred_absolute.max()]), # Auto-scale max
    aspect="auto",
)
plt.colorbar(im, label="Temperature (°C)")
plt.title(f"Predicted Surface Temperature (z=0) at t={t_plot:.2f}s, z_s={z_s_plot} mm")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")
plt.axvline(
    config.X_SOURCE_POS, color="cyan", linestyle="--", lw=1.2, label="Wire position"
)
plt.legend()
plt.tight_layout()
plt.savefig("predicted_temperature.png", dpi=300)
plt.show()
