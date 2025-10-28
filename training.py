import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

import config
import dummy_data
from model import pinn_model, source_term_Q

print(f"using device: {config.DEVICE}")

# === Data preparation ===
train_inputs = torch.tensor(np.vstack(dummy_data.all_train_inputs), dtype=torch.float32).to(config.DEVICE)
train_outputs = torch.tensor(np.vstack(dummy_data.all_train_outputs), dtype=torch.float32).to(config.DEVICE)

print(f"Training inputs shape: {train_inputs.shape}")
print(f"Training outputs shape: {train_outputs.shape}")

optimizer = torch.optim.Adam(pinn_model.parameters(), lr=config.LR)
loss_fn = nn.MSELoss()

# === Training loop ===
loss_data_hist, loss_phys_hist, loss_total_hist = [], [], []

start_time = time.time()
pinn_model.train()

for epoch in range(config.N_EPOCHS):
    optimizer.zero_grad()

    # --- Masks ---
    is_surface_point = train_inputs[:, 3] < 1e-9   # z ~ 0
    is_ic_point = train_inputs[:, 0] < 1e-9        # t ~ 0

    # --- Data loss (superficial points) ---
    data_inputs = train_inputs[is_surface_point]
    data_outputs = train_outputs[is_surface_point]
    T_pred_data = pinn_model(data_inputs)
    loss_data = loss_fn(T_pred_data, data_outputs)

    # --- Physics loss (PDE residual) ---
    inputs_phys = train_inputs.clone()
    t, x, y, z, z_s = inputs_phys.split(1, dim=1)
    t.requires_grad_(); x.requires_grad_(); y.requires_grad_(); z.requires_grad_()

    T_pred_phys = pinn_model(torch.cat([t, x, y, z, z_s], dim=1))
    ones = torch.ones_like(T_pred_phys, device=config.DEVICE)
    grads = torch.autograd.grad(T_pred_phys, [t, x, y, z], grad_outputs=ones, create_graph=True)
    dT_dt, dT_dx, dT_dy, dT_dz = grads
    dT_dxx = torch.autograd.grad(dT_dx, x, grad_outputs=ones, create_graph=True)[0]
    dT_dyy = torch.autograd.grad(dT_dy, y, grad_outputs=ones, create_graph=True)[0]
    dT_dzz = torch.autograd.grad(dT_dz, z, grad_outputs=ones, create_graph=True)[0]
    laplacian = dT_dxx + dT_dyy + dT_dzz
    Q = source_term_Q(x, z, z_s)

    residual = dT_dt - config.ALPHA * laplacian - Q
    loss_pde = loss_fn(residual, torch.zeros_like(residual))

    # --- Initial condition loss (t = 0) ---
    ic_inputs = train_inputs[is_ic_point]
    ic_outputs = train_outputs[is_ic_point]
    T_pred_ic = pinn_model(ic_inputs)
    loss_ic = loss_fn(T_pred_ic, ic_outputs)

    # --- Combine into single physics loss ---
    loss_phys = loss_pde + loss_ic

    # --- Total loss ---
    total_loss = loss_data + config.LAMBDA_PHYSICS * loss_phys

    total_loss.backward()
    optimizer.step()

    loss_data_hist.append(loss_data.item())
    loss_phys_hist.append(loss_phys.item())
    loss_total_hist.append(total_loss.item())
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:04d}: "
              f"Total={total_loss.item():.3e}, "
              f"Data={loss_data.item():.3e}, "
              f"Phys={loss_phys.item():.3e}")

train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} s.")

# === Plot training losses ===
plt.figure(figsize=(8,5))
plt.plot(loss_total_hist, label='Total Loss', color='black', linewidth=1.5)
plt.plot(loss_data_hist, label='Data Loss', color='blue', linestyle='--')
plt.plot(loss_phys_hist, label='Physics Loss', color='red', linestyle=':')
plt.xlabel("Epoch")
plt.ylabel("Loss (log scale)")
plt.title("PINN Training Losses")
plt.legend()
plt.grid(True, which="both", ls=":")
plt.tight_layout()
plt.show()

# === Inference section: Surface temperature map (z = 0) ===
pinn_model.eval()
print("\n Running surface inference (T(x, y, z=0))...")

# Choose one source depth and time for visualization
z_s_plot = config.Z_S_TRAIN[0]    # e.g. depth of the heat source
t_plot = config.T / 2             # snapshot in time
z_surface = 0.0                   # surface plane

# Define spatial grid in (x, y)
nx, ny = 200, 200
x_axis = np.linspace(0, config.X, nx)
y_axis = np.linspace(0, config.Y, ny)
X_grid, Y_grid = np.meshgrid(x_axis, y_axis)

# Build inference tensor
t_grid = np.full_like(X_grid, t_plot)
z_grid = np.full_like(X_grid, z_surface)
z_s_grid = np.full_like(X_grid, z_s_plot)

inputs_infer = np.stack([t_grid, X_grid, Y_grid, z_grid, z_s_grid], axis=-1)
inputs_infer_torch = torch.tensor(inputs_infer.reshape(-1, 5), dtype=torch.float32).to(config.DEVICE)

# Predict temperature
with torch.no_grad():
    T_pred = pinn_model(inputs_infer_torch).cpu().numpy().reshape(ny, nx)

# === Plot surface temperature field ===
plt.figure(figsize=(7,5))
im = plt.imshow(
    T_pred, extent=[0, config.X, 0, config.Y],
    origin='lower', cmap='hot', 
    vmin=config.GLOBAL_MIN_TEMP, vmax=config.GLOBAL_MAX_TEMP,
    aspect='auto'
)
plt.savefig("predicted_temperature.png", dpi=300)
plt.colorbar(im, label='Temperature (°C)')
plt.title(f"Predicted Surface Temperature (z=0) at t={t_plot:.2f}s, z_s={z_s_plot} mm")
plt.xlabel("x (mm)")
plt.ylabel("y (mm)")

# Draw wire location
plt.axvline(config.X_SOURCE_POS, color='cyan', linestyle='--', lw=1.2, label='Wire position')
plt.legend()
plt.tight_layout()
plt.show()