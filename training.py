import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm  # === NEW: Import tqdm ===

import config
import dummy_data
from model import pinn_model, source_term_Q


print(f"using device: {config.DEVICE}")

# === Normalization constants ===
# Note: This assumes (T - T_min) / T_scale normalization
T_scale = config.GLOBAL_PLATE_TEMP - config.GLOBAL_MIN_TEMP
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
train_inputs = torch.tensor(np.vstack(dummy_data.all_train_inputs), dtype=torch.float32)
train_outputs = torch.tensor(
    np.vstack(dummy_data.all_train_outputs), dtype=torch.float32
)

print(f"Training inputs shape: {train_inputs.shape}")
print(f"Training outputs shape: {train_outputs.shape}")

# === NEW: Create DataLoader for batching ===
dataset = TensorDataset(train_inputs, train_outputs)
# Drop last batch if it's smaller, to avoid issues with batch-norm (if used)
# and to keep loss averaging simple.
train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)
print(f"Total data points: {len(dataset)}. Batches per epoch: {len(train_loader)}")


optimizer = torch.optim.Adam(pinn_model.parameters(), lr=config.LR)
loss_fn = nn.MSELoss()
scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
# === Training loop ===
loss_data_hist, loss_phys_hist, loss_total_hist = [], [], []

start_time = time.time()
pinn_model.train()

# === NEW: Wrap epoch loop with tqdm ===
epoch_loop = tqdm(range(config.N_EPOCHS), desc="Training Progress")
for epoch in epoch_loop:

    # === NEW: Averages for one epoch ===
    epoch_loss_data, epoch_loss_phys, epoch_loss_total = 0.0, 0.0, 0.0

    # === NEW: Batch loop ===
    for batch_inputs, batch_outputs in train_loader:

        # Move batch to device
        batch_inputs = batch_inputs.to(config.DEVICE)
        batch_outputs = batch_outputs.to(config.DEVICE)

        optimizer.zero_grad()

        # --- Masks (now operate on BATCH data) ---
        is_ic_point = (batch_inputs[:, 0] < 1e-9) & (
            batch_inputs[:, 3] < 1e-9
        )  # t ~ 0 and z ~ 0
        is_bc_point = batch_inputs[:, 3] >= config.Z
        is_surface_point = (batch_inputs[:, 3] < 1e-9) & (~is_ic_point)
        is_pde_point = ~is_ic_point & ~is_bc_point & ~is_surface_point

        # === Normalize all BATCH inputs ===
        norm_batch_inputs = normalize_inputs(batch_inputs)
        norm_batch_outputs = (batch_outputs - config.GLOBAL_MIN_TEMP) / T_scale

        # --- Data loss (superficial points, t > 0) ---
        data_inputs = norm_batch_inputs[is_surface_point]
        data_outputs = norm_batch_outputs[is_surface_point]

        loss_data = torch.tensor(0.0, device=config.DEVICE)
        if data_inputs.shape[0] > 0:
            T_pred_data = pinn_model(data_inputs)
            T_pred_data_clamped = torch.clamp(
                T_pred_data,config.GLOBAL_MIN_TEMP/T_scale, config.GLOBAL_MAX_TEMP/T_scale
            )  # Clamp to screen range
            loss_data = loss_fn(T_pred_data_clamped, data_outputs)

        # --- Initial condition loss (t = 0) ---
        ic_inputs = norm_batch_inputs[is_ic_point]
        ic_outputs = norm_batch_outputs[is_ic_point]

        loss_ic = torch.tensor(0.0, device=config.DEVICE)
        if ic_inputs.shape[0] > 0:
            T_pred_ic = pinn_model(ic_inputs)
            loss_ic = loss_fn(T_pred_ic, ic_outputs)

        # --- boundary condition loss (z = Z) ---
        bc_inputs = norm_batch_inputs[is_bc_point]
        bc_outputs = norm_batch_outputs[is_bc_point]

        loss_bc = torch.tensor(0.0, device=config.DEVICE)
        if bc_inputs.shape[0] > 0:
            T_pred_bc = pinn_model(bc_inputs)
            loss_bc = loss_fn(T_pred_bc, bc_outputs)

        # --- Physics loss (PDE residual) ---
        # Select only PDE points for gradient calculation
        pde_inputs = norm_batch_inputs[is_pde_point].clone()

        loss_phys = torch.tensor(0.0, device=config.DEVICE)

        if pde_inputs.shape[0] > 0:
            t_norm, x_norm, y_norm, z_norm, z_s_norm = pde_inputs.split(1, dim=1)

            # Enable gradients
            t_norm.requires_grad_()
            x_norm.requires_grad_()
            y_norm.requires_grad_()
            z_norm.requires_grad_()

            # Pass the full tensor of PDE inputs
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
            loss_phys = loss_fn(
                residual, torch.zeros_like(residual, device=config.DEVICE)
            )

        # --- Total loss ---
        total_loss = (
            loss_data
            + config.LAMBDA_PHYSICS * loss_phys
            + config.LAMBDA_CONDITIONS * (loss_ic + loss_bc)
        )

        total_loss.backward()
        optimizer.step()

        # === NEW: Accumulate epoch losses ===
        epoch_loss_data += loss_data.item()
        epoch_loss_phys += loss_phys.item()
        epoch_loss_total += total_loss.item()

    # === End of batch loop ===

    # === NEW: Calculate average losses for the epoch ===
    num_batches = len(train_loader)
    avg_epoch_data = epoch_loss_data / num_batches
    avg_epoch_phys = epoch_loss_phys / num_batches
    avg_epoch_total = epoch_loss_total / num_batches

    # === Track losses ===
    loss_data_hist.append(avg_epoch_data)
    loss_phys_hist.append(avg_epoch_phys)
    loss_total_hist.append(avg_epoch_total)

    # === NEW: Step scheduler per EPOCH ===
    scheduler.step()

    # === NEW: Update tqdm progress bar description ===
    epoch_loop.set_postfix(
        Total=f"{avg_epoch_total:.3e}",
        Data=f"{avg_epoch_data:.3e}",
        Phys=f"{avg_epoch_phys:.3e}",
    )

    # if epoch % 10 == 0:  # === REMOVED: This is now handled by tqdm.set_postfix ===
    #     print(
    #         f"Epoch {epoch:04d}: "
    #         f"Total={avg_epoch_total:.3e}, "
    #         f"Data={avg_epoch_data:.3e}, "
    #         f"Phys={avg_epoch_phys:.3e}"
    #     )

train_time = time.time() - start_time
print(f"Training completed in {train_time:.2f} s.")

# === NEW: Save loss values to a log file ===
print("Saving loss history to training_log.txt...")
# Stack epoch numbers with loss histories
log_data = np.column_stack(
    [np.arange(config.N_EPOCHS), loss_total_hist, loss_data_hist, loss_phys_hist]
)
# Save as a CSV-style text file
np.savetxt(
    "training_log.txt",
    log_data,
    header="Epoch, Total_Loss, Data_Loss, Physics_Loss",
    delimiter=",",
    fmt="%.6e",
)


# === Plot training losses ===
print("Generating loss plot...")
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
# === NEW: Save loss plot ===
plt.savefig("loss_plot.png", dpi=300)
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
    # De-normalize: T_abs = (T_norm * T_scale) + T_min
    T_pred_absolute = (
        pinn_model(inputs_infer_torch).cpu().numpy().reshape(ny, nx) * T_scale
    ) + config.GLOBAL_MIN_TEMP

print(
    f"T_pred_absolute range: {T_pred_absolute.min():.3e} to {T_pred_absolute.max():.3e}"
)
print(
    f"vmin={config.GLOBAL_MIN_TEMP}, vmax={np.max([config.GLOBAL_MAX_TEMP, T_pred_absolute.max()])}"
)

# === Plot surface temperature ===
print("Generating final prediction plot...")
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
# This save call was already in your script, so the request is fulfilled
plt.savefig("predicted_temperature.png", dpi=300)
plt.show()

print("Done.")
