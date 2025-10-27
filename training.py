import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time

import config
import dummy_data
from model import pinn_model, source_term_Q

print(f"using device: {config.DEVICE}")

train_inputs = torch.tensor(np.vstack(dummy_data.all_train_inputs), dtype=torch.float32).to(config.DEVICE)
train_outputs = torch.tensor(np.vstack(dummy_data.all_train_outputs), dtype=torch.float32).to(config.DEVICE)

print(f"Training inputs shape: {train_inputs.shape}")
print(f"Training outputs shape: {train_outputs.shape}")

optimizer = torch.optim.Adam(pinn_model.parameters(), lr=config.LR)
loss_fn = nn.MSELoss()

start_time = time.time()
pinn_model.train()
for epoch in range(config.N_EPOCHS):
    optimizer.zero_grad()
    
    # --- CORREZIONE 1: Controllo sulla colonna Z (indice 3) ---
    is_surface_point = train_inputs[:, 3] < 1e-9 
    
    # 1. Loss on data points (surface points)
    data_inputs = train_inputs[is_surface_point]
    data_outputs = train_outputs[is_surface_point]
    
    # Non c'è bisogno di calcolare i gradienti qui, usa il modello direttamente
    T_pred_data = pinn_model(data_inputs) 
    loss_data = loss_fn(T_pred_data, data_outputs)

    # 2. Loss on physics (collocation points)
    # Usiamo tutti i punti per la fisica (come nel tuo script)
    inputs_for_grad = train_inputs.clone()
    t, x, y, z, z_s = inputs_for_grad.split(1, dim=1)
    
    # --- CORREZIONE 2: Richiedi gradiente anche per Y ---
    t.requires_grad_()
    x.requires_grad_()
    y.requires_grad_() # <-- Aggiunto
    z.requires_grad_()
    
    # --- CORREZIONE 3 (CRASH): Passa tutti e 5 gli input ---
    T_pred_phys = pinn_model(torch.cat([t, x, y, z, z_s], dim=1))
    
    # Calcolo derivate con torch.autograd.grad
    ones = torch.ones_like(T_pred_phys, device=config.DEVICE)
    
    # --- CORREZIONE 4: Calcola le derivate 3D ---
    grads = torch.autograd.grad(T_pred_phys, [t, x, y, z], grad_outputs=ones, create_graph=True)
    dT_dt = grads[0]
    dT_dx = grads[1]
    dT_dy = grads[2] # <-- Aggiunto
    dT_dz = grads[3] # <-- Indice cambiato
    
    dT_dxx = torch.autograd.grad(dT_dx, x, grad_outputs=ones, create_graph=True)[0]
    dT_dyy = torch.autograd.grad(dT_dy, y, grad_outputs=ones, create_graph=True)[0] # <-- Aggiunto
    dT_dzz = torch.autograd.grad(dT_dz, z, grad_outputs=ones, create_graph=True)[0]
    
    # Laplaciano 3D
    laplacian = dT_dxx + dT_dyy + dT_dzz 
    
    # La tua funzione Q(x,z) è 2D, il che è corretto per un filo lungo y
    Q = source_term_Q(x, z, z_s) # Passa solo x, z, z_s (come da tua def)
    
    residual = dT_dt - config.ALPHA * laplacian - Q
    loss_phys = loss_fn(residual, torch.zeros_like(residual))
    
    # 3. Loss Totale
    total_loss = loss_data + config.LAMBDA_PHYSICS * loss_phys
    
    total_loss.backward()
    optimizer.step()
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Total Loss={total_loss.item():.3e}, Data Loss={loss_data.item():.3e}, Physics Loss={loss_phys.item():.3e}")

print(f"Addestramento completato in {time.time() - start_time:.2f} secondi.")