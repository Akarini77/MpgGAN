# -*- coding: utf-8 -*-

"""
GAN-based generation of four rock discontinuity parameters.

Parameters:
    1. dip_direction
    2. dip_angle
    3. spacing
    4. trace_length
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


# ============================================================
# 1. File paths
# ============================================================

input_file = "data/preprocessed_data.csv"
results_dir = "results"

# Create results folder automatically
os.makedirs(results_dir, exist_ok=True)


# ============================================================
# 2. Load training data
# ============================================================

df = pd.read_csv(
    input_file,
    header=None
)

torch_data = torch.as_tensor(
    df.values,
    dtype=torch.float32
)

# Use the original code's selected rows
torch_data = torch_data[1:200]

loader = DataLoader(
    dataset=torch_data,
    batch_size=1,
    shuffle=True
)


# ============================================================
# 3. GAN parameters
# ============================================================

num_epoch = 2       # Change to 30 for the full experiment
z_dimension = 3     # Latent noise dimension

# Number of discontinuity parameters
input_dimension = 4


# ============================================================
# 4. Discriminator
# ============================================================

D = nn.Sequential(

    # Input: 4 discontinuity parameters
    nn.Linear(input_dimension, 512),
    nn.ReLU(),

    nn.Linear(512, 256),
    nn.ReLU(),

    nn.Linear(256, 128),
    nn.ReLU(),

    nn.Linear(128, 128),
    nn.ReLU(),

    nn.Linear(128, 256),
    nn.ReLU(),

    nn.Linear(256, 512),
    nn.ReLU(),

    # Output: real / fake
    nn.Linear(512, 1),
    nn.Sigmoid(),
)


# ============================================================
# 5. Generator
# ============================================================

G = nn.Sequential(

    # Input: 3-dimensional latent noise
    nn.Linear(z_dimension, 1024),
    nn.Tanh(),

    nn.Linear(1024, 512),
    nn.Tanh(),

    nn.Linear(512, 256),
    nn.Tanh(),

    nn.Linear(256, 128),
    nn.Tanh(),

    nn.Linear(128, 256),
    nn.Tanh(),

    nn.Linear(256, 256),
    nn.Tanh(),

    nn.Linear(256, 512),
    nn.Tanh(),

    # Output: 4 discontinuity parameters
    nn.Linear(512, input_dimension),
    nn.Sigmoid(),
)


# ============================================================
# 6. Loss function and optimizers
# ============================================================

loss_func = nn.BCELoss()

d_optimizer = torch.optim.Adam(
    D.parameters(),
    lr=0.00006
)

g_optimizer = torch.optim.Adam(
    G.parameters(),
    lr=0.000001
)


# ============================================================
# 7. Training
# ============================================================

g_losses = []
d_losses = []


for epoch in range(num_epoch):

    print(
        "---------- Epoch",
        epoch + 1,
        "of",
        num_epoch,
        "----------"
    )

    for step, x in enumerate(loader):

        current_batch_size = x.shape[0]

        # ----------------------------------------------------
        # Labels
        # ----------------------------------------------------

        real_labels = torch.ones(
            current_batch_size,
            1
        )

        fake_labels = torch.zeros(
            current_batch_size,
            1
        )

        # ----------------------------------------------------
        # Train discriminator
        # ----------------------------------------------------

        # Real data
        real_out = D(x)

        d_loss_real = loss_func(
            real_out,
            real_labels
        )

        # Fake data
        z = torch.randn(
            current_batch_size,
            z_dimension
        )

        fake_data = G(z)

        fake_out = D(fake_data)

        d_loss_fake = loss_func(
            fake_out,
            fake_labels
        )

        # Total discriminator loss
        d_loss = (
            d_loss_real +
            d_loss_fake
        )

        d_optimizer.zero_grad()

        d_loss.backward()

        d_optimizer.step()

        d_losses.append(
            d_loss.item()
        )

        # ----------------------------------------------------
        # Train generator
        # ----------------------------------------------------

        for _ in range(5):

            z = torch.randn(
                current_batch_size,
                z_dimension
            )

            fake_data = G(z)

            outputs = D(fake_data)

            g_loss = loss_func(
                outputs,
                real_labels
            )

            g_optimizer.zero_grad()

            g_loss.backward()

            g_optimizer.step()

        g_losses.append(
            g_loss.item()
        )

    print(
        "---------- Epoch",
        epoch + 1,
        "finished ----------"
    )


# ============================================================
# 8. Generate new discontinuity samples
# ============================================================

n_samples = 100

z = torch.randn(
    n_samples,
    z_dimension
)

with torch.no_grad():

    fake_data = G(z)


# Convert generated data to NumPy
generated_data = fake_data.cpu().numpy()


# ============================================================
# 9. Save generated normalized data
# ============================================================

prediction_file = os.path.join(
    results_dir,
    "prediction.csv"
)

np.savetxt(
    prediction_file,
    generated_data,
    delimiter=","
)

print(
    f"Generated {n_samples} samples."
)

print(
    f"Generated data saved to: {prediction_file}"
)


# ============================================================
# 10. Save training losses
# ============================================================

g_loss_file = os.path.join(
    results_dir,
    "g_loss.csv"
)

d_loss_file = os.path.join(
    results_dir,
    "d_loss.csv"
)

np.savetxt(
    g_loss_file,
    g_losses,
    delimiter=","
)

np.savetxt(
    d_loss_file,
    d_losses,
    delimiter=","
)


# ============================================================
# 11. Save trained models
# ============================================================

generator_file = os.path.join(
    results_dir,
    "generator.pth"
)

discriminator_file = os.path.join(
    results_dir,
    "discriminator.pth"
)

torch.save(
    G.state_dict(),
    generator_file
)

torch.save(
    D.state_dict(),
    discriminator_file
)


# ============================================================
# 12. Plot training losses
# ============================================================

plt.figure()

plt.plot(
    g_losses,
    label="Generator"
)

plt.plot(
    d_losses,
    label="Discriminator"
)

plt.xlabel(
    "Training iteration"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.tight_layout()

loss_plot_file = os.path.join(
    results_dir,
    "training_loss.png"
)

plt.savefig(
    loss_plot_file,
    dpi=300
)

plt.show()


# ============================================================
# 13. Training completed
# ============================================================

print("\nTraining completed.")
print(f"Generator model: {generator_file}")
print(f"Discriminator model: {discriminator_file}")
print(f"Generated samples: {prediction_file}")
print(f"Loss figure: {loss_plot_file}")