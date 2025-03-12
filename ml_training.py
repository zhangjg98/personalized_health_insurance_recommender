import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from models import DeepAutoencoder

# Load the processed user-item matrix
user_item_matrix = pd.read_csv('processed_user_item_matrix.csv', index_col=0)

# Scale the data
scaler = StandardScaler()
user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)

# Apply PCA
n_components = 9
pca = PCA(n_components=n_components, random_state=42)
user_item_matrix_pca = pca.fit_transform(user_item_matrix_scaled)

# Convert the PCA-transformed data to a PyTorch tensor
user_item_tensor = torch.tensor(user_item_matrix_pca, dtype=torch.float32)

# Adjust model parameters
num_hidden_1 = 200  # Increase model capacity
num_hidden_2 = 100
num_hidden_3 = 50
num_latent = 25  # Expand latent space

kf = KFold(n_splits=5, shuffle=True, random_state=42)
epochs = 1000  # Increase epochs for better training
batch_size = 32  # Adjust batch size
early_stopping_patience = 20

best_overall_loss = float('inf')
best_params = None

for lr in [0.0001, 0.0005, 0.001]:  # Experiment with different learning rates
    fold_losses = []
    for fold, (train_index, val_index) in enumerate(kf.split(user_item_tensor)):
        train_data = user_item_tensor[train_index]
        val_data = user_item_tensor[val_index]
        
        autoencoder = DeepAutoencoder(num_visible=n_components,
                                      num_hidden_1=num_hidden_1,
                                      num_hidden_2=num_hidden_2,
                                      num_hidden_3=num_hidden_3,
                                      num_latent=num_latent)

        optimizer = optim.Adam(autoencoder.parameters(), lr=lr, weight_decay=0.001)  # Add L2 regularization
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            autoencoder.train()
            epoch_loss = 0
            for i in range(0, train_data.shape[0], batch_size):
                batch = train_data[i:i+batch_size]
                output = autoencoder(batch)
                output = torch.sigmoid(output)  # Apply sigmoid to ensure values are between 0 and 1
                batch = torch.sigmoid(batch)  # Apply sigmoid to ensure target values are between 0 and 1
                loss = nn.BCELoss()(output, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss)

            # Validation
            autoencoder.eval()
            val_loss = 0
            batch_count = max(len(val_data) // batch_size, 1)
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i:i+batch_size]
                    output = autoencoder(batch)
                    output = torch.sigmoid(output)  # Apply sigmoid to ensure values are between 0 and 1
                    batch = torch.sigmoid(batch)  # Apply sigmoid to ensure target values are between 0 and 1
                    val_loss += nn.BCELoss()(output, batch).item()
            val_loss /= batch_count
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    break

        fold_losses.append(best_val_loss)

    avg_loss = sum(fold_losses) / len(fold_losses)
    if avg_loss < best_overall_loss:
        best_overall_loss = avg_loss
        best_params = {'lr': lr}

# Train final model
final_autoencoder = DeepAutoencoder(num_visible=n_components,
                                    num_hidden_1=num_hidden_1,
                                    num_hidden_2=num_hidden_2,
                                    num_hidden_3=num_hidden_3,
                                    num_latent=num_latent)

optimizer = optim.Adam(final_autoencoder.parameters(), lr=best_params['lr'], weight_decay=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

for epoch in range(epochs):
    final_autoencoder.train()
    for i in range(0, user_item_tensor.shape[0], batch_size):
        batch = user_item_tensor[i:i+batch_size]
        output = final_autoencoder(batch)
        output = torch.sigmoid(output)  # Apply sigmoid to ensure values are between 0 and 1
        batch = torch.sigmoid(batch)  # Apply sigmoid to ensure target values are between 0 and 1
        loss = nn.BCELoss()(output, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save final model
torch.save(final_autoencoder.state_dict(), 'final_autoencoder.pth')
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('pca.pkl', 'wb') as f:
    pickle.dump(pca, f)
