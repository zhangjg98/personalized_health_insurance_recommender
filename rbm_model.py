import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from models import HybridRBM_SVD

# Load the processed user-item matrix (6 features)
user_item_matrix = pd.read_csv('processed_user_item_matrix.csv', index_col=0)

# Scale the data using StandardScaler
scaler = StandardScaler()
user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)

# Increase SVD components to preserve more state-specific variance (e.g., 6 components)
n_components = 9
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_item_matrix_svd = svd.fit_transform(user_item_matrix_scaled)

# Convert the SVD-transformed data to a PyTorch tensor
user_item_tensor = torch.tensor(user_item_matrix_svd, dtype=torch.float32)
print(f"User-Item Matrix Shape (SVD reduced): {user_item_tensor.shape}")

# Fixed hyperparameters for the model (adjust if needed)
num_hidden_1 = 50
num_hidden_2 = 30
num_latent = 20

# Cross-validation setup and training (code remains largely the same)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
epochs = 2000
batch_size = 64
early_stopping_patience = 50

best_overall_loss = float('inf')
best_params = None

for lr in [0.001, 0.005]:
    print(f"\nTuning with learning rate: {lr}")
    fold_losses = []
    for fold, (train_index, val_index) in enumerate(kf.split(user_item_tensor)):
        print(f"  Training fold {fold+1}")
        train_data = user_item_tensor[train_index]
        val_data = user_item_tensor[val_index]
        
        rbm = HybridRBM_SVD(num_visible=n_components,
                            num_hidden_1=num_hidden_1,
                            num_hidden_2=num_hidden_2,
                            num_latent=num_latent)
        optimizer = optim.Adam(rbm.parameters(), lr=lr, weight_decay=0.005)  # Slightly lower weight_decay
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            rbm.train()
            epoch_loss = 0
            for i in range(0, train_data.shape[0], batch_size):
                batch = train_data[i:i+batch_size]
                v0, v1 = rbm.contrastive_divergence(batch)
                loss = torch.mean((v0 - v1) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            scheduler.step(epoch_loss)
            
            rbm.eval()
            val_loss = 0
            batch_count = max(len(val_data) // batch_size, 1)
            with torch.no_grad():
                for i in range(0, len(val_data), batch_size):
                    batch = val_data[i:i+batch_size]
                    v0, v1 = rbm.contrastive_divergence(batch)
                    val_loss += torch.mean((v0 - v1) ** 2).item()
            val_loss /= batch_count
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"    Early stopping at epoch {epoch} in fold {fold+1}")
                    break
            if epoch % 50 == 0:
                print(f"    Fold {fold+1}, Epoch {epoch}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")
        fold_losses.append(best_val_loss)
        print(f"  Fold {fold+1} best Val Loss: {best_val_loss:.6f}")
    avg_loss = sum(fold_losses) / len(fold_losses)
    print(f"Learning rate {lr} average Val Loss: {avg_loss:.6f}")
    if avg_loss < best_overall_loss:
        best_overall_loss = avg_loss
        best_params = {'lr': lr}

print("\nBest hyperparameters found:", best_params)

final_lr = best_params['lr']
print(f"\nFinal training on full SVD-transformed data with learning rate: {final_lr}")

final_rbm = HybridRBM_SVD(num_visible=n_components,
                          num_hidden_1=num_hidden_1,
                          num_hidden_2=num_hidden_2,
                          num_latent=num_latent)

optimizer = optim.Adam(final_rbm.parameters(), lr=final_lr, weight_decay=0.005)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)

best_loss_full = float('inf')
patience_counter = 0

for epoch in range(epochs):
    final_rbm.train()
    epoch_loss = 0
    for i in range(0, user_item_tensor.shape[0], batch_size):
        batch = user_item_tensor[i:i+batch_size]
        v0, v1 = final_rbm.contrastive_divergence(batch)
        loss = torch.mean((v0 - v1) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    scheduler.step(epoch_loss)
    
    if epoch_loss < best_loss_full:
        best_loss_full = epoch_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= early_stopping_patience:
            print(f"Final training early stopping at epoch {epoch}")
            break
    if epoch % 50 == 0:
        print(f"Final model, Epoch {epoch}, Loss: {epoch_loss:.6f}")

# Save final model state dictionary
torch.save(final_rbm.state_dict(), 'final_rbm.pth')
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('svd.pkl', 'wb') as f:
    pickle.dump(svd, f)
