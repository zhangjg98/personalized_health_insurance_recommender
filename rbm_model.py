import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold

# Load the processed user-item matrix (6 features)
user_item_matrix = pd.read_csv('processed_user_item_matrix.csv', index_col=0)

# Scale the data using StandardScaler
scaler = StandardScaler()
user_item_matrix_scaled = scaler.fit_transform(user_item_matrix)

# Use TruncatedSVD to reduce dimensionality (e.g., to 4 components)
n_components = 4  # Must be <= number of original features (6)
svd = TruncatedSVD(n_components=n_components, random_state=42)
user_item_matrix_svd = svd.fit_transform(user_item_matrix_scaled)

# Convert the SVD-transformed data to a PyTorch tensor
user_item_tensor = torch.tensor(user_item_matrix_svd, dtype=torch.float32)
print(f"User-Item Matrix Shape (SVD reduced): {user_item_tensor.shape}")


# Define a Hybrid RBM Model That Works on SVD Features

class HybridRBM_SVD(nn.Module):
    def __init__(self, num_visible, num_hidden_1, num_hidden_2, num_latent):
        """
        Here, num_visible corresponds to n_components (the SVD-reduced dimensionality).
        """
        super(HybridRBM_SVD, self).__init__()
        self.num_visible = num_visible
        self.num_hidden_1 = num_hidden_1
        self.num_hidden_2 = num_hidden_2
        self.num_latent = num_latent
        
        # First hidden layer
        self.layer1 = nn.Linear(num_visible, num_hidden_1)
        self.relu1 = nn.ReLU()
        
        # Second hidden layer
        self.layer2 = nn.Linear(num_hidden_1, num_hidden_2)
        self.relu2 = nn.ReLU()
        
        # Latent layer (collaborative filtering-like)
        self.latent_layer = nn.Linear(num_hidden_2, num_latent)
        
        # Output layer: reconstruct back to SVD space (num_visible)
        self.output_layer = nn.Linear(num_latent, num_visible)
        
        # Bias parameters (optional here, as layers already have bias)
        self.v_bias = nn.Parameter(torch.zeros(num_visible))
        self.h_bias = nn.Parameter(torch.zeros(num_latent))
        
    def forward(self, v):
        h1 = torch.sigmoid(self.layer1(v))
        h1 = self.relu1(h1)
        
        h2 = torch.sigmoid(self.layer2(h1))
        h2 = self.relu2(h2)
        
        latent = self.latent_layer(h2)
        v_prob = torch.sigmoid(self.output_layer(latent))
        return v_prob

    def contrastive_divergence(self, v0, k=5):
        v = v0
        for _ in range(k):
            h1 = torch.sigmoid(self.layer1(v))
            h1_sample = torch.bernoulli(h1)
            
            h2 = torch.sigmoid(self.layer2(h1_sample))
            h2_sample = torch.bernoulli(h2)
            
            latent = self.latent_layer(h2_sample)
            v_prob = torch.sigmoid(self.output_layer(latent))
            v = torch.bernoulli(v_prob)
        return v0, v

# Define a hyperparameter grid
param_grid = {
    'lr': [0.001, 0.005]
}

# Fixed hyperparameters for the model
num_hidden_1 = 50
num_hidden_2 = 30
num_latent = 10  # latent dimension (must be <= num_visible ideally)

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
epochs = 2000
batch_size = 64
early_stopping_patience = 50

best_overall_loss = float('inf')
best_params = None

# Loop over the hyperparameter grid and perform CV.
for lr in param_grid['lr']:
    print(f"\nTuning with learning rate: {lr}")
    fold_losses = []
    for fold, (train_index, val_index) in enumerate(kf.split(user_item_tensor)):
        print(f"  Training fold {fold+1}")
        
        # Split data for this fold
        train_data = user_item_tensor[train_index]
        val_data = user_item_tensor[val_index]
        
        # Initialize the model; note: input dimension = n_components from SVD
        rbm = HybridRBM_SVD(num_visible=n_components,
                            num_hidden_1=num_hidden_1,
                            num_hidden_2=num_hidden_2,
                            num_latent=num_latent)
        
        # Set up optimizer and scheduler
        optimizer = optim.Adam(rbm.parameters(), lr=lr, weight_decay=0.01)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        # Training loop for this fold
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
            
            # Compute validation loss
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

# Use the best learning rate from tuning
final_lr = best_params['lr']
print(f"\nFinal training on full SVD-transformed data with learning rate: {final_lr}")

# Initialize the final model
final_rbm = HybridRBM_SVD(num_visible=n_components,
                          num_hidden_1=num_hidden_1,
                          num_hidden_2=num_hidden_2,
                          num_latent=num_latent)

optimizer = optim.Adam(final_rbm.parameters(), lr=final_lr, weight_decay=0.01)
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

# For prediction, we need to transform the sample data with the same scaler and SVD.

state_name = "CA"
numeric_aggregated_state = user_item_matrix.loc[state_name].to_frame().T

# Scale the sample using the same scaler
sample_scaled = scaler.transform(numeric_aggregated_state)

# Transform the sample using the fitted SVD model
sample_svd = svd.transform(sample_scaled)
sample_input = torch.tensor(sample_svd, dtype=torch.float32)

# Get prediction in SVD space
final_rbm.eval()
predicted_svd = final_rbm(sample_input).squeeze(0)
predicted_svd_np = predicted_svd.detach().numpy().reshape(1, -1)

# Inverse transform from SVD space back to scaled space
reconstructed_scaled = svd.inverse_transform(predicted_svd_np)

# Then inverse transform from StandardScaler to original space
reconstructed_output = scaler.inverse_transform(reconstructed_scaled)

# Convert the output back to a DataFrame with the original column names
predicted_df = pd.DataFrame(reconstructed_output, columns=user_item_matrix.columns)
print(f"\n Predicted Medicare Spending & Utilization for {state_name}:")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(predicted_df)

torch.save(final_rbm.state_dict(), 'final_rbm.pth')
import pickle
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('svd.pkl', 'wb') as f:
    pickle.dump(svd, f)