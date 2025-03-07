import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

# Load the processed user-item matrix
user_item_matrix = pd.read_csv('processed_user_item_matrix.csv', index_col=0)

# Convert DataFrame to PyTorch tensor
user_item_tensor = torch.tensor(user_item_matrix.values, dtype=torch.float32)

# Print shape
print(f"User-Item Matrix Shape: {user_item_tensor.shape}")

# Define RBM class
class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))
        self.v_bias = nn.Parameter(torch.zeros(num_visible))

    def forward(self, v):
        """ Forward pass to reconstruct visible layer """
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        h_sample = torch.bernoulli(h_prob)
        v_prob = torch.sigmoid(torch.matmul(h_sample, self.W) + self.v_bias)
        return v_prob

    def contrastive_divergence(self, v0, k=1):
        """ Gibbs Sampling (CD-k) """
        v = v0
        for _ in range(k):
            h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
            h_sample = torch.bernoulli(h_prob)
            v_prob = torch.sigmoid(torch.matmul(h_sample, self.W) + self.v_bias)
            v = torch.bernoulli(v_prob)
        return v0, v

# Initialize RBM
num_visible = user_item_tensor.shape[1]  # Number of features
num_hidden = 10  # Adjustable
rbm = RBM(num_visible, num_hidden)

# Training Configuration
optimizer = optim.SGD(rbm.parameters(), lr=0.01)
epochs = 500
batch_size = 32  # Smaller batches since we now have fewer rows

# Train RBM
for epoch in range(epochs):
    for i in range(0, user_item_tensor.shape[0], batch_size):
        batch = user_item_tensor[i:i+batch_size]
        v0, v1 = rbm.contrastive_divergence(batch)
        loss = torch.mean((v0 - v1) ** 2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# **Get Prediction for a Specific State**
state_name = "CA"  # Example: California

# Ensure the state exists in the index
if state_name in user_item_matrix.index:
    sample_input = torch.tensor(user_item_matrix.loc[state_name].values, dtype=torch.float32).unsqueeze(0)  # Ensure shape (1, features)
else:
    raise ValueError(f"State '{state_name}' not found in dataset.")

# Generate Prediction
predicted_output = rbm(sample_input).squeeze().detach().numpy()

# Convert to DataFrame for readability
predicted_df = pd.DataFrame([predicted_output], columns=user_item_matrix.columns)

print(f"\nPredicted Medicare Spending & Utilization for {state_name}:")
print(predicted_df)
