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

# Define RBM
class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))
        self.v_bias = nn.Parameter(torch.zeros(num_visible))

    def forward(self, v):
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        h_sample = torch.bernoulli(h_prob)
        v_prob = torch.sigmoid(torch.matmul(h_sample, self.W) + self.v_bias)
        return v_prob

    def contrastive_divergence(self, v0, k=1):
        v = v0
        for _ in range(k):
            h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
            h_sample = torch.bernoulli(h_prob)
            v_prob = torch.sigmoid(torch.matmul(h_sample, self.W) + self.v_bias)
            v = torch.bernoulli(v_prob)
        return v0, v

# Initialize RBM
num_visible = user_item_tensor.shape[1]
num_hidden = 100
rbm = RBM(num_visible, num_hidden)

# Learning Rate & Optimizer
optimizer = optim.SGD(rbm.parameters(), lr=0.0075)
epochs = 1250
batch_size = 32

# Training Loop
for epoch in range(epochs):
    for i in range(0, user_item_tensor.shape[0], batch_size):
        batch = user_item_tensor[i:i+batch_size]
        v0, v1 = rbm.contrastive_divergence(batch)
        loss = torch.mean((v0 - v1) ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

# Example Prediction for states
state_name = "NC"
numeric_aggregated_state = user_item_matrix.loc[state_name].to_frame().T
sample_input = torch.tensor(numeric_aggregated_state.values, dtype=torch.float32)

predicted_output = rbm(sample_input).squeeze(0)
predicted_df = pd.DataFrame([predicted_output.detach().numpy()], columns=numeric_aggregated_state.columns)
print(f"\n Predicted Medicare Spending & Utilization for {state_name}:")
print(predicted_df)