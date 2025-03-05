import torch
import torch.nn as nn
import torch.optim as optim

# Import user_item_matrix from matrix_conversion.py
from matrix_conversion import user_item_matrix

# Convert Pandas DataFrame to PyTorch tensor
user_item_tensor = torch.tensor(user_item_matrix.values, dtype=torch.float32)

# Define RBM class
class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super(RBM, self).__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden

        # Initialize weights and biases
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible) * 0.1)
        self.h_bias = nn.Parameter(torch.zeros(num_hidden))
        self.v_bias = nn.Parameter(torch.zeros(num_visible))

    def forward(self, v):
        """ Forward pass: Visible to Hidden to Reconstructed Visible """
        # Hidden probabilities and sampling
        h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
        h_sample = torch.bernoulli(h_prob)

        # Visible probabilities and sampling
        v_prob = torch.sigmoid(torch.matmul(h_sample, self.W) + self.v_bias)
        return v_prob

    def contrastive_divergence(self, v0, k=1):
        """ Perform Gibbs Sampling (CD-k) """
        v = v0
        for _ in range(k):
            h_prob = torch.sigmoid(torch.matmul(v, self.W.t()) + self.h_bias)
            h_sample = torch.bernoulli(h_prob)
            v_prob = torch.sigmoid(torch.matmul(h_sample, self.W) + self.v_bias)
            v = torch.bernoulli(v_prob)

        return v0, v

# Initialize RBM
num_visible = user_item_tensor.shape[1]  # Number of input features
num_hidden = 10  # Number of hidden nodes (adjustable)
rbm = RBM(num_visible, num_hidden)

# Training RBM
optimizer = optim.SGD(rbm.parameters(), lr=0.1)
epochs = 500

for epoch in range(epochs):
    v0, v1 = rbm.contrastive_divergence(user_item_tensor)
    loss = torch.mean((v0 - v1) ** 2)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")