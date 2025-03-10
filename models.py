import torch
import torch.nn as nn

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
