import torch
import torch.nn as nn

class DeepAutoencoder(nn.Module):
    def __init__(self, num_visible, num_hidden_1, num_hidden_2, num_hidden_3, num_latent):
        super(DeepAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(num_visible, num_hidden_1),
            nn.BatchNorm1d(num_hidden_1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(num_hidden_1, num_hidden_2),
            nn.BatchNorm1d(num_hidden_2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(num_hidden_2, num_hidden_3),
            nn.BatchNorm1d(num_hidden_3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(num_hidden_3, num_latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(num_latent, num_hidden_3),
            nn.BatchNorm1d(num_hidden_3),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(num_hidden_3, num_hidden_2),
            nn.BatchNorm1d(num_hidden_2),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(num_hidden_2, num_hidden_1),
            nn.BatchNorm1d(num_hidden_1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(num_hidden_1, num_visible),
            nn.Sigmoid()  # Ensure output is between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
