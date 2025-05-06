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

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, hidden_dim, dropout_rate=0.3, pretrained_item_embeddings=None):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)

        # Initialize item embeddings with pretrained embeddings if provided
        if pretrained_item_embeddings is not None:
            self.item_embedding.weight.data.copy_(torch.tensor(pretrained_item_embeddings, dtype=torch.float32))

        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.sigmoid(self.fc_layers(x)).squeeze()
