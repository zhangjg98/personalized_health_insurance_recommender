import torch
import torch.nn as nn
import torch.optim as optim
from models import NeuralCollaborativeFiltering  # Import from models.py

def train_and_save_model(user_item_matrix, latent_dim=20, hidden_dim=64, epochs=10, lr=0.001, dropout_rate=0.3, l2_reg=0.001, model_path="ncf_model.pth", pretrained_item_embeddings=None):
    """
    Train and save the Neural Collaborative Filtering (NCF) model.
    """
    num_users, num_items = user_item_matrix.shape
    model = NeuralCollaborativeFiltering(num_users, num_items, latent_dim, hidden_dim, dropout_rate, pretrained_item_embeddings)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)
    criterion = nn.BCELoss()

    user_item_tensor = torch.tensor(user_item_matrix, dtype=torch.float32)
    user_indices, item_indices = user_item_tensor.nonzero(as_tuple=True)
    ratings = user_item_tensor[user_indices, item_indices]

    # Normalize ratings to the range [0, 1]
    max_rating = ratings.max().item()
    if max_rating > 1.0:
        ratings = ratings / max_rating

    # Assign higher weights to rare items
    item_interaction_counts = torch.bincount(item_indices, minlength=num_items)
    weights = 1.0 / (item_interaction_counts[item_indices] + 1.0)  # Inverse frequency weighting

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(user_indices, item_indices)
        loss = criterion(predictions, ratings) * weights.mean()  # Apply weighted loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "num_users": num_users,
        "num_items": num_items,
        "latent_dim": latent_dim,
        "hidden_dim": hidden_dim,
        "dropout_rate": dropout_rate
    }, model_path)
    print(f"Model saved to {model_path}")
    return model
