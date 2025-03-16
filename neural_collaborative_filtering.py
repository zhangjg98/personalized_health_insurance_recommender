import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, hidden_dim):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.fc_layers(x).squeeze()

def train_and_save_model(user_item_matrix, latent_dim=50, hidden_dim=128, epochs=20, lr=0.001, model_path="ncf_model.pth"):
    num_users, num_items = user_item_matrix.shape
    model = NeuralCollaborativeFiltering(num_users, num_items, latent_dim, hidden_dim)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    user_item_tensor = torch.tensor(user_item_matrix, dtype=torch.float32)
    user_indices, item_indices = user_item_tensor.nonzero(as_tuple=True)
    ratings = user_item_tensor[user_indices, item_indices]

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        predictions = model(user_indices, item_indices)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model

def load_ncf_model(model_path="ncf_model.pth", latent_dim=50, hidden_dim=128):
    user_item_matrix = pd.read_csv("processed_user_item_matrix.csv", index_col=0)
    num_users, num_items = user_item_matrix.shape
    model = NeuralCollaborativeFiltering(num_users, num_items, latent_dim, hidden_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model, user_item_matrix

def predict_user_item_interactions(model, user_item_matrix, user_id, top_k=5):
    user_tensor = torch.tensor([user_id], dtype=torch.long)
    item_tensor = torch.arange(user_item_matrix.shape[1], dtype=torch.long)

    with torch.no_grad():
        predictions = model(user_tensor.repeat(len(item_tensor)), item_tensor)
        top_items = torch.topk(predictions, top_k).indices.numpy()

    item_names = user_item_matrix.columns[top_items].tolist()
    return item_names

def evaluate_model(model, user_item_matrix, threshold=0.5):
    """
    Evaluate the model using MSE and F1-score.

    Parameters:
        model (NeuralCollaborativeFiltering): Trained NCF model.
        user_item_matrix (numpy.ndarray): User-item interaction matrix.
        threshold (float): Threshold for binary classification (default: 0.5).

    Returns:
        tuple: Mean Squared Error (MSE) and F1-score.
    """
    user_item_tensor = torch.tensor(user_item_matrix, dtype=torch.float32)
    user_indices, item_indices = user_item_tensor.nonzero(as_tuple=True)
    true_ratings = user_item_tensor[user_indices, item_indices].numpy()

    with torch.no_grad():
        predictions = model(user_indices, item_indices).numpy()

    # Calculate MSE
    mse = mean_squared_error(true_ratings, predictions)

    # Calculate F1-score
    binary_true = (true_ratings >= threshold).astype(int)
    binary_pred = (predictions >= threshold).astype(int)
    f1 = f1_score(binary_true, binary_pred, average="weighted")

    return mse, f1
