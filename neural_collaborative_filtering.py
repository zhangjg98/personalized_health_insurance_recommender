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
    # Load the correct user-item matrix
    user_item_matrix = pd.read_csv("user_item_matrix.csv", index_col=0)
    num_users, num_items = user_item_matrix.shape

    # Validate the dimensions of the user-item matrix
    print(f"User-item matrix dimensions: {num_users} users, {num_items} items")

    model = NeuralCollaborativeFiltering(num_users, num_items, latent_dim, hidden_dim)
    try:
        model.load_state_dict(torch.load(model_path))
        model.eval()
        print("Model loaded successfully.")
    except RuntimeError as e:
        raise RuntimeError(
            f"Model dimensions do not match the current user-item matrix. "
            f"Retrain the model to fix this issue. Error: {e}"
        )

    return model, user_item_matrix

def predict_user_item_interactions(model, user_item_matrix, user_id, top_k=5):
    print("Starting predict_user_item_interactions function...")  # Debugging log
    print(f"User ID: {user_id}, Top-K: {top_k}")  # Debugging log

    num_users, num_items = user_item_matrix.shape
    print(f"User-item matrix dimensions: {num_users} users, {num_items} items")  # Debugging log

    # Validate user_id
    if user_id < 0 or user_id >= num_users:
        print(f"Invalid user_id: {user_id}. Valid range: [0, {num_users - 1}]")  # Debugging log
        raise IndexError(f"User ID {user_id} is out of range. Valid range: [0, {num_users - 1}]")

    # Validate item indices
    if num_items < 2:  # Ensure at least 2 items for meaningful predictions
        print("Insufficient items in the user-item matrix for predictions.")  # Debugging log
        return {}  # Return an empty dictionary if there are not enough items

    # Adjust top_k to ensure it does not exceed the number of items
    top_k = min(top_k, num_items) if top_k else num_items

    user_tensor = torch.tensor([user_id], dtype=torch.long)
    item_tensor = torch.arange(num_items, dtype=torch.long)

    try:
        with torch.no_grad():
            predictions = model(user_tensor.repeat(len(item_tensor)), item_tensor)
            print(f"Predictions for user_id {user_id}: {predictions}")  # Debugging log

            # Validate predictions
            if predictions is None or predictions.numel() == 0:
                print("No valid predictions generated.")  # Debugging log
                return {}  # Return an empty dictionary if predictions are invalid

            # Map item indices to scores
            item_scores = {item_idx: predictions[item_idx].item() for item_idx in range(num_items)}
            print(f"Item scores for user_id {user_id}: {item_scores}")  # Debugging log

            return item_scores
    except Exception as e:
        print(f"Error during prediction: {e}")  # Debugging log
        return {}  # Return an empty dictionary if an error occurs

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
