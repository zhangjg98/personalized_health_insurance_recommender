import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score
import numpy as np

class NeuralCollaborativeFiltering(nn.Module):
    def __init__(self, num_users, num_items, latent_dim, hidden_dim, dropout_rate=0.3):
        super(NeuralCollaborativeFiltering, self).__init__()
        self.user_embedding = nn.Embedding(num_users, latent_dim)
        self.item_embedding = nn.Embedding(num_items, latent_dim)
        self.fc_layers = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # Add batch normalization
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout
            nn.Linear(hidden_dim, 1)
        )
        self.sigmoid = nn.Sigmoid()  # Add sigmoid activation for binary classification

    def forward(self, user, item):
        user_emb = self.user_embedding(user)
        item_emb = self.item_embedding(item)
        x = torch.cat([user_emb, item_emb], dim=1)
        return self.sigmoid(self.fc_layers(x)).squeeze()  # Apply sigmoid activation

def train_and_save_model(user_item_matrix, latent_dim=50, hidden_dim=128, epochs=20, lr=0.001, dropout_rate=0.3, l2_reg=0.001, model_path="ncf_model.pth"):
    num_users, num_items = user_item_matrix.shape
    model = NeuralCollaborativeFiltering(num_users, num_items, latent_dim, hidden_dim, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=l2_reg)  # Add L2 regularization
    criterion = nn.BCELoss()

    user_item_tensor = torch.tensor(user_item_matrix, dtype=torch.float32)
    user_indices, item_indices = user_item_tensor.nonzero(as_tuple=True)
    ratings = user_item_tensor[user_indices, item_indices]

    # Normalize ratings to the range [0, 1]
    max_rating = ratings.max().item()
    if max_rating > 1.0:
        ratings = ratings / max_rating

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

def load_ncf_model(model_path="ncf_model.pth", num_users=None, num_items=None, latent_dim=50, hidden_dim=128, dropout_rate=0.3):
    """
    Load the Neural Collaborative Filtering model from a saved checkpoint.
    """
    print(f"Loading NCF model from {model_path}...")  # Debugging log

    # Validate num_users and num_items
    if num_users is None or num_items is None or num_users < 1 or num_items < 1:
        print("Invalid dimensions for user-item matrix. Using placeholder dimensions.")  # Debugging log
        num_users, num_items = 1, 7  # Placeholder dimensions

    # Initialize the model with the current dimensions
    model = NeuralCollaborativeFiltering(num_users=num_users, num_items=num_items, latent_dim=latent_dim, hidden_dim=hidden_dim, dropout_rate=dropout_rate)

    try:
        # Attempt to load the saved model
        state_dict = torch.load(model_path)
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")  # Debugging log
    except RuntimeError as e:
        # Handle dimension mismatch errors
        print(f"Model dimensions do not match the current user-item matrix or architecture. Retraining is required. Error: {e}")  # Debugging log
        raise RuntimeError("Model dimensions do not match. Retrain the model.")

    return model

def predict_user_item_interactions(model, user_item_matrix, user_id, top_k=5, matrix_index_to_item_id=None):
    """
    Predict user-item interaction scores using the trained NCF model.

    Parameters:
        model (NeuralCollaborativeFiltering): Trained NCF model.
        user_item_matrix (numpy.ndarray): User-item interaction matrix.
        user_id (int): User ID for predictions.
        top_k (int): Number of top recommendations to return (default: 5).
        matrix_index_to_item_id (dict): Mapping from matrix indices to actual item IDs.

    Returns:
        dict: Item scores mapped to actual item IDs.
    """
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

            # Debugging log: Print the mapping
            print("Matrix index to item ID mapping:", matrix_index_to_item_id)

            # Map matrix indices to actual item IDs
            item_scores = {}
            for matrix_index in range(num_items):
                item_id = matrix_index_to_item_id.get(matrix_index) if matrix_index_to_item_id else matrix_index
                item_scores[item_id] = predictions[matrix_index].item()

            # Debugging log: Print the item scores
            print(f"Item scores for user_id {user_id}: {item_scores}")
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

    # Ensure predictions and true ratings are aligned
    if predictions.shape != true_ratings.shape:
        print(f"Shape mismatch: true_ratings={true_ratings.shape}, predictions={predictions.shape}")  # Debugging log
        raise ValueError("Predictions and true ratings must have the same shape.")

    # Calculate MSE
    mse = mean_squared_error(true_ratings, predictions)

    # Calculate F1-score
    binary_true = (true_ratings >= threshold).astype(int)
    binary_pred = (predictions >= threshold).astype(int)
    f1 = f1_score(binary_true, binary_pred, average="weighted")

    return mse, f1

def precision_at_k(predictions, ground_truth, k):
    """
    Calculate Precision@K.
    """
    top_k_preds = np.argsort(predictions)[-k:][::-1]
    relevant_items = set(np.where(ground_truth > 0)[0])
    recommended_items = set(top_k_preds)
    return len(recommended_items & relevant_items) / k

def recall_at_k(predictions, ground_truth, k):
    """
    Calculate Recall@K.
    """
    top_k_preds = np.argsort(predictions)[-k:][::-1]
    relevant_items = set(np.where(ground_truth > 0)[0])
    recommended_items = set(top_k_preds)
    return len(recommended_items & relevant_items) / len(relevant_items) if relevant_items else 0

def ndcg_at_k(predictions, ground_truth, k):
    """
    Calculate NDCG@K.
    """
    top_k_preds = np.argsort(predictions)[-k:][::-1]
    dcg = sum((ground_truth[i] / np.log2(idx + 2)) for idx, i in enumerate(top_k_preds))
    ideal_dcg = sum((ground_truth[i] / np.log2(idx + 2)) for idx, i in enumerate(np.argsort(ground_truth)[-k:][::-1]))
    return dcg / ideal_dcg if ideal_dcg > 0 else 0

def hit_rate(predictions, ground_truth, k):
    """
    Calculate Hit Rate.
    """
    top_k_preds = np.argsort(predictions)[-k:][::-1]
    relevant_items = set(np.where(ground_truth > 0)[0])
    return 1 if relevant_items & set(top_k_preds) else 0

def evaluate_model_metrics(model, user_item_matrix, k=5):
    """
    Evaluate the model using Precision@K, Recall@K, NDCG@K, and Hit Rate.

    Parameters:
        model (NeuralCollaborativeFiltering): Trained NCF model.
        user_item_matrix (numpy.ndarray): User-item interaction matrix.
        k (int): Number of top recommendations to consider.

    Returns:
        dict: Evaluation metrics.
    """
    num_users, num_items = user_item_matrix.shape
    precision_scores, recall_scores, ndcg_scores, hit_rates = [], [], [], []

    for user_id in range(num_users):
        ground_truth = user_item_matrix[user_id]
        if np.sum(ground_truth) == 0:
            continue  # Skip users with no interactions

        with torch.no_grad():
            predictions = model(
                torch.tensor([user_id] * num_items, dtype=torch.long),
                torch.arange(num_items, dtype=torch.long)
            ).numpy()

        precision_scores.append(precision_at_k(predictions, ground_truth, k))
        recall_scores.append(recall_at_k(predictions, ground_truth, k))
        ndcg_scores.append(ndcg_at_k(predictions, ground_truth, k))
        hit_rates.append(hit_rate(predictions, ground_truth, k))

    return {
        "Precision@K": np.mean(precision_scores),
        "Recall@K": np.mean(recall_scores),
        "NDCG@K": np.mean(ndcg_scores),
        "Hit Rate": np.mean(hit_rates)
    }
