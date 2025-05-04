import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ml_model import generate_embeddings  # Import content-based embedding generator
import psutil  # Import psutil for memory monitoring
import os  # Import os for process management

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

def train_and_save_model(user_item_matrix, latent_dim=50, hidden_dim=128, epochs=20, lr=0.001, dropout_rate=0.3, l2_reg=0.001, model_path="ncf_model.pth", pretrained_item_embeddings=None):
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

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model

def hybrid_recommendation(user_input, user_item_matrix, model, top_k=5, matrix_index_to_item_id=None, plans=None):
    """
    Combine collaborative filtering and content-based filtering for recommendations.

    Parameters:
        user_input (dict): User inputs for content-based filtering.
        user_item_matrix (numpy.ndarray): User-item interaction matrix.
        model (NeuralCollaborativeFiltering): Trained NCF model.
        top_k (int): Number of top recommendations to return.
        matrix_index_to_item_id (dict): Mapping from matrix indices to actual item IDs.
        plans (list): List of plans for content-based filtering.

    Returns:
        list: Combined recommendations.
    """
    print("Starting hybrid recommendation system...")  # Debugging log

    # Collaborative Filtering Predictions
    user_id = user_input.get("user_id", 0)
    try:
        user_index = int(user_id)  # Try converting to int
    except ValueError:
        user_index = None  # Default to None if user_id is not valid

    num_items = user_item_matrix.shape[1]
    if user_index is not None and 0 <= user_index < user_item_matrix.shape[0]:
        # Valid user ID found in the matrix
        user_tensor = torch.tensor([user_index] * num_items, dtype=torch.long)
    else:
        # Use a default "average user" profile for new/guest users
        print("User ID not found in the matrix. Using default collaborative filtering profile.")  # Debugging log
        average_user_profile = user_item_matrix.mean(axis=0)
        user_tensor = torch.tensor([0] * num_items, dtype=torch.long)  # Dummy tensor for compatibility
        average_user_tensor = torch.tensor(average_user_profile, dtype=torch.float32)

    item_tensor = torch.arange(num_items, dtype=torch.long)

    with torch.no_grad():
        if user_index is not None and 0 <= user_index < user_item_matrix.shape[0]:
            cf_predictions = model(user_tensor, item_tensor).numpy()
        else:
            # Use the average user profile for predictions
            cf_predictions = average_user_tensor.numpy()

    # Map predictions to item IDs
    cf_scores = {matrix_index_to_item_id[i]: cf_predictions[i] for i in range(num_items)}

    # Content-Based Filtering
    if plans:
        cb_recommendations = generate_embeddings(user_input, plans)
        cb_scores = {plan["id"]: plan["similarity_score"] for plan in cb_recommendations}
    else:
        cb_scores = {}

    # Combine Scores
    combined_scores = {}
    for item_id in set(cf_scores.keys()).union(cb_scores.keys()):
        cf_score = cf_scores.get(item_id, 0)
        cb_score = cb_scores.get(item_id, 0)
        combined_scores[item_id] = 0.7 * cf_score + 0.3 * cb_score  # Adjust weights as needed

    # Sort by combined scores
    sorted_items = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    # Format recommendations
    recommendations = [{"item_id": item_id, "score": score} for item_id, score in sorted_items]
    return recommendations

def recall_at_k_dynamic(predictions, ground_truth, k):
    """
    Dynamically adjust Recall@K based on the number of relevant items.

    Parameters:
        predictions (numpy.ndarray): Predicted scores for items.
        ground_truth (numpy.ndarray): Ground truth relevance scores.
        k (int): Number of top recommendations to consider.

    Returns:
        float: Recall@K.
    """
    relevant_items = set(np.where(ground_truth > 0)[0])
    recommended_items = set(np.argsort(predictions)[-k:][::-1])
    adjusted_k = min(k, len(relevant_items))  # Adjust K dynamically
    return len(recommended_items & relevant_items) / adjusted_k if adjusted_k > 0 else 0

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
    process = psutil.Process(os.getpid())
    print(f"Memory usage at start: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    num_users, num_items = user_item_matrix.shape
    print(f"User-item matrix dimensions: {num_users} users, {num_items} items")  # Debugging log

    # Validate user_id
    if user_id is None or user_id < 0 or user_id >= num_users:
        print(f"Invalid or missing user_id: {user_id}. Skipping collaborative filtering.")  # Debugging log
        return {}  # Return an empty dictionary if user_id is invalid

    # Convert user-item matrix to a sparse tensor
    user_item_tensor = torch.tensor(user_item_matrix, dtype=torch.float32)
    user_item_sparse = user_item_tensor.to_sparse()
    print(f"Converted user-item matrix to sparse tensor. Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    # Process predictions in batches
    batch_size = 1000  # Adjust batch size based on available memory
    item_scores = {}
    for start_idx in range(0, num_items, batch_size):
        end_idx = min(start_idx + batch_size, num_items)
        item_batch = torch.arange(start_idx, end_idx, dtype=torch.long)

        print(f"Processing batch {start_idx}-{end_idx}. Memory usage: {process.memory_info().rss / 1024 ** 2:.2f} MB")

        with torch.no_grad():
            predictions = model(torch.tensor([user_id] * len(item_batch), dtype=torch.long), item_batch)
            for idx, score in zip(item_batch, predictions):
                item_id = matrix_index_to_item_id.get(idx.item()) if matrix_index_to_item_id else idx.item()
                item_scores[item_id] = score.item()

    # Sort and return top_k items
    top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"Top {top_k} items: {top_items}")
    print(f"Memory usage at end: {process.memory_info().rss / 1024 ** 2:.2f} MB")
    return dict(top_items)

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