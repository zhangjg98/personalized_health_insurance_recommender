import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from ml_model import generate_embeddings  # Import content-based embedding generator
import psutil  # Import psutil for memory monitoring
import pandas as pd  # Import pandas for DataFrame handling
import os  # Import os for process management
from models import NeuralCollaborativeFiltering  # Import from models.py
from resource_manager import get_user_item_matrix, get_ncf_model  # Import cached resources

USER_ITEM_MATRIX = get_user_item_matrix()
NCF_MODEL = get_ncf_model()

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

def predict_user_item_interactions(model=None, user_item_matrix=None, user_id=None, top_k=5, matrix_index_to_item_id=None):
    """
    Predict user-item interactions using the NCF model.
    """
    print("Starting predict_user_item_interactions function...")  # Debugging log

    # Use the globally cached USER_ITEM_MATRIX and NCF_MODEL if not provided
    user_item_matrix = user_item_matrix or USER_ITEM_MATRIX
    model = model or NCF_MODEL

    if user_item_matrix is None or model is None:
        print("Error: USER_ITEM_MATRIX or NCF_MODEL is not loaded.")
        return {}

    print("Starting predict_user_item_interactions function...")  # Keep concise logs
    process = psutil.Process(os.getpid())
    print(f"Memory usage at start: {process.memory_info().rss / 1024 ** 2:.2f} MB")

    if isinstance(user_item_matrix, pd.DataFrame):
        user_item_matrix = user_item_matrix.values  # Convert once

    num_users, num_items = user_item_matrix.shape
    print(f"User-item matrix dimensions: {num_users} users, {num_items} items")

    # Validate user_id
    if user_id is None or user_id < 0 or user_id >= num_users:
        print(f"Invalid or missing user_id: {user_id}. Skipping collaborative filtering.")
        return {}

    # Process predictions in batches
    batch_size = 1000  # Adjust batch size based on available memory
    item_scores = {}
    for start_idx in range(0, num_items, batch_size):
        end_idx = min(start_idx + batch_size, num_items)
        item_batch = torch.arange(start_idx, end_idx, dtype=torch.long)

        with torch.no_grad():
            predictions = model(torch.tensor([user_id] * len(item_batch), dtype=torch.long), item_batch)
            for idx, score in zip(item_batch, predictions):
                item_id = matrix_index_to_item_id.get(idx.item()) if matrix_index_to_item_id else idx.item()
                item_scores[item_id] = score.item()

    # Sort and return top_k items
    top_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    print(f"Top {top_k} items retrieved.")
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