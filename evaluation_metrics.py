import torch
import numpy as np
from neural_collaborative_filtering import precision_at_k, recall_at_k, ndcg_at_k, hit_rate
from utils import filter_irrelevant_plans  # Import the filtering function
from plans import PLANS

def evaluate_model_metrics(model, user_item_matrix, k=5, user_inputs=None, plans=None):
    """
    Evaluate the model using Precision@K, Recall@K, NDCG@K, and Hit Rate.

    Parameters:
        model (NeuralCollaborativeFiltering): Trained NCF model.
        user_item_matrix (numpy.ndarray): User-item interaction matrix.
        k (int): Number of top recommendations to consider.
        user_inputs (dict): User inputs for filtering irrelevant plans.
        plans (dict): List of plans for filtering.

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

        # Apply filtering logic to remove irrelevant plans
        if user_inputs and plans:
            # Convert plans dictionary to a list of plans
            plans_list = list(PLANS.values())
            predicted_items = [{"id": i + 1, "description": plans_list[i]["description"]} for i in range(num_items)]
            filtered_items = filter_irrelevant_plans(predicted_items, user_inputs)
            filtered_indices = [item["id"] - 1 for item in filtered_items]  # Adjust indices to be 0-based

            # Debugging logs
            print(f"Original predictions shape: {predictions.shape}")
            print(f"Original ground_truth shape: {ground_truth.shape}")
            print(f"Number of filtered items: {len(filtered_items)}")
            print(f"Filtered indices: {filtered_indices}")

            # Check if filtered_indices is empty
            if not filtered_indices:
                print("Warning: filtered_indices is empty. Skipping filtering.")
                continue  # Skip to the next user if no plans are relevant

            # Filter predictions and ground truth based on filtered indices
            predictions = predictions[filtered_indices]
            ground_truth = ground_truth[filtered_indices]

            # Debugging logs after filtering
            print(f"Predictions shape after filtering: {predictions.shape}")
            print(f"Ground_truth shape after filtering: {ground_truth.shape}")

        # Check if ground_truth is empty after filtering
        if np.sum(ground_truth) == 0:
            print("Warning: ground_truth is all zeros after filtering. Skipping user.")
            continue

        # Debugging logs for relevant and recommended items
        relevant_items = set(np.where(ground_truth > 0)[0])
        top_k_preds = np.argsort(predictions)[-k:][::-1]
        recommended_items = set(top_k_preds)
        print(f"Relevant items (indices): {relevant_items}")
        print(f"Recommended items (indices): {recommended_items}")
        print(f"k value: {k}")
        print(f"Number of relevant items: {len(relevant_items)}")

        precision_scores.append(precision_at_k(predictions, ground_truth, k))
        recall_scores.append(recall_at_k(predictions, ground_truth, k))
        ndcg_scores.append(ndcg_at_k(predictions, ground_truth, k))
        hit_rates.append(hit_rate(predictions, ground_truth, k))

    return {
        "Precision@K": np.mean(precision_scores) if precision_scores else 0.0,
        "Recall@K": np.mean(recall_scores) if recall_scores else 0.0,
        "NDCG@K": np.mean(ndcg_scores) if ndcg_scores else 0.0,
        "Hit Rate": np.mean(hit_rates) if hit_rates else 0.0
    }
