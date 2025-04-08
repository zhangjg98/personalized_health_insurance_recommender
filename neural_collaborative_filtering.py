import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn.metrics import mean_squared_error, f1_score
import shap
import numpy as np

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

def load_ncf_model(model_path="ncf_model.pth", num_users=None, num_items=None, latent_dim=50, hidden_dim=128):
    """
    Load the Neural Collaborative Filtering model from a saved checkpoint.
    """
    print(f"Loading NCF model from {model_path}...")  # Debugging log

    # Validate num_users and num_items
    if num_users is None or num_items is None or num_users < 1 or num_items < 1:
        print("Invalid dimensions for user-item matrix. Using placeholder dimensions.")  # Debugging log
        num_users, num_items = 1, 7  # Placeholder dimensions

    # Initialize the model with the current dimensions
    model = NeuralCollaborativeFiltering(num_users=num_users, num_items=num_items, latent_dim=latent_dim, hidden_dim=hidden_dim)

    try:
        # Attempt to load the saved model
        state_dict = torch.load(model_path)
        if state_dict["user_embedding.weight"].shape[0] != num_users or state_dict["item_embedding.weight"].shape[0] != num_items:
            raise RuntimeError(
                f"Model dimensions do not match the current user-item matrix. "
                f"Expected ({num_users}, {num_items}), but got "
                f"({state_dict['user_embedding.weight'].shape[0]}, {state_dict['item_embedding.weight'].shape[0]})."
            )
        model.load_state_dict(state_dict)
        print("Model loaded successfully.")  # Debugging log
    except RuntimeError as e:
        # Handle dimension mismatch errors
        print(f"Model dimensions do not match the current user-item matrix. Retraining is required. Error: {e}")  # Debugging log
        raise RuntimeError(
            f"Model dimensions do not match the current user-item matrix. Retrain the model to fix this issue. Error: {e}"
        )

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

def explain_ncf_predictions(model, user_item_matrix, user_id, item_index, top_n=2):
    """
    Use SHAP to explain the predictions of the Neural Collaborative Filtering model.

    Parameters:
        model (NeuralCollaborativeFiltering): Trained NCF model.
        user_item_matrix (numpy.ndarray or DataFrame): User-item interaction matrix.
        user_id (int): User ID for the explanation.
        item_index (int): Column index of the item in the user-item matrix.
        top_n (int): Number of top features to include in the explanation.

    Returns:
        dict: SHAP explanation with top features and their impacts.
    """
    # Ensure item_index is valid
    if item_index is None or item_index < 0 or item_index >= user_item_matrix.shape[1]:
        print(f"Invalid item_index: {item_index}. Skipping SHAP explanation.")
        return {"top_features": [], "explanation": "Invalid item_index for SHAP explanation."}

    print(f"Generating SHAP values for user_id {user_id}, item_index {item_index}")  # Debugging log
    try:
        # Convert user_item_matrix to a NumPy array if it's a DataFrame
        if isinstance(user_item_matrix, pd.DataFrame):
            feature_names = user_item_matrix.columns.tolist()  # Use column names as feature names
            user_item_matrix = user_item_matrix.values
        else:
            feature_names = [f"Plan ID {i}" for i in range(user_item_matrix.shape[1])]  # Fallback feature names

        # Debugging log: Print feature names and matrix dimensions
        print(f"Feature names: {feature_names}")
        print(f"User-item matrix dimensions: {user_item_matrix.shape}")

        # Prepare SHAP input data
        input_data = np.array([[user_id, item_index]])  # SHAP expects user and item indices as input
        print(f"SHAP input data (shape: {input_data.shape}): {input_data}")  # Debugging log

        # Use all user-item pairs as representative data
        representative_data = np.array([
            [user_id, i] for i in range(user_item_matrix.shape[1])
        ])
        print(f"SHAP representative data (shape: {representative_data.shape}): {representative_data}")  # Debugging log

        # Define a SHAP explainer
        explainer = shap.Explainer(lambda x: model(torch.tensor(x[:, 0], dtype=torch.long), torch.tensor(x[:, 1], dtype=torch.long)).detach().numpy(), representative_data)
        print("SHAP explainer initialized.")  # Debugging log

        # Compute SHAP values
        shap_values = explainer(input_data)
        if shap_values is None or shap_values.values is None:
            print("SHAP values are None. Explanation cannot be generated.")  # Debugging log
            return {"top_features": [], "explanation": "SHAP explanation could not be generated."}

        # Convert SHAP values to a NumPy array
        shap_values = np.array(shap_values.values)
        print(f"SHAP values (shape: {shap_values.shape if hasattr(shap_values, 'shape') else 'unknown'}): {shap_values}")  # Debugging log

        # Validate SHAP values dimensions
        if shap_values.ndim != 2 or shap_values.shape[1] != len(feature_names):
            print(f"SHAP values have incorrect dimensions: {shap_values.shape}. Expected: (1, {len(feature_names)})")  # Debugging log
            return {"top_features": [], "explanation": "SHAP explanation could not be generated due to dimension mismatch."}

        # Extract top features
        top_features = sorted(
            enumerate(shap_values[0]), key=lambda x: abs(x[1]), reverse=True
        )[:top_n]  # Top `n` features

        explanation = []
        for i, impact in top_features:
            feature_name = feature_names[i]
            explanation.append({
                "feature": feature_name,
                "impact": round(float(impact), 4),
                "description": f"The feature '{feature_name}' contributed {round(float(impact), 4)} to this recommendation."
            })

        return {"top_features": explanation, "explanation": "These features had the highest impact on the recommendation."}
    except Exception as e:
        print(f"Error generating SHAP explanation for item_index {item_index}: {e}")  # Debugging log
        return {"top_features": [], "explanation": "Error occurred while generating SHAP explanation."}
